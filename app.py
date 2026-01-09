import streamlit as st
import json, os, datetime, gspread, uuid, pytz
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ================= 1. 初始化與環境配置 =================
load_dotenv()

# 設定頁面屬性
st.set_page_config(page_title="ibookle 童書專家", layout="wide", initial_sidebar_state="collapsed")

# 初始化 Session State
if "session_id" not in st.session_state: 
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "last_row_idx" not in st.session_state:
    st.session_state.last_row_idx = None

# 初始化 AI Client
if "GOOGLE_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    client = None

# ================= 2. 核心函式定義 =================

def get_google_sheet():
    """穩定連線 Google Sheets"""
    try:
        raw_json = st.secrets["GOOGLE_CREDENTIALS"]
        creds_info = json.loads(raw_json.strip(), strict=False)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
        client_gs = gspread.authorize(creds)
        return client_gs.open("AI_User_Logs").worksheet("Brief_Logs")
    except:
        return None

def save_to_log(user_input, ai_response, recommended_books):
    """依照後台欄位對齊：Time, SessionID, Input, AI, Books, Feedback"""
    try:
        sheet = get_google_sheet()
        if sheet:
            tw_tz = pytz.timezone('Asia/Taipei')
            now_tw = datetime.datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M:%S")
            # 寫入新紀錄，Feedback 欄位(第6欄)預設為空
            new_row = [now_tw, st.session_state.session_id, user_input, ai_response, recommended_books, ""]
            sheet.append_row(new_row)
            return len(sheet.get_all_values())
        return None
    except:
        return None

def update_log_feedback():
    """處理 👍/👎 回饋並觸發感謝彈窗"""
    row_idx = st.session_state.last_row_idx
    fb_key = f"fb_key_{row_idx}"
    if row_idx and fb_key in st.session_state:
        score = st.session_state[fb_key]
        if score is not None:
            try:
                sheet = get_google_sheet()
                feedback_text = "👍" if score == 1 else "👎"
                # 更新試算表第 6 欄
                sheet.update_cell(row_idx, 6, feedback_text)
                
                # 手機版即時感謝通知
                if score == 1:
                    st.toast("感謝您的鼓勵！我們會繼續為您挑選好書。🌟", icon="❤️")
                else:
                    st.toast("感謝您的回饋，我們會持續進步。", icon="📝")
            except:
                pass

def get_recommendations(user_query):
    """
    優化後的雙軌檢索邏輯：
    1. 書名精準比對 (Exact Title Match)
    2. 條件式向量搜尋 (Metadata Filtering + Semantic Search)
    3. 注音彈性降級 (Pinyin Relaxation)
    """
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        pinecone_key = st.secrets["PINECONE_API_KEY"]
        
        # 初始化 Embedding
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key, 
            task_type="retrieval_query"
        )
        
        # 維度修正與 VectorStore 初始化
        class DimensionFixer:
            def __init__(self, model): self.model = model
            def embed_query(self, text): return self.model.embed_query(text)[:768]
            def embed_documents(self, texts): return [v[:768] for v in self.model.embed_documents(texts)]
        
        fixed_embeddings = DimensionFixer(embeddings_model)
        vectorstore = PineconeVectorStore(
            index_name="gemini768", 
            embedding=fixed_embeddings, 
            pinecone_api_key=pinecone_key
        )

        # --- 第一軌：意圖解析與書名精準比對 ---
        # 嘗試在標題欄位進行過濾（假設 Pinecone metadata 有 Title 欄位）
        exact_match_results = vectorstore.similarity_search(
            user_query, 
            k=2, 
            filter={"Title": {"$eq": user_query}}
        )

        # --- 第二軌：判定搜尋模式與初步過濾 ---
        vague_keywords = ["推薦", "好書", "小學生", "繪本", "有什麼書", "介紹", "童書", "閱讀"]
        is_vague = len(user_query.strip()) <= 4 or user_query.strip() in vague_keywords
        
        # 這裡我們可以先根據 user_query 解析出是否有「注音」關鍵字
        needs_pinyin = "注音" in user_query and "不" not in user_query
        
        final_candidates = []
        
        if is_vague:
            # 模糊模式：抓取高星等
            raw_results = vectorstore.similarity_search(user_query, k=20)
            for d in raw_results:
                final_candidates.append({
                    "doc": d,
                    "rating": float(d.metadata.get('Expert_Rating', 0)),
                    "has_pinyin": d.metadata.get('注音標籤') == "有注音"
                })
            final_candidates.sort(key=lambda x: x['rating'], reverse=True)
        else:
            # 明確模式：權重檢索
            # 先抓多一點 (k=15) 用於後續的注音篩選排序
            search_results = vectorstore.similarity_search_with_score(user_query, k=15)
            for doc, score in search_results:
                final_candidates.append({
                    "doc": doc,
                    "rating": float(doc.metadata.get('Expert_Rating', 0)),
                    "score": score,
                    "has_pinyin": doc.metadata.get('注音標籤') == "有注音"
                })
            
            # --- 邏輯：注音彈性排序 ---
            # 如果使用者要求注音，我們將「有注音」且「相關度高」的往前排
            if needs_pinyin:
                final_candidates.sort(key=lambda x: (x['has_pinyin'], x['score']), reverse=True)
            else:
                final_candidates.sort(key=lambda x: (x['rating'], x['score']), reverse=True)

        # --- 整合結果 ---
        # 1. 將書名精確符合的放在最前面
        final_docs = [doc for doc in exact_match_results]
        
        # 2. 補足後續結果 (去重)
        existing_titles = [d.metadata.get('Title') for d in final_docs]
        for item in final_candidates:
            if len(final_docs) >= 5: break
            if item["doc"].metadata.get('Title') not in existing_titles:
                final_docs.append(item["doc"])
                existing_titles.append(item["doc"].metadata.get('Title'))

        return final_docs, is_vague

    except Exception as e:
        st.error(f"檢索系統異常: {e}")
        return None, False

# ================= 3. UI 介面樣式 (視覺深度優化) =================

st.markdown("""
    <style>
    /* 隱藏預設元件 */
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    div[data-testid="stStatusWidget"], .stAppViewFooter, [data-testid="stDecoration"], [data-testid="stHeader"] { display: none !important; }
    button[title="View fullscreen"] { display: none !important; }

    /* 1. 側邊欄按鈕：橘色圓圈 + 白色反轉箭頭 (>>) */
    [data-testid="stSidebarCollapsedControl"] {
        background-color: #E67E22 !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        left: 15px !important;
        top: 15px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg {
        fill: white !important;
        transform: scale(1.2);
    }

    /* 2. 消除輸入框綠線：打字時保持橘色 */
    .stTextInput input:focus {
        border-color: #E67E22 !important;
        box-shadow: 0 0 0 1px #E67E22 !important;
        outline: none !important;
    }
    
    /* 3. 專家建議：簡單純文字 */
    .expert-suggestion-text {
        margin: 20px 0;
        line-height: 1.8;
        color: #34495E;
        font-size: 1.05rem;
    }

    /* 4. 移除問卷多餘灰色塊與陰影 */
    [data-testid="stFeedbackAdmonition"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    .feedback-container {
        padding: 10px 0;
        text-align: center;
        margin-top: 20px;
    }

    /* 基礎控制 */
    .stTextInput input { border: 2px solid #E67E22 !important; border-radius: 25px !important; }
    </style>
    """, unsafe_allow_html=True)

# 側邊欄：計次、燈號與問卷連結
with st.sidebar:
    st.header("📊 ibookle 統計")
    total_answers = "---"
    system_status = "🔴 系統連線中..."
    sheet_data = get_google_sheet()
    if sheet_data:
        try:
            total_answers = len(sheet_data.get_all_values()) - 1
            system_status = "🟢 系統正常運作"
        except:
            system_status = "🟡 系統忙碌中"
    
    st.metric("已解答家長疑問", f"{total_answers} 次")
    st.write(system_status)
    st.divider()
    
    # 側邊欄問卷連結區
    st.subheader("📢 意見回饋")
    st.write("您的建議是我們進步的動力")
    st.link_button("📝 填寫使用問卷", "https://childrening.pse.is/8jjrrl", use_container_width=True)
    
    st.divider()
    st.caption("© 2026 ibookle")

# 主頁面
st.title("💡 ibookle 童書共讀專家")
st.markdown("##### *為每一本好書，找到懂它的家長；為每一個孩子，挑選最好的陪伴。*")
st.write("你好！我是你的共讀專家。輸入孩子的狀況或想找的主題，我會為你挑選最適合的童書。")

user_query = st.text_input("", placeholder="🔍 想找關於天氣的知識書，或是適合小學生的奇幻小說...", key="main_search")

# ================= 4. 搜尋與生成邏輯 (稱謂修正與 Prompt 優化) =================

if user_query and (not st.session_state.search_results or st.session_state.get("prev_query") != user_query):
    with st.spinner("🔍 正在為您翻閱書櫃並整理建議..."):
        results, is_vague_mode = get_recommendations(user_query)
        
        if results:
            book_titles = [d.metadata.get('Title','未知') for d in results]
            titles_str = ", ".join(book_titles)
            
            # 根據模式切換 Prompt
            if is_vague_mode:
                prompt = f"""
                使用者問了一個模糊的問題："{user_query}"
                我們目前挑選了專家評分最高(三星)的經典書：{titles_str}
                
                請以 ibookle 專家身份回覆：
                1. 開頭請說「您好！」(禁止說家長您好)。
                2. 說明這個問題範圍較廣，因此您先準備了幾本「絕對不容錯過的專家首選」。
                3. 溫柔地詢問更多細節（如：孩子的年級、興趣、或特定的困擾）。
                4. 語氣親切，約 150 字，禁止使用表情符號。
                """
            else:
                prompt = f"""
                使用者需求：{user_query}
                相關精選童書：{titles_str}
                
                請以 ibookle 專家身份回覆：
                1. 開頭請說「您好！」(禁止說家長您好)。
                2. 簡述為什麼這幾本書適合目前的提問情境。
                3. 提到這些書是經過專家深度導讀後的精選建議。
                4. 語氣親切專業，約 150 字，禁止使用表情符號。
                """
            
            try:
                response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
                ai_response = response.text
                
                # 存入 Session State (包含 Rating 資訊)
                st.session_state.search_results = {
                    "ai_response": ai_response, 
                    "books": [{
                        "Title": d.metadata.get('Title', '未知'), 
                        "Author": d.metadata.get('Author', '未知'), 
                        "Illustrator": d.metadata.get('Illustrator', '未知'), 
                        "Category": d.metadata.get('Category', '一般'),
                        "Quick_Summary": d.metadata.get('Quick_Summary', ''), 
                        "Refine_Content": d.metadata.get('Refine_Content', '暫無導讀'), 
                        "Link": d.metadata.get('Link', ''),
                        "Rating": d.metadata.get('Expert_Rating', 0)
                    } for d in results]
                }
                st.session_state.prev_query = user_query
                st.session_state.last_row_idx = save_to_log(user_query, ai_response, titles_str)
            except:
                st.error("AI 專家目前連線不穩，請稍候。")

# ================= 5. 結果顯示 (加入專家推薦標籤) =================

if st.session_state.search_results:
    res = st.session_state.search_results
    st.markdown(f'<div class="expert-suggestion-text"><b>🤖 專家建議：</b><br>{res["ai_response"]}</div>', unsafe_allow_html=True)
    
    st.markdown("### 📖 精選推薦清單")
    for b in res["books"]:
        with st.container():
            # 修改標題，如果星等為 3.0，加上特別標記
            header_text = f"《{b['Title']}》"
            if float(b['Rating']) >= 3.0:
                header_text += " ✨ [專家首選]"
            
            st.subheader(header_text)
            st.caption(f"✍️ 作者：{b['Author']} | 🏷️ 分類：{b['Category']} | ⭐ 推薦指數：{b['Rating']}")
            
            if b['Quick_Summary']: 
                st.info(b['Quick_Summary'])
                
            with st.expander("🔍 點擊查看專家深度導讀"):
                st.markdown(b['Refine_Content'])
            
            if b['Link']: 
                st.link_button(f"🛒 前往購買《{b['Title']}》", b['Link'], use_container_width=True)
        
        st.divider()

import datetime

# 檢查是否有搜尋結果
if "search_results" in st.session_state and st.session_state.search_results:
    res = st.session_state.search_results
    
    st.divider() # 視覺分割線
    
    # --- 建立分享內容字串 ---
    # 1. 標題與 AI 的總結建議
    share_content = f"🌟 ibookle 專家選書報告 🌟\n"
    share_content += f"📅 日期：{datetime.date.today().strftime('%Y-%m-%d')}\n"
    share_content += f"🔍 您諮詢的需求：{user_query}\n\n"
    share_content += f"💡 專家分析建議：\n{res['ai_response']}\n\n"
    share_content += f"📚 精選推薦書單：\n"
    
    # 2. 迭代書籍清單
    for i, book in enumerate(res["books"], 1):
        share_content += f"{i}. 《{book['Title']}》\n"
        share_content += f"   ⭐ 專家評分：{book['Rating']} / 3.0\n"
        share_content += f"   📌 專業導讀：{book['Quick_Summary']}\n"
        share_content += f"   🔗 連結：{book['Link']}\n\n"
    
    share_content += f"--- 分享自 ibookle AI 專家導讀系統 ---"

    # --- 顯示分享功能區塊 ---
    st.subheader("📤 儲存與分享本次報告")
    
    col_copy, col_dl = st.columns(2)
    
    with col_copy:
        # 使用 st.code 讓使用者容易點擊複製，或用按鈕觸發 toast
        if st.button("📋 生成分享文字 (Line/FB)"):
            st.info("下方文字已準備好，您可以直接長按複製並分享！")
            st.code(share_content, language=None)
            st.toast("報告已生成，準備好分享囉！", icon="✨")

    with col_dl:
        # 提供下載功能，讓家長存檔
        st.download_button(
            label="📄 下載書單文字檔 (.txt)",
            data=share_content,
            file_name=f"ibookle_report_{datetime.date.today().strftime('%m%d')}.txt",
            mime="text/plain",
            help="將整份專家建議存成純文字檔，方便日後查看"
        )

    # 預留 Pro 版功能預覽 (增加計畫書說服力)
    with st.expander("🔒 進階功能預覽（製作中）"):
        st.write("✨ **加入圖書館借閱清單**")
        st.write("✨ **加入自訂書單**")
        st.write("✨ **生成閱讀分析報告**")


# ... (後續回饋與 footer 保持不變)

    # 問卷回饋區 (透明背景)
    if st.session_state.last_row_idx:
        fb_key = f"fb_key_{st.session_state.last_row_idx}"
        st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
        if fb_key not in st.session_state or st.session_state[fb_key] is None:
            st.write("🌟 這份建議對您有幫助嗎？")
        else:
            st.write("✅ 感謝您的回饋，讓 ibookle 變得更好！")
        st.feedback("thumbs", key=fb_key, on_change=update_log_feedback)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("---")
    st.caption("👋 歡迎使用 ibookle！請描述孩子目前的狀況，讓專家為您挑選適合的童書。")

st.caption("© 2026 ibookle - 讓每一段共讀時光都更有意義")