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

st.set_page_config(page_title="ibookle 童書專家", layout="wide", initial_sidebar_state="collapsed")

if "session_id" not in st.session_state: 
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "last_row_idx" not in st.session_state:
    st.session_state.last_row_idx = None
if "prev_query" not in st.session_state:
    st.session_state.prev_query = ""

if "GOOGLE_API_KEY" in st.secrets:
    # 這裡維持您原本的 genai.Client 語法
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    client = None

# ================= 2. 核心函式定義 =================

def get_google_sheet():
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
    try:
        sheet = get_google_sheet()
        if sheet:
            tw_tz = pytz.timezone('Asia/Taipei')
            now_tw = datetime.datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M:%S")
            new_row = [now_tw, st.session_state.session_id, user_input, ai_response, recommended_books, ""]
            sheet.append_row(new_row)
            return len(sheet.get_all_values())
        return None
    except:
        return None

def update_log_feedback():
    row_idx = st.session_state.last_row_idx
    fb_key = f"fb_key_{row_idx}"
    if row_idx and fb_key in st.session_state:
        score = st.session_state[fb_key]
        if score is not None:
            try:
                sheet = get_google_sheet()
                feedback_text = "👍" if score == 1 else "👎"
                sheet.update_cell(row_idx, 6, feedback_text)
                if score == 1:
                    st.toast("感謝您的鼓勵！🌟", icon="❤️")
                else:
                    st.toast("感謝您的回饋。", icon="📝")
            except:
                pass

def get_recommendations(user_query):
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        pinecone_key = st.secrets["PINECONE_API_KEY"]
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key, 
            task_type="retrieval_query"
        )
        class DimensionFixer:
            def __init__(self, model): self.model = model
            def embed_query(self, text): return self.model.embed_query(text)[:768]
            def embed_documents(self, texts): return [v[:768] for v in self.model.embed_documents(texts)]
        
        fixed_embeddings = DimensionFixer(embeddings_model)
        vectorstore = PineconeVectorStore(index_name="gemini768", embedding=fixed_embeddings, pinecone_api_key=pinecone_key)

        vague_keywords = ["推薦", "好書", "小學生", "繪本", "有什麼書", "介紹", "童書", "閱讀"]
        is_vague = len(user_query.strip()) <= 4 or user_query.strip() in vague_keywords

        if is_vague:
            raw_results = vectorstore.similarity_search(user_query, k=50)
            candidate_books = [{"doc": d, "rating": float(d.metadata.get('Expert_Rating', 0))} for d in raw_results]
            candidate_books.sort(key=lambda x: x['rating'], reverse=True)
            return [item["doc"] for item in candidate_books[:5]], True
        else:
            search_results = vectorstore.similarity_search_with_score(user_query, k=15)
            candidate_books = [{"doc": doc, "rating": float(doc.metadata.get('Expert_Rating', 0)), "score": score} for doc, score in search_results]
            candidate_books.sort(key=lambda x: (x['rating'], x['score']), reverse=True)
            return [item["doc"] for item in candidate_books[:5]], False
    except Exception as e:
        st.error(f"檢索系統異常: {e}")
        return None, False

# ================= 3. UI 介面樣式 =================

st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    div[data-testid="stStatusWidget"], .stAppViewFooter, [data-testid="stDecoration"], [data-testid="stHeader"] { display: none !important; }
    button[title="View fullscreen"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] {
        background-color: #E67E22 !important; border-radius: 50% !important;
        width: 40px !important; height: 40px !important; left: 15px !important; top: 15px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important; display: flex !important; align-items: center !important; justify-content: center !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg { fill: white !important; transform: scale(1.2); }
    .stTextInput input:focus { border-color: #E67E22 !important; box-shadow: 0 0 0 1px #E67E22 !important; outline: none !important; }
    .expert-suggestion-text { margin: 20px 0; line-height: 1.8; color: #34495E; font-size: 1.05rem; }
    [data-testid="stFeedbackAdmonition"] { background-color: transparent !important; border: none !important; box-shadow: none !important; }
    .stTextInput input { border: 2px solid #E67E22 !important; border-radius: 25px !important; }
    .feedback-container { padding: 10px 0; text-align: center; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# 側邊欄：計次、燈號、分享與問卷
with st.sidebar:
    st.header("📊 ibookle 統計")
    total_answers = "---"
    system_status = "🔴 系統連線中..."
    sheet_data = get_google_sheet()
    if sheet_data:
        try:
            total_answers = len(sheet_data.get_all_values()) - 1
            system_status = "🟢 系統正常運作"
        except: system_status = "🟡 系統忙碌中"
    
    st.metric("已解答家長疑問", f"{total_answers} 次")
    st.write(system_status)
    st.divider()

    # --- 常駐分享功能 ---
    st.subheader("📤 儲存與分享報告")
    has_res = st.session_state.search_results is not None
    
    if has_res:
        res = st.session_state.search_results
        share_content = f"🌟 ibookle 專家選書報告 🌟\n📅 日期：{datetime.date.today().strftime('%Y-%m-%d')}\n🔍 需求：{st.session_state.prev_query}\n\n💡 專家建議：\n{res['ai_response']}\n\n📚 書單：\n"
        for i, book in enumerate(res["books"], 1):
            share_content += f"{i}. 《{book['Title']}》 (⭐{book['Rating']})\n   🔗 {book['Link']}\n\n"
        share_content += "--- 分享自 ibookle AI ---"
    else:
        share_content = "尚未生成報告，請先進行諮詢。"

    if st.button("📋 生成分享文字 (Line/FB)", disabled=not has_res, use_container_width=True):
        st.code(share_content, language=None)
        st.toast("報告已生成！", icon="✨")

    st.download_button(
        label="📄 下載建議報告 (.txt)",
        data=share_content,
        file_name=f"ibookle_report.txt",
        mime="text/plain",
        use_container_width=True,
        disabled=not has_res
    )
    st.divider()

    # --- 常駐滿意度回饋 ---
    st.subheader("🌟 滿意度回饋")
    if st.session_state.last_row_idx:
        fb_key = f"fb_key_{st.session_state.last_row_idx}"
        st.feedback("thumbs", key=fb_key, on_change=update_log_feedback)
        st.caption("您的回饋能讓專家建議更準確")
    else:
        st.caption("諮詢後即可在此提供回饋")
    
    st.divider()
    st.subheader("📢 意見回饋")
    st.link_button("📝 填寫使用問卷", "https://your-google-form-link", use_container_width=True)
    st.caption("© 2026 ibookle")

# 主頁面
st.title("💡 ibookle 童書共讀專家")
st.markdown("##### *為每一本好書，找到懂它的家長；為每一個孩子，挑選最好的陪伴。*")
st.write("你好！我是你的共讀專家。輸入孩子的狀況或想找的主題，我會為你挑選最適合的童書。")

user_query = st.text_input("", placeholder="🔍 例如：想找關於克服分離焦慮的童書...", key="main_search")

# ================= 4. 搜尋與生成邏輯 =================

if user_query and (not st.session_state.search_results or st.session_state.get("prev_query") != user_query):
    with st.spinner("🔍 正在為您翻閱書櫃並整理建議..."):
        results, is_vague_mode = get_recommendations(user_query)
        
        if results:
            book_titles = [d.metadata.get('Title','未知') for d in results]
            titles_str = ", ".join(book_titles)
            
            if is_vague_mode:
                prompt = f"使用者問了一個模糊的問題：\"{user_query}\"\n我們目前挑選了專家評分最高(三星)的經典書：{titles_str}\n\n請以 ibookle 專家身份回覆：\n1. 開頭請說「您好！」(禁止說家長您好)。\n2. 說明這個問題範圍較廣，因此您先準備了幾本「絕對不容錯過的專家首選」。\n3. 溫柔地詢問更多細節（如：孩子的年級、興趣、或特定的困擾）。\n4. 語氣親切，約 150 字，禁止使用表情符號。"
            else:
                prompt = f"使用者需求：{user_query}\n相關精選童書：{titles_str}\n\n請以 ibookle 專家身份回覆：\n1. 開頭請說「您好！」(禁止說家長您好)。\n2. 簡述為什麼這幾本書適合目前的提問情境。\n3. 提到這些書是經過專家深度導讀後的精選建議。\n4. 語氣親切專業，約 150 字，禁止使用表情符號。"
            
            try:
                # 這裡若失敗會顯示具體原因
                response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
                ai_response = response.text
                
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
                st.rerun() 
            except Exception as e:
                st.error(f"AI 專家目前連線不穩。錯誤原因: {e}")

# ================= 5. 結果顯示 =================

if st.session_state.search_results:
    res = st.session_state.search_results
    st.markdown(f'<div class="expert-suggestion-text"><b>🤖 專家建議：</b><br>{res["ai_response"]}</div>', unsafe_allow_html=True)
    
    st.markdown("### 📖 精選推薦清單")
    for b in res["books"]:
        with st.container():
            header_text = f"《{b['Title']}》" + (" ✨ [專家首選]" if float(b['Rating']) >= 3.0 else "")
            st.subheader(header_text)
            st.caption(f"✍️ 作者：{b['Author']} | 🏷️ 分類：{b['Category']} | ⭐ 推薦指數：{b['Rating']}")
            if b['Quick_Summary']: st.info(b['Quick_Summary'])
            with st.expander("🔍 點擊查看專家深度導讀"): st.markdown(b['Refine_Content'])
            if b['Link']: st.link_button(f"🛒 前往購買", b['Link'], use_container_width=True)
        st.divider()

    with st.expander("🔒 進階功能 (Pro 版預覽)"):
        st.write("✨ **一鍵加入圖書館借閱清單**")
        st.write("✨ **同步至 Notion/Evernote 閱讀筆記**")
else:
    st.markdown("---")
    st.caption("👋 歡迎使用 ibookle！請描述孩子目前的狀況，讓專家為您挑選適合的童書。")

st.caption("© 2026 ibookle - 讓每一段共讀時光都更有意義")