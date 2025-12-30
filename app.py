import streamlit as st
import json, os, datetime, gspread, uuid, pytz, io
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from PIL import Image, ImageDraw, ImageFont

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
                sheet.update_cell(row_idx, 6, feedback_text)
                if score == 1:
                    st.toast("感謝您的鼓勵！我們會繼續為您挑選好書。🌟", icon="❤️")
                else:
                    st.toast("感謝您的回饋，我們會持續進步。", icon="📝")
            except:
                pass

def get_recommendations(user_query):
    """雙層邏輯與原始搜尋演算法"""
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

def generate_share_image(query, ai_res, books):
    """繪製專家報告圖片"""
    width = 800
    item_h = 160
    header_h = 280
    footer_h = 80
    total_h = header_h + (len(books) * item_h) + footer_h
    img = Image.new('RGB', (width, total_h), color=(252, 251, 247))
    draw = ImageDraw.Draw(img)
    try:
        f_title = ImageFont.load_default()
        f_text = ImageFont.load_default()
    except:
        f_title = f_text = ImageFont.load_default()
    
    draw.rectangle([0, 0, width, 180], fill=(230, 126, 34))
    draw.text((40, 50), "ibookle 專家選書報告", fill=(255, 255, 255), font=f_title)
    draw.text((40, 110), f"諮詢需求：{query[:30]}", fill=(255, 255, 255), font=f_text)
    
    y = 200
    draw.text((40, y), "💡 專家分析建議：", fill=(230, 126, 34), font=f_text)
    y += 40
    ai_lines = [ai_res[i:i+40] for i in range(0, min(len(ai_res), 160), 40)]
    for line in ai_lines:
        draw.text((40, y), line, fill=(52, 73, 94), font=f_text)
        y += 25
    
    y += 20
    for b in books:
        draw.rectangle([30, y, 770, y+140], fill=(255, 255, 255), outline=(236, 240, 241), width=2)
        draw.text((50, y+25), f"《{b['Title']}》", fill=(44, 62, 80), font=f_text)
        draw.text((50, y+65), f"⭐ 專家評分：{b['Rating']} / 3.0", fill=(241, 196, 15), font=f_text)
        draw.text((50, y+95), f"推薦理由：{b['Quick_Summary'][:35]}...", fill=(127, 140, 141), font=f_text)
        y += 160
    
    draw.rectangle([0, total_h-footer_h, width, total_h], fill=(44, 62, 80))
    draw.text((220, total_h-50), "© 2026 ibookle 專業 AI 導讀系統 - 嚴禁翻拷", fill=(255, 255, 255), font=f_text)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ================= 3. UI 介面樣式 =================

st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    div[data-testid="stStatusWidget"], .stAppViewFooter, [data-testid="stDecoration"], [data-testid="stHeader"] { display: none !important; }
    
    [data-testid="stSidebarCollapsedControl"] {
        background-color: #E67E22 !important; border-radius: 50% !important;
        width: 40px !important; height: 40px !important; left: 15px !important; top: 15px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important; display: flex !important; align-items: center !important; justify-content: center !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg { fill: white !important; transform: scale(1.2); }

    .stTextInput input:focus { border-color: #E67E22 !important; box-shadow: 0 0 0 1px #E67E22 !important; outline: none !important; }
    .expert-suggestion-text { margin: 20px 0; line-height: 1.8; color: #34495E; font-size: 1.05rem; }
    .feedback-container { padding: 10px 0; text-align: center; margin-top: 20px; }
    .stTextInput input { border: 2px solid #E67E22 !important; border-radius: 25px !important; }

    /* 浮動分享按鈕 */
    .float-share-btn {
        position: fixed; bottom: 30px; right: 30px;
        background-color: #E67E22; color: white !important;
        padding: 12px 24px; border-radius: 30px;
        text-decoration: none !important; font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); z-index: 9999;
        display: flex; align-items: center; transition: 0.3s;
    }
    .float-share-btn:hover { transform: scale(1.05); background-color: #D35400; }
    </style>
    """, unsafe_allow_html=True)

# 側邊欄
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
    st.subheader("📢 意見回饋")
    st.link_button("📝 填寫使用問卷", "https://your-google-form-link", use_container_width=True)
    st.caption("© 2026 ibookle")

# 主頁面
st.title("💡 ibookle 童書共讀專家")
st.markdown("##### *為每一本好書，找到懂它的家長；為每一個孩子，挑選最好的陪伴。*")
user_query = st.text_input("", placeholder="🔍 例如：想找關於克服分離焦慮的童書...", key="main_search")

# ================= 4. 搜尋與生成邏輯 (原始 Prompt 完全保留) =================

if user_query and (not st.session_state.search_results or st.session_state.get("prev_query") != user_query):
    with st.spinner("🔍 正在為您翻閱書櫃並整理建議..."):
        results, is_vague_mode = get_recommendations(user_query)
        if results:
            book_titles = [d.metadata.get('Title','未知') for d in results]
            titles_str = ", ".join(book_titles)
            
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
                st.session_state.search_results = {
                    "ai_response": ai_response, 
                    "books": [{
                        "Title": d.metadata.get('Title', '未知'), "Author": d.metadata.get('Author', '未知'), 
                        "Category": d.metadata.get('Category', '一般'), "Quick_Summary": d.metadata.get('Quick_Summary', ''), 
                        "Refine_Content": d.metadata.get('Refine_Content', '暫無導讀'), "Link": d.metadata.get('Link', ''),
                        "Rating": d.metadata.get('Expert_Rating', 0)
                    } for d in results]
                }
                st.session_state.prev_query = user_query
                st.session_state.last_row_idx = save_to_log(user_query, ai_response, titles_str)
            except: st.error("AI 專家連線不穩。")

# ================= 5. 結果顯示與分享區 =================

if st.session_state.search_results:
    # 浮動分享按鈕
    st.markdown('<a class="float-share-btn" href="#share_zone">📤 分享報告</a>', unsafe_allow_html=True)
    
    res = st.session_state.search_results
    st.markdown(f'<div class="expert-suggestion-text"><b>🤖 專家建議：</b><br>{res["ai_response"]}</div>', unsafe_allow_html=True)
    
    st.markdown("### 📖 精選推薦清單")
    for b in res["books"]:
        with st.container():
            header_text = f"《{b['Title']}》" + (" ✨ [專家首選]" if float(b['Rating']) >= 3.0 else "")
            st.subheader(header_text)
            st.caption(f"✍️ 作者：{b['Author']} | ⭐ 推薦指數：{b['Rating']}")
            if b['Quick_Summary']: st.info(b['Quick_Summary'])
            with st.expander("🔍 專家深度導讀"): st.markdown(b['Refine_Content'])
            if b['Link']: st.link_button(f"🛒 前往購買", b['Link'], use_container_width=True)
        st.divider()

    # --- 圖片與文字分享區 (帶錨點) ---
    st.markdown('<div id="share_zone"></div>', unsafe_allow_html=True)
    st.subheader("📤 儲存與分享本次報告")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🖼️ 生成圖片報告 (含版權)", use_container_width=True):
            with st.spinner("製作中..."):
                img_data = generate_share_image(st.session_state.prev_query, res["ai_response"], res["books"])
                st.image(img_data)
                st.download_button("💾 下載圖片報告", img_data, "ibookle_report.png", "image/png", use_container_width=True)
    with c2:
        if st.button("📋 複製分享文字", use_container_width=True):
            share_text = f"🌟 ibookle 專家選書報告\n需求：{st.session_state.prev_query}\n\n💡 建議：{res['ai_response']}\n\n© ibookle"
            st.code(share_text, language=None)
            st.toast("報告已就緒！", icon="✨")

    # 回饋區
    if st.session_state.last_row_idx:
        fb_key = f"fb_key_{st.session_state.last_row_idx}"
        st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
        st.write("🌟 這份建議對您有幫助嗎？")
        st.feedback("thumbs", key=fb_key, on_change=update_log_feedback)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("---")
    st.caption("👋 歡迎使用 ibookle！請描述孩子目前的狀況，讓專家為您挑選適合的童書。")