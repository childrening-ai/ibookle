import streamlit as st
import json, os, datetime, gspread, uuid
import pytz 
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ================= 1. 初始化與環境配置 =================
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

if "session_id" not in st.session_state: 
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "last_row_idx" not in st.session_state: 
    st.session_state.last_row_idx = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None

# ================= 2. 功能函數定義 =================

def get_google_sheet():
    creds_json_str = st.secrets["GOOGLE_CREDENTIALS"]
    creds_info = json.loads(creds_json_str.strip())
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
    client_gs = gspread.authorize(creds)
    return client_gs.open("AI_User_Logs").worksheet("Brief_Logs")

def save_to_log(user_input, ai_response, recommended_books):
    try:
        sheet = get_google_sheet()
        tw_tz = pytz.timezone('Asia/Taipei')
        now_tw = datetime.datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M:%S")
        new_row = [now_tw, st.session_state.session_id, user_input, ai_response, recommended_books, ""]
        sheet.append_row(new_row)
        return len(sheet.get_all_values())
    except Exception as e:
        return None

def update_log_feedback():
    row_idx = st.session_state.last_row_idx
    if row_idx:
        score = st.session_state.get(f"fb_key_{row_idx}")
        if score is not None:
            try:
                sheet = get_google_sheet()
                feedback_text = "👍" if score == 1 else "👎"
                sheet.update_cell(row_idx, 6, feedback_text)
                st.session_state[f"submitted_{row_idx}"] = True
            except Exception as e:
                pass

def get_recommendations(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"), task_type="retrieval_query", output_dimensionality=768)
    vectorstore = PineconeVectorStore(index_name="gemini768", embedding=embeddings, pinecone_api_key=os.getenv("PINECONE_API_KEY"))
    return vectorstore.similarity_search(user_query, k=5)

# ================= 3. 介面設計與 CSS =================

# 加入預設展開側邊欄設定
st.set_page_config(page_title="ibookle", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* 1. 基礎 UI 隱藏 */
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    div[data-testid="stStatusWidget"], .stAppViewFooter, [data-testid="stDecoration"], [data-testid="stHeader"] {display: none !important;}
    
    /* 2. 手機版優化：強化左上角箭頭按鈕 */
    button[data-testid="stSidebarCollapseButton"] {
        background-color: #E67E22 !important;
        color: white !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
        top: 10px !important;
        left: 10px !important;
    }

    /* 3. 主容器與輸入框樣式 */
    .stTextInput input { border: 2px solid #E67E22 !important; border-radius: 25px !important; }
    .expert-box { margin: 20px 0; padding: 15px; background-color: #FEF9E7; border-left: 5px solid #F39C12; border-radius: 5px; color: #5D6D7E; line-height: 1.8; }
    [data-testid="stSidebar"] { background-color: #FDFEFE; border-right: 1px solid #F4F6F7; }
    </style>
    """, unsafe_allow_html=True)

# 預先抓取統計數據 (為了雙重顯示)
total_answers = 0
try:
    sheet_data = get_google_sheet()
    total_answers = len(sheet_data.get_all_records())
except:
    total_answers = "---"

# ================= 4. 側邊欄配置 =================

with st.sidebar:
    st.markdown("## 💡 ibookle 簡介")
    st.info("ibookle 是一個專為家長設計的選書工具，精選最適合的繪本陪伴。")
    
    st.divider()
    st.metric("📊 服務熱度", f"{total_answers} 次")
    st.write(f"已解答家長疑問：**{total_answers}** 次")
    
    st.divider()
    st.markdown("### 📋 問卷回饋")
    st.link_button("👉 填寫體驗問卷", "https://your-survey-link.com", use_container_width=True)
    st.caption("© 2026 ibookle")

# ================= 5. 主內容區 =================

st.title("💡 ibookle 繪本共讀專家")
st.markdown("##### *為每一本好書，找到懂它的家長；為每一個孩子，挑選最好的陪伴。*")

user_query = st.text_input("", placeholder="🔍 輸入孩子的狀況...", key="main_search")

# 搜尋邏輯
if user_query and (not st.session_state.search_results or st.session_state.get("prev_query") != user_query):
    with st.spinner("專家選書中..."):
        results = get_recommendations(user_query)
        if results:
            titles_str = ", ".join([d.metadata.get('Title','未知') for d in results])
            prompt = f"使用者問題：{user_query}\n相關書籍：{titles_str}\n請以親子專家口吻簡述選書理由，不使用表情符號，約150字。"
            response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
            ai_response = response.text
            
            st.session_state.search_results = {"ai_response": ai_response, "books": [{"Title": d.metadata.get('Title', '未知'), "Author": d.metadata.get('Author', '未知'), "Illustrator": d.metadata.get('Illustrator', '未知'), "Quick_Summary": d.metadata.get('Quick_Summary', ''), "Refine_Content": d.metadata.get('Refine_Content', '暫無導讀'), "Link": d.metadata.get('Link', '')} for d in results]}
            st.session_state.prev_query = user_query
            st.session_state.last_row_idx = save_to_log(user_query, ai_response, titles_str)

# 顯示結果
if st.session_state.search_results:
    res = st.session_state.search_results
    st.markdown(f'<div class="expert-box">{res["ai_response"]}</div>', unsafe_allow_html=True)
    st.markdown("### 📖 精選推薦")
    for b in res["books"]:
        with st.container():
            st.subheader(f"《{b['Title']}》")
            st.caption(f"作者：{b['Author']} | 繪者：{b['Illustrator']}")
            if b['Quick_Summary']: st.info(b['Quick_Summary'])
            with st.expander("🔍 專家詳細導讀"):
                st.write(b['Refine_Content'])
                if b['Link']: st.link_button("🛒 前往購書", b['Link'])
        st.divider() 

    if st.session_state.last_row_idx:
        st.write("📢 **滿意這次的建議嗎？**")
        st.feedback("thumbs", key=f"fb_key_{st.session_state.last_row_idx}", on_change=update_log_feedback)
else:
    st.info("👋 你好！我是你的共讀專家。在上方輸入框描述狀況，我會為您推薦最適合的書單。")

# ================= 6. 手機版底部統計 (雙重顯示) =================
st.write("") # 空行
st.write("")
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.write(f"✨ **ibookle 服務紀錄**")
    st.write(f"已解答家長疑問：**{total_answers}** 次")
with c2:
    st.write("📢 **意見回饋**")
    st.link_button("填寫問卷", "https://your-survey-link.com", use_container_width=True)