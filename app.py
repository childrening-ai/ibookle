import streamlit as st
import json, os, datetime, gspread, uuid
import requests
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ================= 1. 初始化與環境配置 =================
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm_model = genai.GenerativeModel('gemini-2.0-flash')

# 初始化 Session 狀態
if "messages" not in st.session_state: 
    st.session_state.messages = []
if "session_id" not in st.session_state: 
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "last_row_idx" not in st.session_state: 
    st.session_state.last_row_idx = None

# ================= 2. 功能函數定義 =================

def save_to_log_chat(user_input, ai_response, recommended_books):
    try:
        creds_json_str = st.secrets["GOOGLE_CREDENTIALS"]
        creds_info = json.loads(creds_json_str.strip())
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
        client = gspread.authorize(creds)
        # 請確保 Google Sheet 第一個分頁有 6 欄：Time, SessionID, Input, AI, Books, Feedback
        sheet = client.open("AI_User_Logs").sheet1
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [now, st.session_state.session_id, user_input, ai_response, recommended_books, ""]
        sheet.append_row(row)
        return len(sheet.get_all_values())
    except Exception as e:
        print(f"Log Error: {e}")
        return None

def update_log_feedback(row_index, score):
    try:
        if not row_index: return
        creds_json_str = st.secrets["GOOGLE_CREDENTIALS"]
        creds_info = json.loads(creds_json_str.strip())
        client = gspread.authorize(ServiceAccountCredentials.from_json_keyfile_dict(creds_info, ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']))
        sheet = client.open("AI_User_Logs").worksheet("Dialogue_Logs")
        feedback_text = "👍" if score == 1 else "👎"
        sheet.update_cell(row_index, 6, feedback_text) # 對話版回饋在第 6 欄
    except Exception as e:
        print(f"Feedback Error: {e}")

def get_recommendations(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=os.getenv("GOOGLE_API_KEY"), 
        task_type="retrieval_query", 
        output_dimensionality=768
    )
    vectorstore = PineconeVectorStore(
        index_name="gemini768", 
        embedding=embeddings, 
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    return vectorstore.similarity_search(user_query, k=5)

# ================= 3. 介面設計與 CSS =================

st.set_page_config(page_title="ibookle Chat", layout="wide")

st.markdown("""
    <style>
    /* 隱藏 Streamlit 原生組件 */
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    div[data-testid="stStatusWidget"], .stAppViewFooter, [data-testid="stDecoration"], [data-testid="stHeader"] {display: none !important;}
    
    /* 背景與高度優化 */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: visible !important; 
        height: auto !important; 
        background-color: white !important;
    }
    
    /* 調整內容邊距，預防 WordPress 裁切 */
    .main .block-container { 
        padding: 2rem 1.5rem 10rem 1.5rem !important; 
        max-width: 95% !important;
    }

    /* 讓對話訊息盒更美觀 */
    [data-testid="stChatMessage"] {
        background-color: #F8F9F9;
        border-radius: 15px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- UI 呈現層 ---
st.title("💡 ibookle 對話助理")
st.markdown("##### *帶著之前的問題繼續聊，我會記得剛才說過的話。*")

# A. 顯示對話歷史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# B. 聊天輸入框
if prompt := st.chat_input("🔍 請問孩子怎麼了？或是針對剛才的建議追問..."):
    # 加入使用者訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("專家思考中..."):
        # 1. 檢索書籍
        results = get_recommendations(prompt)
        titles = ", ".join([d.metadata.get('Title','未知') for d in results])
        
        # 2. 生成帶有上下文的回應
        # 取最近 4 筆對話作為背景
        history_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
        system_instruction = "你是一位溫暖的親子專家。請根據歷史對話與新推薦書籍回答問題。不使用表情符號。"
        full_prompt = f"{system_instruction}\n\n歷史紀錄：\n{history_context}\n\n最新搜尋到的書目：{titles}\n\n請回覆使用者："
        
        ai_response = llm_model.generate_content(full_prompt).text
        
        # 3. 顯示 AI 回應與書籍卡片
        with st.chat_message("assistant"):
            st.markdown(ai_response)
            if results:
                st.markdown("---")
                st.write("📖 **本次推薦書單：**")
                for d in results:
                    m = d.metadata
                    st.markdown(f"**《{m.get('Title')}》**")
                    st.caption(f"作者：{m.get('Author', '未知')} | 繪者：{m.get('Illustrator', '未知')}")
                    with st.expander("🔍 詳細導讀與購書"):
                        st.write(m.get('Refine_Content', '暫無導讀'))
                        if m.get('Link'): st.link_button("🛒 前往購書", m.get('Link'))
                    st.write("") # 間隔

        # 4. 更新歷史與紀錄 Log
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        st.session_state.last_row_idx = save_to_log_chat(prompt, ai_response, titles)

# C. 回饋機制 (顯示在頁面底部)
if st.session_state.last_row_idx:
    st.write("---")
    st.caption("您滿意剛才的建議嗎？您的回饋能讓專家變得更聰明：")
    # key 加入 row_idx 是為了讓每一輪對話的回饋組件都是唯一的
    fb = st.feedback("thumbs", key=f"fb_{st.session_state.last_row_idx}")
    if fb is not None:
        update_log_feedback(st.session_state.last_row_idx, fb)
        st.toast("感謝您的回饋！", icon="❤️")

st.caption(f"© 2026 ibookle | 對話編號: {st.session_state.session_id}")