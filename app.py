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
if "current_books" not in st.session_state:
    st.session_state.current_books = [] 

# ================= 2. 功能函數定義 =================

def get_google_sheet():
    """連線並開啟指定的分頁 Dialogue_Logs"""
    creds_json_str = st.secrets["GOOGLE_CREDENTIALS"]
    creds_info = json.loads(creds_json_str.strip())
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
    client = gspread.authorize(creds)
    # 指定開啟名為 Dialogue_Logs 的分頁
    return client.open("AI_User_Logs").worksheet("Dialogue_Logs")

def save_to_log_chat(user_input, ai_response, recommended_books):
    try:
        sheet = get_google_sheet()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 標題欄順序：Time, SessionID, Input, AI, Books, Feedback
        new_row = [
            now_str,                      # Time
            st.session_state.session_id,  # SessionID
            user_input,                   # Input
            ai_response,                  # AI
            recommended_books,            # Books
            ""                            # Feedback
        ]
        sheet.append_row(new_row)
        return len(sheet.get_all_values())
    except Exception as e:
        print(f"Log Error: {e}")
        return None

def update_log_feedback(row_index, score):
    try:
        if not row_index: return
        sheet = get_google_sheet()
        feedback_text = "👍" if score == 1 else "👎"
        # 更新第 6 欄 (Feedback)
        sheet.update_cell(row_index, 6, feedback_text) 
    except Exception as e:
        print(f"Feedback Error: {e}")

def get_recommendations(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"), task_type="retrieval_query", output_dimensionality=768)
    vectorstore = PineconeVectorStore(index_name="gemini768", embedding=embeddings, pinecone_api_key=os.getenv("PINECONE_API_KEY"))
    return vectorstore.similarity_search(user_query, k=5)

# ================= 3. 介面設計與 CSS =================

st.set_page_config(page_title="ibookle", layout="wide")

st.markdown("""
    <style>
    /* 隱藏原生組件 */
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    div[data-testid="stStatusWidget"], .stAppViewFooter, [data-testid="stDecoration"], [data-testid="stHeader"] {display: none !important;}
    
    html, body, [data-testid="stAppViewContainer"] {
        overflow: visible !important; 
        height: auto !important; 
        background-color: white !important;
    }
    
    /* 調整 Padding：底部從 5rem 縮減至 2rem，讓輸入框在 iframe 中更靠上 */
    .main .block-container { 
        padding: 1.5rem 1.5rem 2rem 1.5rem !important; 
        max-width: 95% !important;
    }

    [data-testid="stChatMessage"] { 
        background-color: #FDFEFE; 
        border-radius: 12px; 
        border: 1px solid #F2F4F4; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- UI 呈現層 ---
st.title("💡 ibookle")
st.markdown("##### *為每一本好書，找到懂它的家長；為每一個孩子，挑選最好的陪伴。*")

# 1. 顯示對話歷史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. 聊天輸入框
if prompt := st.chat_input("🔍 請問孩子怎麼了？或是針對剛才的建議追問..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("專家選書中..."):
        results = get_recommendations(prompt)
        
        # 鎖定書籍狀態
        st.session_state.current_books = [
            {
                "Title": d.metadata.get('Title', '未知'),
                "Author": d.metadata.get('Author', '未知'),
                "Illustrator": d.metadata.get('Illustrator', '未知'),
                "Quick_Summary": d.metadata.get('Quick_Summary', ''),
                "Refine_Content": d.metadata.get('Refine_Content', '暫無導讀'),
                "Link": d.metadata.get('Link', '')
            } for d in results
        ]
        
        titles_str = ", ".join([b['Title'] for b in st.session_state.current_books])
        history_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
        system_instruction = "你是一位溫暖的親子專家。請根據歷史對話與推薦書目回答問題。不使用表情符號。"
        full_prompt = f"{system_instruction}\n\n歷史紀錄：\n{history_context}\n\n搜尋書目：{titles_str}\n\n請回覆："
        
        ai_response = llm_model.generate_content(full_prompt).text
        
        with st.chat_message("assistant"):
            st.markdown(ai_response)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        st.session_state.last_row_idx = save_to_log_chat(prompt, ai_response, titles_str)

# 3. 渲染推薦書卡
if st.session_state.current_books:
    st.markdown("---")
    st.write("📖 **為您精選的共讀書單：**")
    for b in st.session_state.current_books:
        with st.container():
            st.markdown(f"**《{b['Title']}》**")
            st.caption(f"作者：{b['Author']} | 繪者：{b['Illustrator']}")
            if b['Quick_Summary']:
                st.info(b['Quick_Summary'])
            with st.expander("🔍 查看詳細專家導讀"):
                st.write(b['Refine_Content'])
                if b['Link']: st.link_button("🛒 前往購書連結", b['Link'])
        st.write("")

# 4. 回饋機制
if st.session_state.last_row_idx:
    st.write("---")
    st.caption("滿意這次的推薦嗎？")
    fb = st.feedback("thumbs", key=f"fb_{st.session_state.last_row_idx}")
    if fb is not None:
        update_log_feedback(st.session_state.last_row_idx, fb)
        st.toast("感謝您的回饋！", icon="❤️")

st.caption("© 2026 ibookle")