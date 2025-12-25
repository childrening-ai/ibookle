import streamlit as st
import json, os, datetime, gspread, uuid
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm_model = genai.GenerativeModel('gemini-2.0-flash')

# --- 初始化 Session ---
if "messages" not in st.session_state: st.session_state.messages = []
if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())[:8]
if "last_row_idx" not in st.session_state: st.session_state.last_row_idx = None

# --- 功能函數 (對話版特有) ---
def save_to_log_chat(user_input, ai_response, recommended_books):
    try:
        creds_json_str = st.secrets["GOOGLE_CREDENTIALS"]
        creds_info = json.loads(creds_json_str.strip())
        client = gspread.authorize(ServiceAccountCredentials.from_json_keyfile_dict(creds_info, ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']))
        sheet = client.open("AI_User_Logs").worksheet("Dialogue_Logs")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 欄位：時間, SessionID, 使用者, AI回覆, 書目, 回饋
        row = [now, st.session_state.session_id, user_input, ai_response, recommended_books, ""]
        sheet.append_row(row)
        return len(sheet.get_all_values())
    except: return None

def update_log_feedback(row_index, score):
    try:
        creds_json_str = st.secrets["GOOGLE_CREDENTIALS"]
        creds_info = json.loads(creds_json_str.strip())
        client = gspread.authorize(ServiceAccountCredentials.from_json_keyfile_dict(creds_info, ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']))
        sheet = client.open("AI_User_Logs").sheet1
        feedback_text = "👍" if score == 1 else "👎"
        sheet.update_cell(row_index, 6, feedback_text) # 對話版在第 6 欄
    except: pass

def get_recommendations(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"), task_type="retrieval_query", output_dimensionality=768)
    vectorstore = PineconeVectorStore(index_name="gemini768", embedding=embeddings, pinecone_api_key=os.getenv("PINECONE_API_KEY"))
    return vectorstore.similarity_search(user_query, k=5)

# --- UI & CSS ---
st.set_page_config(page_title="ibookle Chat", layout="wide")
st.markdown("""<style>
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    html, body, [data-testid="stAppViewContainer"] {overflow: visible !important; height: auto !important; background-color: white !important;}
    .main .block-container {padding: 2rem 1.5rem 10rem 1.5rem !important; max-width: 95% !important;}
</style>""", unsafe_allow_html=True)

st.title("💡 ibookle 對話助理")

# 顯示歷史對話
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("🔍 請輸入您的需求或追問..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.spinner("思考中..."):
        results = get_recommendations(prompt)
        titles = ", ".join([d.metadata.get('Title','') for d in results])
        
        # 建立帶歷史的回應
        history_text = "\n".join([f"{m['role']}:{m['content']}" for m in st.session_state.messages[-4:]])
        full_prompt = f"你是親子專家。歷史：{history_text}\n新推薦書：{titles}\n請針對新問題回覆(不含表情)。"
        ai_response = llm_model.generate_content(full_prompt).text
        
        with st.chat_message("assistant"):
            st.markdown(ai_response)
            if results:
                st.write("---")
                for d in results:
                    m = d.metadata
                    st.markdown(f"**《{m.get('Title')}》** | {m.get('Author')}")
                    with st.expander("導讀"): 
                        st.write(m.get('Refine_Content'))
                        if m.get('Link'): st.link_button("🛒 購書", m.get('Link'))

        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        st.session_state.last_row_idx = save_to_log_chat(prompt, ai_response, titles)

# 回饋顯示在最下方
if st.session_state.last_row_idx:
    fb = st.feedback("thumbs", key=f"fb_{st.session_state.last_row_idx}")
    if fb is not None:
        update_log_feedback(st.session_state.last_row_idx, fb)
        st.toast("感謝您的回饋！")