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

# --- 功能函數 ---
def save_to_log(user_input, ai_response, recommended_books):
    try:
        creds_json_str = st.secrets["GOOGLE_CREDENTIALS"]
        creds_info = json.loads(creds_json_str.strip())
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
        client = gspread.authorize(creds)
        sheet = client.open("AI_User_Logs").worksheet("Brief_Logs")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 欄位：時間, 使用者輸入, AI回覆, 書目, 回饋
        row = [now, user_input, ai_response, recommended_books, ""]
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
        sheet.update_cell(row_index, 5, feedback_text)
    except: pass

def get_recommendations(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"), task_type="retrieval_query", output_dimensionality=768)
    vectorstore = PineconeVectorStore(index_name="gemini768", embedding=embeddings, pinecone_api_key=os.getenv("PINECONE_API_KEY"))
    return vectorstore.similarity_search(user_query, k=5)

# --- UI & CSS ---
st.set_page_config(page_title="ibookle Search", layout="wide")
st.markdown("""<style>
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    html, body, [data-testid="stAppViewContainer"] {overflow: visible !important; height: auto !important; background-color: white !important;}
    .main .block-container {padding: 2rem 1.5rem 10rem 1.5rem !important; max-width: 95% !important;}
    .stTextInput input {border: 2px solid #E67E22 !important; border-radius: 25px !important;}
    .expert-box {margin: 20px 0; padding-left: 15px; border-left: 3px solid #F39C12; color: #555; font-style: italic; line-height: 1.8;}
</style>""", unsafe_allow_html=True)

st.title("💡 ibookle 搜尋版")
user_input = st.text_input("", placeholder="🔍 想找什麼樣的書？")

if user_input:
    with st.spinner("專家選書中..."):
        results = get_recommendations(user_input)
        if not results: st.warning("查無結果")
        else:
            titles = ", ".join([d.metadata.get('Title','') for d in results])
            ai_response = llm_model.generate_content(f"使用者：{user_input}\n推薦書：{titles}\n請以親子專家口吻簡述理由(100字，不含表情)。").text
            st.markdown(f'<div class="expert-box">{ai_response}</div>', unsafe_allow_html=True)
            for d in results:
                m = d.metadata
                st.subheader(f"《{m.get('Title')}》")
                st.caption(f"作者：{m.get('Author')} | 繪者：{m.get('Illustrator')}")
                st.info(m.get('Quick_Summary'))
                with st.expander("🔍 完整導讀"):
                    st.write(m.get('Refine_Content'))
                    if m.get('Link'): st.link_button("🛒 前往購書", m.get('Link'))
                st.divider()
            
            row_idx = save_to_log(user_input, ai_response, titles)
            st.write("📢 **滿意這次的建議嗎？**")
            fb = st.feedback("thumbs")
            if fb is not None:
                update_log_feedback(row_idx, fb)
                st.success("感謝回饋！")