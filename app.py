import streamlit as st
import json, os, datetime, gspread, uuid
import pytz 
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
    client = gspread.authorize(creds)
    return client.open("AI_User_Logs").worksheet("Brief_Logs")

def save_to_log(user_input, ai_response, recommended_books):
    try:
        sheet = get_google_sheet()
        tw_tz = pytz.timezone('Asia/Taipei')
        now_tw = datetime.datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M:%S")
        
        new_row = [now_tw, st.session_state.session_id, user_input, ai_response, recommended_books, ""]
        sheet.append_row(new_row)
        # 獲取最新行號
        return len(sheet.get_all_values())
    except Exception as e:
        st.error(f"Log Error: {e}")
        return None

def update_log_feedback():
    # 當 feedback 改變時觸發的 callback
    if st.session_state.last_row_idx:
        score = st.session_state.get(f"fb_key_{st.session_state.last_row_idx}")
        if score is not None:
            try:
                sheet = get_google_sheet()
                feedback_text = "👍" if score == 1 else "👎"
                sheet.update_cell(st.session_state.last_row_idx, 6, feedback_text)
                st.toast("感謝您的回饋！", icon="❤️")
            except Exception as e:
                print(f"Feedback Update Error: {e}")

def get_recommendations(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"), task_type="retrieval_query", output_dimensionality=768)
    vectorstore = PineconeVectorStore(index_name="gemini768", embedding=embeddings, pinecone_api_key=os.getenv("PINECONE_API_KEY"))
    return vectorstore.similarity_search(user_query, k=5)

# ================= 3. 介面設計與 CSS =================

st.set_page_config(page_title="ibookle", layout="wide")

# 強力消除綠框與藍框的 CSS
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    div[data-testid="stStatusWidget"], .stAppViewFooter, [data-testid="stDecoration"], [data-testid="stHeader"] {display: none !important;}
    
    html, body, [data-testid="stAppViewContainer"] {
        overflow: visible !important; 
        height: auto !important; 
        background-color: white !important;
    }
    
    .main .block-container { padding: 1.5rem 1.5rem 5rem 1.5rem !important; }

    /* 修正 3：強力移除所有狀態下的綠色框線 (focus, active, hover) */
    .stTextInput input {
        border: 2px solid #E67E22 !important; 
        border-radius: 25px !important;
    }
    /* 針對 Streamlit 的內層容器進行強制覆蓋 */
    div[data-baseweb="input"] {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    .stTextInput input:focus, .stTextInput input:active {
        border-color: #D35400 !important;
        box-shadow: 0 0 0 2px rgba(211, 84, 0, 0.2) !important; /* 改用淡橘色光暈代替綠光 */
        outline: none !important;
    }

    .expert-box {
        margin: 20px 0;
        padding: 15px;
        background-color: #FEF9E7;
        border-left: 5px solid #F39C12;
        border-radius: 5px;
        color: #5D6D7E;
        line-height: 1.8;
    }
    </style>
    """, unsafe_allow_html=True)

# --- UI 呈現層 ---
st.title("💡 ibookle")
st.markdown("##### *為每一本好書，找到懂它的家長；為每一個孩子，挑選最好的陪伴。*")

# 使用 on_change 來處理搜尋，避免干擾回饋
user_query = st.text_input("", placeholder="🔍 輸入孩子的狀況...", key="main_search")

if user_query and (not st.session_state.search_results or st.session_state.get("prev_query") != user_query):
    with st.spinner("專家選書中..."):
        results = get_recommendations(user_query)
        if results:
            titles_str = ", ".join([d.metadata.get('Title','未知') for d in results])
            prompt = f"使用者問題：{user_query}\n相關書籍：{titles_str}\n請以親子專家口吻簡述選書理由，不使用表情符號，約150字。"
            ai_response = llm_model.generate_content(prompt).text
            
            st.session_state.search_results = {
                "ai_response": ai_response,
                "books": [
                    {
                        "Title": d.metadata.get('Title', '未知'),
                        "Author": d.metadata.get('Author', '未知'),
                        "Illustrator": d.metadata.get('Illustrator', '未知'),
                        "Quick_Summary": d.metadata.get('Quick_Summary', ''),
                        "Refine_Content": d.metadata.get('Refine_Content', '暫無導讀'),
                        "Link": d.metadata.get('Link', '')
                    } for d in results
                ]
            }
            st.session_state.prev_query = user_query
            # 儲存 Log 並回傳行號
            st.session_state.last_row_idx = save_to_log(user_query, ai_response, titles_str)

# 渲染搜尋結果
if st.session_state.search_results:
    res = st.session_state.search_results
    st.markdown(f'<div class="expert-box">{res["ai_response"]}</div>', unsafe_allow_html=True)
    
    st.markdown("### 📖 精選推薦")
    for b in res["books"]:
        with st.container():
            st.subheader(f"《{b['Title']}》")
            st.caption(f"作者：{b['Author']} | 繪者：{b['Illustrator']}")
            if b['Quick_Summary']:
                st.info(b['Quick_Summary'])
            with st.expander("🔍 專家詳細導讀"):
                st.write(b['Refine_Content'])
                if b['Link']: st.link_button("🛒 前往購書", b['Link'])
        st.write("")

    # 修正點讚：使用 callback 確保點擊後立刻執行寫入，且 key 穩定
    if st.session_state.last_row_idx:
        st.divider()
        st.write("📢 **滿意這次的建議嗎？**")
        st.feedback(
            "thumbs", 
            key=f"fb_key_{st.session_state.last_row_idx}", 
            on_change=update_log_feedback
        )
else:
    st.info("👋 你好！我是你的共讀專家。在上方輸入框描述狀況，我會為您推薦最適合的書單。")

st.caption("© 2026 ibookle")