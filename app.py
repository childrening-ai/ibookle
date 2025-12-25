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

st.set_page_config(page_title="ibookle", layout="wide")

st.markdown("""
    <style>
    /* 1. 隱藏頂部與底部雜訊 */
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    div[data-testid="stStatusWidget"], .stAppViewFooter, [data-testid="stDecoration"], [data-testid="stHeader"] {display: none !important;}
    
    /* 2. 背景與容器調整 */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: visible !important; 
        height: auto !important; 
        background-color: white !important;
    }
    
    .main .block-container { 
        padding: 1.5rem 1.5rem 5rem 1.5rem !important; 
        max-width: 900px !important;
    }

    /* 3. 強力消除綠框與藍框 (對主內容與側邊欄同時有效) */
    div[data-baseweb="input"] {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    .stTextInput input {
        border: 2px solid #E67E22 !important; 
        border-radius: 25px !important;
        background-color: white !important;
    }

    .stTextInput input:focus {
        border-color: #D35400 !important;
        box-shadow: 0 0 0 2px rgba(211, 84, 0, 0.1) !important;
        outline: none !important;
    }

    /* 4. 分隔線 (灰線) 樣式穩定化 */
    hr {
        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
        border-bottom: 1px solid #EAECEE !important;
    }

    /* 5. 專家引言框 */
    .expert-box {
        margin: 20px 0;
        padding: 15px;
        background-color: #FEF9E7;
        border-left: 5px solid #F39C12;
        border-radius: 5px;
        color: #5D6D7E;
        line-height: 1.8;
    }
    
    /* 6. 側邊欄專屬樣式 */
    [data-testid="stSidebar"] {
        background-color: #FDFEFE;
        border-right: 1px solid #F4F6F7;
    }
    </style>
    """, unsafe_allow_html=True)

# ================= 4. 側邊欄配置 =================

with st.sidebar:
    st.markdown("## 💡 ibookle 簡介")
    st.info("ibookle 是一個專為家長設計的選書工具，為孩子不同成長階段的挑戰，精選最適合的繪本陪伴。")
    
    st.divider()
    
    st.markdown("### 📋 問卷回饋")
    st.warning("歡迎分享您的使用感受，您的建議是 ibookle 成長的動力！")
    st.link_button("👉 填寫體驗問卷", "https://your-survey-link.com", use_container_width=True)
    
    st.divider()
    
    st.markdown("### ⚡ 服務狀態")
    # 狀態燈號控制
    db_status = "green" 
    
    if db_status == "green":
        st.success("🟢 資料庫：正常運作")
    elif db_status == "yellow":
        st.warning("🟡 資料庫：負載較高")
    else:
        st.error("🔴 資料庫：維護中")
    
    st.caption(f"Session: {st.session_state.session_id}")
    st.caption("© 2026 ibookle")

# ================= 5. 主內容區 =================

st.title("💡 ibookle 繪本共讀專家")
st.markdown("##### *為每一本好書，找到懂它的家長；為每一個孩子，挑選最好的陪伴。*")

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
            st.session_state.last_row_idx = save_to_log(user_query, ai_response, titles_str)

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
        # 這裡會產生之前的灰線 (st.divider 效果相同)
        st.write("")
        st.divider() 

    if st.session_state.last_row_idx:
        st.write("📢 **滿意這次的建議嗎？**")
        st.feedback("thumbs", key=f"fb_key_{st.session_state.last_row_idx}", on_change=update_log_feedback)
        if st.session_state.get(f"submitted_{st.session_state.last_row_idx}"):
            st.toast("感謝您的回饋！", icon="❤️")
            st.success("感謝您的回饋！")
else:
    st.info("👋 你好！我是你的共讀專家。在上方輸入框描述狀況，我會為您推薦最適合的書單。")