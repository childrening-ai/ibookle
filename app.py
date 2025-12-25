import streamlit as st
import json
import pandas as pd
import os
import datetime
import gspread
import requests
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ================= 1. 初始化與環境配置 =================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "gemini768"

genai.configure(api_key=GOOGLE_API_KEY)
llm_model = genai.GenerativeModel('gemini-2.0-flash')

# ================= 2. 功能函數定義 =================

# --- Google Books 封面抓取 (優化：找不到回傳 None) ---
def get_book_cover(title, isbn=""):
    """抓取封面圖，若失敗回傳 None 觸發自動排版調整"""
    query = f"isbn:{isbn}" if isbn and str(isbn) != "nan" and len(str(isbn)) > 5 else title
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
    try:
        res = requests.get(url, timeout=5)
        data = res.json()
        if 'items' in data and 'imageLinks' in data['items'][0]['volumeInfo']:
            img_url = data['items'][0]['volumeInfo']['imageLinks']['thumbnail']
            # 強制換成 https
            return img_url.replace("http://", "https://") + "&zoom=1"
    except:
        pass
    return None

# --- Google Sheets 紀錄功能 (靜默報錯) ---
def save_to_log(user_input, ai_response, recommended_books):
    try:
        creds_json_str = st.secrets["GOOGLE_CREDENTIALS"]
        creds_info = json.loads(creds_json_str.strip())
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
        client = gspread.authorize(creds)
        sheet = client.open("AI_User_Logs").sheet1
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([now, user_input, ai_response, recommended_books])
    except Exception as e:
        print(f"❌ [Log Error] {e}")

# --- Pinecone 向量檢索功能 ---
def get_recommendations(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_query",
        output_dimensionality=768
    )
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )
    return vectorstore.similarity_search(user_query, k=5)

# ================= 3. Streamlit UI 與 CSS 視覺調校 =================

st.set_page_config(
    page_title="ibookle",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* A. 隱藏所有原生組件，消除灰線與多餘白邊 */
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    div[data-testid="stStatusWidget"], .stAppViewFooter, [data-testid="stDecoration"], [data-testid="stHeader"] {display: none !important;}
    
    /* B. 處理雙捲軸：使 Streamlit 內部不捲動，由外部 iFrame 高度撐開 */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: visible !important;
        height: auto !important;
    }
    .main .block-container { 
        padding-top: 1rem !important; 
        padding-bottom: 0px !important;
        overflow: visible !important;
        max-width: 98% !important;
    }

    /* C. 搜尋列：徹底移除原生框線，僅顯示圓角橘框 */
    .stTextInput div[data-baseweb="input"] {
        border: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    .stTextInput input {
        border: 2px solid #E67E22 !important; 
        border-radius: 25px !important;
        padding: 10px 20px !important;
        font-size: 16px !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }

    /* D. 專家建議：簡約引言設計 */
    .expert-box {
        margin: 25px 0;
        padding: 5px 0 5px 15px;
        border-left: 3px solid #EBEDEF;
        color: #5D6D7E;
        font-style: italic;
        line-height: 1.7;
        font-size: 1rem;
    }

    /* E. 其他雜訊消除 */
    button[title="View fullscreen"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 介面呈現 ---
st.title("💡 ibookle")
st.markdown("##### *為每一本好書，找到懂它的家長；為每一個孩子，挑選最好的陪伴。*")

user_input = st.text_input("", placeholder="🔍 輸入孩子的狀況或主題...")

if user_input:
    with st.spinner("🔍 專家正在為您翻閱書櫃..."):
        results = get_recommendations(user_input)
        
        if not results:
            st.warning("查無相關書籍，請換個關鍵字試試看。")
        else:
            book_titles = [doc.metadata.get('Title', '未知書名') for doc in results]
            titles_str = ", ".join(book_titles)
            
            # AI 专家回覆
            prompt = f"使用者問題：{user_input}\n相關書籍：{titles_str}\n請以親子專家溫暖口吻簡述選書理由，禁止表情符號。"
            ai_response = llm_model.generate_content(prompt).text
            
            st.markdown(f'<div class="expert-box">{ai_response}</div>', unsafe_allow_html=True)
            st.markdown("### 📖 為您精選的推薦清單")
            
            for doc in results:
                m = doc.metadata
                title = m.get('Title', '未知書名')
                isbn = m.get('ISBN', '')
                cover_url = get_book_cover(title, isbn)
                
                with st.container():
                    # 關鍵邏輯：判斷有無圖片網址，動態決定排版
                    if cover_url:
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.image(cover_url, use_container_width=True)
                        with col2:
                            st.subheader(f"《{title}》")
                            st.caption(f"✍️ {m.get('Author', '未知')} | 🎨 {m.get('Illustrator', '未知')} | 🏷️ {m.get('Category', '一般')}")
                            if m.get('Quick_Summary'): st.info(m.get('Quick_Summary'))
                            with st.expander("🔍 查看詳細導讀"):
                                st.markdown(m.get('Refine_Content', "暫無詳細內容"))
                                if m.get('Link'): st.link_button("🛒 前往購買 / 查看詳情", m.get('Link'))
                    else:
                        # 無圖片時採用全寬度排版，不留空白
                        st.subheader(f"《{title}》")
                        st.caption(f"✍️ {m.get('Author', '未知')} | 🎨 {m.get('Illustrator', '未知')} | 🏷️ {m.get('Category', '一般')}")
                        if m.get('Quick_Summary'): st.info(m.get('Quick_Summary'))
                        with st.expander("🔍 查看詳細導讀"):
                            st.markdown(m.get('Refine_Content', "暫無詳細內容"))
                            if m.get('Link'): st.link_button("🛒 前往購買 / 查看詳情", m.get('Link'))
                    
                    st.divider()
            
            save_to_log(user_input, ai_response, titles_str)

else:
    st.info("👋 你好！我是你的共讀專家。在上方輸入孩子的狀況，我會為你挑選最適合的書。")

st.markdown("<br>", unsafe_allow_html=True)
st.caption("© 2026 ibookle - 讓每一段共讀時光都更有意義")