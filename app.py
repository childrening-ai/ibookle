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

# --- Google Books 封面抓取 ---
def get_book_cover(title, isbn=""):
    query = f"isbn:{isbn}" if isbn and str(isbn) != "nan" and len(str(isbn)) > 5 else title
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
    try:
        res = requests.get(url, timeout=5)
        data = res.json()
        return data['items'][0]['volumeInfo']['imageLinks']['thumbnail'].replace("http://", "https://")
    except:
        return "https://via.placeholder.com/150x200?text=No+Image"

# --- Google Sheets 紀錄功能 (已移除前端報錯) ---
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
        # 修改點：改用 print 而非 st.error，這樣錯誤只會出現在你的後台控制台
        print(f"❌ [Log Error] 紀錄失敗: {e}")

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

# ================= 3. Streamlit UI 介面配置 =================

st.set_page_config(
    page_title="ibookle",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 綜合 CSS 優化
st.markdown("""
    <style>
    /* A. 徹底消除邊框與灰線 */
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    div[data-testid="stStatusWidget"], .stAppViewFooter, [data-testid="stDecoration"], [data-testid="stHeader"] {display: none !important;}
    [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"], .stApp, .main, .block-container {
        border: none !important; box-shadow: none !important; outline: none !important;
    }
    div[class*="st-emotion-cache"] { box-shadow: none !important; border: none !important; }

    /* B. 瀏覽軸優化 */
    html, body { overflow: visible !important; }
    .main .block-container { 
        padding-top: 1rem !important; 
        padding-bottom: 3rem !important; 
        max-width: 95% !important; 
    }

    /* C. 搜尋區塊明顯化 */
    .stTextInput > div > div > input {
        border: 2px solid #E67E22 !important; 
        border-radius: 25px !important;
        padding: 12px 20px !important;
        font-size: 18px !important;
        box-shadow: 0 4px 12px rgba(230, 126, 34, 0.2) !important;
    }
    
    /* D. 專家建議區塊美化 */
    .expert-box {
        background-color: #FFF5EB;
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #E67E22;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 標題區 ---
st.title("💡 ibookle")
st.markdown("##### *為每一本好書，找到懂它的家長；為每一個孩子，挑選最好的陪伴。*")

# --- 搜尋區 ---
user_input = st.text_input("", placeholder="🔍 輸入孩子的狀況或主題（例如：怕黑、愛生氣、想學科學...）")

if user_input:
    with st.spinner("🔍 專家正在為您翻閱書櫃..."):
        results = get_recommendations(user_input)
        
        if not results:
            st.warning("查發相關書籍，請換個關鍵字試試看。")
        else:
            book_titles = [doc.metadata.get('Title', '未知書名') for doc in results]
            titles_str = ", ".join(book_titles)
            
            prompt = f"使用者問題：{user_input}\n相關書籍：{titles_str}\n請以親子專家身份溫暖鼓勵使用者，簡述選書邏輯，禁止符號。"
            ai_response = llm_model.generate_content(prompt).text
            
            # 顯示 AI 專家回覆
            st.markdown(f'<div class="expert-box"><b>🤖 專家導讀建議</b><br>{ai_response}</div>', unsafe_allow_html=True)
            
            st.markdown("### 📖 為您精選的推薦清單")
            
            for doc in results:
                m = doc.metadata
                title = m.get('Title', '未知書名')
                isbn = m.get('ISBN', '')
                
                cover_url = get_book_cover(title, isbn)
                
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.image(cover_url, use_container_width=True)
                    with col2:
                        st.subheader(f"《{title}》")
                        st.caption(f"✍️ {m.get('Author', '未知')} | 🎨 {m.get('Illustrator', '未知')} | 🏷️ {m.get('Category', '一般')}")
                        
                        quick = m.get('Quick_Summary', "")
                        if quick:
                            st.info(quick)
                        
                        with st.expander("🔍 查看詳細導讀"):
                            st.markdown(m.get('Refine_Content', "暫無詳細內容"))
                            if m.get('Link'):
                                st.link_button("🛒 前往書店查看", m.get('Link'))
                    st.divider()
            
            # 靜默執行 Log 紀錄
            save_to_log(user_input, ai_response, titles_str)

else:
    st.info("👋 你好！我是你的共讀專家。在上方輸入孩子的狀況，我會為你挑選最適合的書。")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.caption("© 2026 ibookle - 讓每一段共讀時光都更有意義")