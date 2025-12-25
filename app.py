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

def get_book_cover(title, isbn=""):
    """
    優化後的抓取機制：
    1. 優先使用 ISBN (去槓號)
    2. 失敗則嘗試『書名』
    3. 找不到則回傳 None，觸發全寬文字模式
    """
    title = str(title).strip()
    isbn = str(isbn).replace("-", "").strip() if isbn and str(isbn) != "nan" else ""
    
    search_queries = []
    if len(isbn) >= 10:
        search_queries.append(f"isbn:{isbn}")
    search_queries.append(f"intitle:{title}")

    for query in search_queries:
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=1"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                data = res.json()
                if 'items' in data:
                    volume_info = data['items'][0].get('volumeInfo', {})
                    image_links = volume_info.get('imageLinks', {})
                    # 嘗試抓取不同尺寸的圖
                    img_url = image_links.get('thumbnail') or image_links.get('smallThumbnail')
                    
                    if img_url:
                        # 強制 HTTPS 並確保連結有效
                        return img_url.replace("http://", "https://")
        except:
            continue
    return None

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

# ================= 3. 介面設計與 CSS 校準 =================

st.set_page_config(page_title="ibookle", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* A. 隱藏原生組件 */
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    div[data-testid="stStatusWidget"], .stAppViewFooter, [data-testid="stDecoration"], [data-testid="stHeader"] {display: none !important;}
    
    /* B. 強制背景白色與高度自適應 (配合外部滾動) */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: visible !important;
        height: auto !important;
        background-color: white !important;
    }
    
    /* C. 內容 Padding 校準：留出 1.5rem 以免內容被邊界裁切 */
    .main .block-container { 
        padding-top: 2rem !important; 
        padding-bottom: 10rem !important; 
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 95% !important;
    }

    /* D. 搜尋列：橘色圓角邊框 */
    .stTextInput div[data-baseweb="input"] {
        border: none !important;
        background-color: transparent !important;
    }
    .stTextInput input {
        border: 2px solid #E67E22 !important; 
        border-radius: 25px !important;
        padding: 10px 20px !important;
    }

    /* E. 專家引言盒 */
    .expert-box {
        margin: 20px 0;
        padding-left: 15px;
        border-left: 3px solid #F39C12;
        color: #555;
        font-style: italic;
        line-height: 1.8;
    }
    </style>
    """, unsafe_allow_html=True)

# --- UI 呈現層 ---
st.title("💡 ibookle")
st.markdown("##### *為每一本好書，找到懂它的家長。*")

user_input = st.text_input("", placeholder="🔍 輸入孩子的狀況（例如：不愛收玩具、害怕看醫生...）")

if user_input:
    with st.spinner("🔍 專家正在為您選書..."):
        results = get_recommendations(user_input)
        
        if not results:
            st.warning("查無相關書籍，請換個關鍵字試試看。")
        else:
            book_titles = [doc.metadata.get('Title', '未知書名') for doc in results]
            titles_str = ", ".join(book_titles)
            
            # AI 專家建議
            prompt = f"使用者問題：{user_input}\n相關書籍：{titles_str}\n請以親子專家口吻簡述選書理由，不使用表情符號，約100字。"
            ai_response = llm_model.generate_content(prompt).text
            
            st.markdown(f'<div class="expert-box">{ai_response}</div>', unsafe_allow_html=True)
            st.markdown("### 📖 精選推薦")
            
            for doc in results:
                m = doc.metadata
                title = m.get('Title', '未知書名')
                cover_url = get_book_cover(title, m.get('ISBN', ''))
                
                with st.container():
                    if cover_url:
                        # 有圖：顯示左圖右文
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.image(cover_url, use_container_width=True)
                        with col2:
                            st.subheader(f"《{title}》")
                            st.caption(f"作者：{m.get('Author', '未知')} | 繪者：{m.get('Illustrator', '未知')}")
                            if m.get('Quick_Summary'): st.info(m.get('Quick_Summary'))
                            with st.expander("🔍 完整導讀"):
                                st.write(m.get('Refine_Content', "暫無內容"))
                                if m.get('Link'): st.link_button("🛒 前往購書", m.get('Link'))
                    else:
                        # 無圖：全寬文字顯示
                        st.subheader(f"《{title}》")
                        st.caption(f"作者：{m.get('Author', '未知')} | 繪者：{m.get('Illustrator', '未知')}")
                        if m.get('Quick_Summary'): st.info(m.get('Quick_Summary'))
                        with st.expander("🔍 完整導讀"):
                            st.write(m.get('Refine_Content', "暫無內容"))
                            if m.get('Link'): st.link_button("🛒 前往購書", m.get('Link'))
                st.divider()
            
            save_to_log(user_input, ai_response, titles_str)

else:
    st.info("👋 你好！我是你的共讀專家。在上方搜尋框輸入孩子的需求，我會為您推薦最適合的書單。")

st.caption("© 2026 ibookle")