import streamlit as st
import json  # 必須多匯入這個庫
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import os
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ================= 1. 初始化與環境配置 =================
load_dotenv()

# API Keys (本地端從 .env 讀取，雲端從 Secrets 讀取)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "gemini768"

# 初始化 Gemini 模型 (用於生成建議)
genai.configure(api_key=GOOGLE_API_KEY)
llm_model = genai.GenerativeModel('gemini-2.0-flash')

# ================= 2. 功能函數定義 =================

# --- Google Sheets 紀錄功能 ---
def save_to_log(user_input, ai_response, recommended_books):
    try:
        
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

        # 1. 從 Secrets 讀取那串長文字
        creds_json_str = st.secrets["GOOGLE_CREDENTIALS"]

        # 2. 將文字轉成 Python 字典 (這步能解決之前的 'str' object 錯誤)
        creds_info = json.loads(creds_json_str)

        # 3. 使用 dict 方式讀取
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
        
        client = gspread.authorize(creds)
        
        sheet = client.open("AI_User_Logs").sheet1
        
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([now, user_input, ai_response, recommended_books])
    
    except Exception as e:
        st.error(f"⚠️ Log 紀錄失敗: {e}")

# --- Pinecone 向量檢索功能 ---
def get_recommendations(user_query):
    # 初始化 768 維 Embedding
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_query",
        output_dimensionality=768
    )
    
    # 連接 Vector Store
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )
    
    # 檢索相似書籍 (k=3 代表找最相關的 3 筆)
    return vectorstore.similarity_search(user_query, k=5)

# ================= 3. Streamlit UI 介面 =================

st.set_page_config(
    page_title="ibookle",
    layout="wide",                # 讓內容填滿寬度
    initial_sidebar_state="collapsed"  # 自動把左邊那塊深色的收起來
)

st.markdown("""
    <style>
    /* 隱藏所有選單、標籤與底部工具欄 */
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    
    /* 針對嵌入模式下的 StatusWidget (包含 Built with Streamlit 的那一條) */
    div[data-testid="stStatusWidget"], 
    .stAppViewFooter, 
    [data-testid="stDecoration"],
    [data-testid="stHeader"] {
        display: none !important;
    }

    /* 移除底部多餘的 Padding */
    .main .block-container {
        padding-bottom: 0px !important;
        margin-bottom: -50px !important;
    }

    /* 隱藏右下角的 Fullscreen 圖示 */
    button[title="View fullscreen"] {
        display: none !important;
    }

    /* 移除外層容器的邊框與陰影 */
    [data-testid="stAppViewContainer"] {
        border: none !important;
    }
    
    /* 移除主要的區塊間隙與可能的細線 */
    .main {
        background-color: transparent !important;
    }
    
    /* 針對嵌入模式下的特定容器進行邊框消除 */
    div[class*="stApp"] {
        border: none !important;
        box-shadow: none !important;
    }

    /* 1. 徹底拔除所有層級的陰影與邊框 */
    [data-testid="stAppViewContainer"], 
    [data-testid="stAppViewBlockContainer"], 
    .stApp, .main, .block-container {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* 2. 針對嵌入模式下最頑固的「白色卡片」邊緣 */
    div[class*="st-emotion-cache"] {
        box-shadow: none !important;
        border: none !important;
    }

    /* 3. 移除頂部 header 的底線 */
    header {
        border-bottom: none !important;
    }

    /* 4. 確保背景透明度，消除色差造成的「偽線條」 */
    .stAppViewMain {
        background-color: transparent !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

st.title("💡 ibookle")
st.markdown("##### *為每一本好書，找到懂它的家長；為每一個孩子，挑選最好的陪伴。*")
st.markdown("---")
st.write("你好！我是你的共讀專家。輸入孩子的狀況或想找的主題，我會為你挑選最適合的書。")

# 側邊欄：顯示目前狀態
with st.sidebar:
    st.header("關於 ibookle")
    st.write("ibookle 透過 AI 技術，從專業視角為家長挑選最適合孩子的繪本與書籍。")
    st.divider()
    st.success(f"✅ 資料庫已連線: 311 筆精選")

st.markdown("---")
st.caption("© 2026 ibookle - 讓每一段共讀時光都更有意義")

# 使用者輸入
user_input = st.text_input("想找什麼樣的書？", placeholder="例如：想找關於克服恐懼的繪本...")

if user_input:
    with st.spinner("🔍 正在為您翻閱書櫃並整理建議..."):
        # 1. 檢索書籍
        results = get_recommendations(user_input)
        
        if not results:
            st.warning("查無相關書籍，請換個關鍵字試試看。")
        else:
            # 2. 準備給 AI 的 Prompt (讓回答更具關聯性)
            book_titles = [doc.metadata.get('Title', '未知書名') for doc in results]
            titles_str = ", ".join(book_titles)
            
            prompt = f"""
            使用者目前的問題：{user_input}
            我為他找到的相關書籍包括：{titles_str}
            
            請以專業親子共讀專家的身份，用親切溫暖的語氣，簡述為什麼這幾本書適合使用者。
            不需要詳細介紹每本書（下方會有詳細內容），只要針對使用者的情境給予一段鼓勵與引導即可。
            (禁止使用表情符號)
            """
            
            # 3. 生成 AI 回覆
            ai_response = llm_model.generate_content(prompt).text
            
            # 4. 顯示結果
            st.markdown("### 🤖 專家建議")
            st.write(ai_response)
            
            st.markdown("---")
            st.markdown("### 📖 精選推薦清單")
            
            for doc in results:
                m = doc.metadata
                with st.container():
                    # 顯示書名與基本資訊
                    st.subheader(f"《{m.get('Title', '未知書名')}》")
                    st.caption(f"✍️ 作者：{m.get('Author', '未知')} | 🎨 繪者：{m.get('Illustrator', '未知')} | 🏷️ 分類：{m.get('Category', '一般')}")
                    
                    # 顯示快速摘要 (Quick_Summary)
                    quick = m.get('Quick_Summary', "")
                    if quick:
                        st.info(quick)
                    
                    # 深度導讀摺疊區
                    with st.expander("🔍 點擊查看專家深度導讀"):
                        refine = m.get('Refine_Content', "暫無詳細導讀內容")
                        st.markdown(refine)
                        
                        link = m.get('Link', "")
                        if link:
                            st.link_button("🛒 前往購買 / 查看更多", link)
                    
                    st.write("") # 增加間距
            
            # 5. 紀錄對話到 Google Sheets
            save_to_log(user_input, ai_response, titles_str)

st.markdown("---")
st.caption("© 2026 ibookle - 讓每一段共讀時光都更有意義")