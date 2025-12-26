import streamlit as st
import json, os, datetime, gspread, uuid
import pytz 
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ================= 1. 初始化與環境配置 =================
load_dotenv()

# 使用新版 SDK 初始化 Client
if "GOOGLE_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    client = None

if "session_id" not in st.session_state: 
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "last_row_idx" not in st.session_state:
    st.session_state.last_row_idx = None

# ================= 2. 功能函數定義 =================

def get_recommendations(user_query):
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        pinecone_key = st.secrets["PINECONE_API_KEY"]
        
        # 1. 初始化 Embedding
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key, 
            task_type="retrieval_query"
        )
        
        # 2. 手動測試維度並截斷的包裹函式
        # 有些版本的 LangChain 會忽略 output_dimensionality，我們手動切片
        class DimensionFixer:
            def __init__(self, model):
                self.model = model
            def embed_query(self, text):
                vec = self.model.embed_query(text)
                return vec[:768] # 強制只取前 768 碼
            def embed_documents(self, texts):
                vecs = self.model.embed_documents(texts)
                return [v[:768] for v in vecs]

        fixed_embeddings = DimensionFixer(embeddings_model)
        
        # 3. 連接 Pinecone
        vectorstore = PineconeVectorStore(
            index_name="gemini768", 
            embedding=fixed_embeddings, 
            pinecone_api_key=pinecone_key
        )
        
        return vectorstore.similarity_search(user_query, k=5)
        
    except Exception as e:
        st.error(f"🔍 搜尋引擎暫時無法連線: {e}")
        return None

# ================= 3. 介面設計與 CSS =================

st.set_page_config(page_title="ibookle", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    .stTextInput input { border: 2px solid #E67E22 !important; border-radius: 25px !important; }
    .expert-box { margin: 20px 0; padding: 20px; background-color: #FEF9E7; border-left: 5px solid #F39C12; border-radius: 10px; color: #5D6D7E; line-height: 1.8; font-size: 1.1em; }
    button[data-testid="stSidebarCollapseButton"] { background-color: #E67E22 !important; color: white !important; border-radius: 50% !important; }
    </style>
    """, unsafe_allow_html=True)

# 抓取統計數據
total_answers = "---"
sheet_for_count = get_google_sheet()
if sheet_for_count:
    try:
        total_answers = len(sheet_for_count.get_all_values()) - 1
    except:
        pass

# ================= 4. 側邊欄 =================

with st.sidebar:
    st.markdown("## 💡 ibookle 簡介")
    st.info("專為家長設計的選書工具，精選最適合的繪本陪伴。")
    st.divider()
    st.metric("📊 服務熱度", f"{total_answers} 次")
    st.divider()
    st.markdown("### 📋 意見回饋")
    st.link_button("👉 填寫問卷", "https://your-survey-link.com", use_container_width=True)

# ================= 5. 主內容區 =================

st.title("💡 ibookle 繪本共讀專家")
st.markdown("##### *為每一本好書，找到懂它的家長；為每一個孩子，挑選最好的陪伴。*")

user_query = st.text_input("", placeholder="🔍 輸入孩子的狀況，例如：孩子最近怕黑、不愛刷牙...", key="main_search")

# 搜尋觸發邏輯
if user_query and (not st.session_state.search_results or st.session_state.get("prev_query") != user_query):
    with st.spinner("專家正在為您選書..."):
        results = get_recommendations(user_query)
        if results:
            titles_str = ", ".join([d.metadata.get('Title','未知') for d in results])
            prompt = f"使用者問題：{user_query}\n相關書籍：{titles_str}\n請以親子專家口吻簡述選書理由，不使用表情符號，約150字。"
            
            try:
                response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
                ai_response = response.text
                
                # 封裝結果
                st.session_state.search_results = {
                    "ai_response": ai_response, 
                    "books": [{
                        "Title": d.metadata.get('Title', '未知'), 
                        "Author": d.metadata.get('Author', '未知'), 
                        "Illustrator": d.metadata.get('Illustrator', '未知'), 
                        "Quick_Summary": d.metadata.get('Quick_Summary', ''), 
                        "Refine_Content": d.metadata.get('Refine_Content', '暫無導讀'), 
                        "Link": d.metadata.get('Link', '')
                    } for d in results]
                }
                st.session_state.prev_query = user_query
                # 寫入日誌
                st.session_state.last_row_idx = save_to_log(user_query, ai_response, titles_str)
            except Exception as e:
                st.error("AI 專家暫時休息中，請稍後再試。")

# 結果顯示
if st.session_state.search_results:
    res = st.session_state.search_results
    st.markdown(f'<div class="expert-box">{res["ai_response"]}</div>', unsafe_allow_html=True)
    
    st.markdown("### 📖 精選推薦")
    for b in res["books"]:
        with st.container():
            st.subheader(f"《{b['Title']}》")
            st.write(f"**作者：** {b['Author']} | **繪者：** {b['Illustrator']}")
            if b['Quick_Summary']: 
                st.info(b['Quick_Summary'])
            with st.expander("🔍 專家詳細導讀"):
                st.write(b['Refine_Content'])
                if b['Link']: 
                    st.link_button("🛒 查看書籍詳情", b['Link'])
        st.divider()

    if st.session_state.last_row_idx:
        st.write("📢 **滿意這次的建議嗎？**")
        st.feedback("thumbs", key=f"fb_key_{st.session_state.last_row_idx}", on_change=update_log_feedback)
else:
    st.info("👋 您好！我是您的共讀專家。在上方輸入框描述狀況，我會為您推薦最適合的書單。")

# 底部統計 (手機版友善)
st.write("")
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.write(f"✨ **服務紀錄**：已解答 **{total_answers}** 次")
with c2:
    st.link_button("回饋建議", "https://your-survey-link.com", use_container_width=True)