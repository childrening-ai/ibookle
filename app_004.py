import streamlit as st
import json, os, datetime, gspread, uuid, pytz
from dotenv import load_dotenv

# [修改] 改回使用 LangChain，確保與上傳程式邏輯一致
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

# ================= 1. 初始化 =================
load_dotenv()
st.set_page_config(page_title="ibookle 004 測試版", layout="wide", initial_sidebar_state="collapsed")

if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())[:8]

# ================= 2. 核心搜尋函式 (修正版) =================
def get_recommendations_004(user_query):
    try:
        # 1. 初始化 Embedding 模型 (LangChain)
        # 這裡的設定必須跟 upload_004.py 一模一樣！
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",      
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            task_type="retrieval_query", # 搜尋時設為 query
            output_dimensionality=768
        )

        # 2. 初始化 Pinecone
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index = pc.Index("ibookle-004") 

        # 3. 生成向量
        # 使用 LangChain 生成，保證與上傳時的向量空間一致
        query_vector = embeddings.embed_query(user_query)

        # 4. 執行搜尋
        res = index.query(
            vector=query_vector, 
            top_k=15, 
            include_metadata=True 
        )

        # 5. 處理結果
        vague_keywords = ["推薦", "好書", "小學生", "繪本", "介紹", "童書"]
        is_vague = len(user_query.strip()) <= 4 or user_query.strip() in vague_keywords

        candidate_books = []
        for match in res.matches:
            meta = match.metadata or {}
            try:
                rating = float(meta.get('Expert_Rating', 0))
            except: rating = 0.0

            candidate_books.append({
                "meta": meta,
                "score": match.score,
                "rating": rating
            })

        if is_vague:
            candidate_books.sort(key=lambda x: x['rating'], reverse=True)
        else:
            candidate_books.sort(key=lambda x: (x['rating'], x['score']), reverse=True)

        return [item["meta"] for item in candidate_books[:5]], is_vague

    except Exception as e:
        st.error(f"搜尋異常: {e}")
        return [], False

# ================= 3. UI 顯示 (簡化版) =================
st.title("🧪 ibookle 004 驗證 (LangChain版)")
st.caption("使用 LangChain 確保向量一致性")

user_query = st.text_input("", placeholder="輸入關鍵字測試...", key="main_search")

if user_query:
    with st.spinner("搜尋中..."):
        results, is_vague = get_recommendations_004(user_query)
        
        if results:
            for b in results:
                with st.container():
                    st.subheader(f"《{b.get('Title')}》")
                    st.caption(f"作者：{b.get('Author')} | 評分：{b.get('Expert_Rating')}")
                    st.info(b.get('Quick_Summary', '')[:100] + "...")
                    st.divider()
        else:
            st.warning("找不到結果，請檢查 Pinecone 是否有資料")