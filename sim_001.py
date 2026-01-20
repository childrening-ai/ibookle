import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ================= 1. 設定區 =================
load_dotenv()
st.set_page_config(page_title="雙軌搜尋驗證 (LangChain版)", layout="wide")

# [設定] 必須與上傳程式的 Index 名稱一致
INDEX_NAME = "ibookle-dual-langchain-001" 

# API Keys 檢查
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("❌ 缺少 API Key，請檢查 .env")
    st.stop()

# ================= 2. 初始化核心 (LangChain) =================
@st.cache_resource
def get_search_engines():
    """
    初始化 Embedding 模型與連接 Pinecone
    """
    # A. 模型設定 (必須與上傳時一致，但 task_type 改為 query)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_query", # 搜尋時使用 query 模式
        output_dimensionality=768    # [關鍵] 必須強制設定為 768
    )

    # B. 連接 Shell 軌道 (Namespace='shell')
    store_shell = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace="shell"
    )

    # C. 連接 Core 軌道 (Namespace='core')
    store_core = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace="core"
    )
    
    return store_shell, store_core

try:
    db_shell, db_core = get_search_engines()
except Exception as e:
    st.error(f"連線失敗: {e}")
    st.stop()

# ================= 3. UI 介面 =================
st.title("🔎 雙軌搜尋驗證器 (LangChain 001)")
st.caption(f"Index: `{INDEX_NAME}` | Model: `gemini-embedding-001 (768)`")

query = st.text_input("輸入關鍵字測試 (例如：自信、恐龍、不想上學)", value="自信")

if st.button("🚀 執行雙軌搜尋"):
    if not query:
        st.warning("請輸入內容")
    else:
        with st.spinner("正在分別搜尋 Shell 與 Core 軌道..."):
            
            # --- 搜尋 Shell (外殼) ---
            # k=3 代表取前 3 名
            results_shell = db_shell.similarity_search_with_score(query, k=3)
            
            # --- 搜尋 Core (核心) ---
            results_core = db_core.similarity_search_with_score(query, k=3)

        # --- 顯示結果 ---
        col1, col2 = st.columns(2)

        # 左欄：Shell 結果
        with col1:
            st.header("🐚 Shell (外殼/故事)")
            if results_shell:
                for doc, score in results_shell:
                    st.success(f"相關度: {score:.4f}")
                    st.markdown(f"**《{doc.metadata.get('Title')}》**")
                    st.info(f"摘要: {doc.page_content[:100]}...")
                    st.caption(f"ID: {doc.metadata.get('ISBN')}")
                    st.divider()
            else:
                st.warning("Shell 軌道無結果")

        # 右欄：Core 結果
        with col2:
            st.header("☢️ Core (核心/教育)")
            if results_core:
                for doc, score in results_core:
                    st.success(f"相關度: {score:.4f}")
                    st.markdown(f"**《{doc.metadata.get('Title')}》**")
                    # 這裡顯示的 page_content 應該要是「教育功能/導讀」
                    st.warning(f"導讀: {doc.page_content[:100]}...") 
                    st.caption(f"ID: {doc.metadata.get('ISBN')}")
                    st.divider()
            else:
                st.warning("Core 軌道無結果")