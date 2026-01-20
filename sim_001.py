import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ================= 1. 設定區 =================
load_dotenv()
st.set_page_config(page_title="Layer 4 雙軌整合搜尋 (正式版)", layout="centered")

# [設定] Index 名稱
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
    # 模型設定 (維持 001 + 768維)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_query",
        output_dimensionality=768
    )

    # 連接 Shell 軌道
    store_shell = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace="shell"
    )

    # 連接 Core 軌道
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

# ================= 3. 核心邏輯：雙軌合併演算法 =================
def dual_track_search(query, top_k=20):
    """
    執行雙軌搜尋並合併結果 (Max Strategy)
    """
    # 1. 分別搜尋 (稍微多抓一點，以便合併後還有足夠數量)
    results_shell = db_shell.similarity_search_with_score(query, k=top_k)
    results_core = db_core.similarity_search_with_score(query, k=top_k)

    # 2. 合併邏輯
    candidates = {}

    # 處理 Shell 結果
    for doc, score in results_shell:
        # 使用 ISBN 作為唯一識別碼 (若無 ISBN 則用書名)
        book_id = doc.metadata.get('ISBN') or doc.metadata.get('Title')
        if not book_id: continue

        candidates[book_id] = {
            "doc": doc,           # 文件內容
            "score": score,       # 分數
            "matched_via": ["殼 (故事)"] # 標記來源
        }

    # 處理 Core 結果
    for doc, score in results_core:
        book_id = doc.metadata.get('ISBN') or doc.metadata.get('Title')
        if not book_id: continue

        if book_id in candidates:
            # 【策略核心】如果兩邊都有，取最高分 (Max Strategy)
            if score > candidates[book_id]["score"]:
                candidates[book_id]["score"] = score
                # 這裡不需要換 doc，因為 Metadata 是一樣的
            
            # 標記它也在 Core 被找到了
            candidates[book_id]["matched_via"].append("核 (教育)")
        else:
            candidates[book_id] = {
                "doc": doc,
                "score": score,
                "matched_via": ["核 (教育)"]
            }

    # 3. 轉為列表並排序
    final_list = list(candidates.values())
    # 依照分數由高到低排序
    final_list.sort(key=lambda x: x["score"], reverse=True)

    # 4. 只回傳前 10 名
    return final_list[:10]

# ================= 4. UI 介面 (正式輸出樣式) =================
st.title("📚 ibookle 智慧搜尋 (Layer 4 整合版)")
st.caption(f"Engine: LangChain + Gemini-001 | Index: {INDEX_NAME}")

query = st.text_input("請輸入孩子的狀況或感興趣的主題...", value="建立自信")

if st.button("🔍 專家推薦"):
    if not query:
        st.warning("請輸入內容")
    else:
        with st.spinner("專家正在從「故事趣味」與「教育功能」雙軌分析..."):
            results = dual_track_search(query)

        if results:
            st.success(f"為您找到 {len(results)} 本相關好書！")
            
            for rank, item in enumerate(results, 1):
                doc = item['doc']
                score = item['score']
                sources = item['matched_via']
                meta = doc.metadata
                
                # 處理評分顯示
                try:
                    rating = float(meta.get('Expert_Rating', 0))
                except:
                    rating = 0.0
                
                # UI 卡片設計
                with st.container():
                    col_info, col_score = st.columns([4, 1])
                    
                    with col_info:
                        title_str = f"{rank}. 《{meta.get('Title', '未知標題')}》"
                        if rating >= 4.0:
                            title_str += " 🏆"
                        st.subheader(title_str)
                        st.caption(f"作者：{meta.get('Author', '未知')} | ISBN: {meta.get('ISBN', '')}")
                        
                        # 顯示摘要 (Quick Summary)
                        st.markdown(f"**📖 內容摘要：** {meta.get('Quick_Summary', '')[:120]}...")
                        
                        # 顯示導讀 (Refine Content) - 這是最有價值的資料
                        with st.expander("💡 專家導讀與互動建議"):
                            st.markdown(meta.get('Refine_Content', '暫無詳細導讀'))

                    with col_score:
                        # 顯示分數與來源標籤
                        st.metric("關聯度", f"{score:.3f}")
                        st.metric("專家評分", f"{rating:.1f}")
                        
                        # 顯示匹配來源 (除錯用，讓您知道是哪邊搜到的)
                        st.caption("匹配來源:")
                        for src in sources:
                            if "殼" in src:
                                st.markdown(f":orange[{src}]")
                            else:
                                st.markdown(f":blue[{src}]")
                    
                    st.divider()
        else:
            st.warning("抱歉，資料庫中暫時找不到符合的書籍。")