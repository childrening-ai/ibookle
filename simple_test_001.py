import streamlit as st
import os
import json
import requests
from pinecone import Pinecone
from dotenv import load_dotenv

# ================= 1. 初始化與設定 =================
load_dotenv()
st.set_page_config(page_title="Layer 4 核心驗證 (001)", layout="centered")

# 讀取 API Key (優先從 Secrets 讀取，方便雲端部署，若無則讀本地 .env)
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")

# [設定] 請確保這裡的 Index 名稱與您剛剛上傳程式設定的一樣
INDEX_NAME = "ibookle-dual-001" 

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("❌ API Key 未設定，請檢查 .env 或 Secrets")
    st.stop()

# ================= 2. 核心函式：生成向量 (REST API + 001) =================
def get_embedding_001(text):
    """
    使用 REST API 呼叫 embedding-001 模型。
    不使用 SDK，確保與上傳端的邏輯 100% 一致。
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    # 001 模型的標準 Payload (不需要 task_type, 不需要 config)
    payload = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": text}]}
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()['embedding']['values']
        else:
            st.error(f"Embedding API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"連線錯誤: {e}")
        return None

# ================= 3. 核心函式：雙軌搜尋 (Core + Shell) =================
def search_dual_track(query):
    """
    執行純粹的 Layer 4 雙軌搜尋
    1. 產生查詢向量
    2. 同時搜 Shell (外殼) 與 Core (核心)
    3. 合併結果 (Max Strategy)
    """
    # A. 產生向量
    vector = get_embedding_001(query)
    if not vector:
        return []

    try:
        # B. 連線 Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)

        # C. 雙軌搜尋 (分別搜兩個 Namespace)
        # 搜 Shell (故事/簡介)
        res_shell = index.query(
            vector=vector, 
            top_k=20, 
            namespace="shell", 
            include_metadata=True
        )
        
        # 搜 Core (教育/導讀)
        res_core = index.query(
            vector=vector, 
            top_k=20, 
            namespace="core", 
            include_metadata=True
        )

        # D. 合併邏輯 (Max Strategy)
        # 如果一本書在兩個軌道都被搜到，取分數高的那個
        candidates = {}

        # 處理 Shell 結果
        if res_shell.matches:
            for match in res_shell.matches:
                candidates[match.id] = {
                    "meta": match.metadata,
                    "score": match.score,
                    "source": "shell (外殼)" # 標記來源方便除錯
                }

        # 處理 Core 結果
        if res_core.matches:
            for match in res_core.matches:
                if match.id in candidates:
                    # 如果已經存在，比較分數
                    if match.score > candidates[match.id]["score"]:
                        candidates[match.id]["score"] = match.score
                        candidates[match.id]["source"] = "core (核心)" # 更新來源
                        # 這裡也可以選擇混合分數，但 Max 策略最簡單有效
                else:
                    # 如果不存在，直接加入
                    candidates[match.id] = {
                        "meta": match.metadata,
                        "score": match.score,
                        "source": "core (核心)"
                    }

        # E. 排序並回傳
        # 轉成 List 並依分數高低排序
        final_results = list(candidates.values())
        final_results.sort(key=lambda x: x["score"], reverse=True)

        return final_results[:10] # 只看前 10 名

    except Exception as e:
        st.error(f"Pinecone Error: {e}")
        return []

# ================= 4. UI 介面 (除錯用) =================
st.title("🧪 Layer 4 核心驗證 (001版)")
st.caption("僅測試：向量生成 -> 雙軌搜尋 -> 結果合併")

# 顯示目前的設定，方便確認
with st.expander("查看環境設定"):
    st.write(f"- **Model**: models/embedding-001")
    st.write(f"- **Index**: {INDEX_NAME}")
    st.write(f"- **API Key Status**: {'✅ OK' if GOOGLE_API_KEY else '❌ Missing'}")

query = st.text_input("輸入關鍵字測試 (例如: 恐龍, 分離焦慮, 或是書名)", value="天氣")

if st.button("🚀 執行搜尋"):
    if not query:
        st.warning("請輸入關鍵字")
    else:
        with st.spinner("正在生成向量並搜尋雙軌資料庫..."):
            results = search_dual_track(query)

        if results:
            st.success(f"找到 {len(results)} 筆相關資料")
            
            for i, item in enumerate(results):
                score = item['score']
                meta = item['meta'] or {}
                source = item['source']
                
                # 視覺化分數條
                st.markdown(f"### {i+1}. 《{meta.get('Title', '無標題')}》")
                st.progress(min(score, 1.0)) 
                st.caption(f"🔍 相關度: **{score:.4f}** | 來源軌道: `{source}` | 評分: {meta.get('Expert_Rating', 0)}")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.info(f"**摘要 (Quick_Summary):**\n\n{meta.get('Quick_Summary', '')[:100]}...")
                with c2:
                    st.warning(f"**導讀 (Refine_Content):**\n\n{meta.get('Refine_Content', '')[:100]}...")
                
                with st.expander("查看完整 Metadata (除錯用)"):
                    st.json(meta)
                
                st.divider()
        else:
            st.error("找不到任何結果。")
            st.markdown("""
            **可能原因排除：**
            1. 資料庫是空的 (請確認 `upload` 程式是否跑完)。
            2. Index 名稱不對 (請檢查 `INDEX_NAME`)。
            3. API Key 錯誤。
            """)