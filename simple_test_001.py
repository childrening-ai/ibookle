import streamlit as st
import os
import json
import requests
import numpy as np
from pinecone import Pinecone
from dotenv import load_dotenv

# ================= 設定區 =================
load_dotenv()
st.set_page_config(page_title="001 核心診斷 (Core Only)", layout="centered")

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
INDEX_NAME = "ibookle-dual-001" 

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("❌ API Key 未設定")
    st.stop()

# ================= 核心函式 (001 REST) =================
def get_embedding_001(text):
    """
    取得 001 向量 (REST API)
    """
    if not text: return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": str(text)}]}
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()['embedding']['values']
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"連線錯誤: {e}")
        return None

# ================= 主程式 =================
st.title("🩺 001 核心診斷室")
st.write("目標：檢查模型是否正常，並單純搜尋 Core (教育功能) 軌道。")

# -------------------------------------------
# 步驟 1: 模型自我檢測 (關鍵！)
# -------------------------------------------
st.subheader("1. 模型自我檢測 (Self-Check)")
if st.button("檢測 001 模型是否活著？"):
    with st.spinner("正在生成測試向量..."):
        vec_a = get_embedding_001("天氣")
        vec_b = get_embedding_001("恐龍")
        
        if vec_a and vec_b:
            # 計算相似度
            dot_product = np.dot(vec_a, vec_b)
            
            st.write(f"🔹 「天氣」向量前5碼: `{vec_a[:5]}...`")
            st.write(f"🔹 「恐龍」向量前5碼: `{vec_b[:5]}...`")
            st.metric("兩者相似度 (應低於 0.8)", f"{dot_product:.4f}")
            
            if dot_product > 0.99:
                st.error("🚨 **嚴重警告**：模型失效！天氣和恐龍的向量完全一樣。這代表 API 回傳了預設值或錯誤。")
                st.stop()
            else:
                st.success("✅ 模型正常！能夠區分不同語意。")
        else:
            st.error("❌ 無法生成向量，請檢查 API Key。")

st.divider()

# -------------------------------------------
# 步驟 2: 單軌搜尋 (Core Only)
# -------------------------------------------
st.subheader("2. 單軌搜尋 (Edu/Core Only)")
query = st.text_input("輸入關鍵字 (例如: 建立自信, 分離焦慮)", value="建立自信")

if st.button("搜尋 Core 軌道"):
    # A. 產生向量
    query_vec = get_embedding_001(query)
    
    if query_vec:
        # B. 搜尋 Pinecone
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(INDEX_NAME)
            
            # 只搜 core namespace
            res = index.query(
                vector=query_vec, 
                top_k=5, 
                namespace="core", # 鎖定教育功能軌道
                include_metadata=True
            )
            
            if res.matches:
                st.success(f"找到 {len(res.matches)} 筆結果")
                for match in res.matches:
                    meta = match.metadata
                    score = match.score
                    
                    with st.container():
                        st.markdown(f"**《{meta.get('Title')}》** (相似度: `{score:.4f}`)")
                        # 顯示導讀內容 (這是我們 Core 軌道的核心)
                        st.info(meta.get('Refine_Content', '')[:150] + "...")
                        st.divider()
            else:
                st.warning("找不到結果 (Core 軌道無資料或無匹配)。")
                
        except Exception as e:
            st.error(f"Pinecone Error: {e}")