import streamlit as st
import os
import numpy as np
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

st.title("🩺 ibookle 004 核心診斷室")

# 1. 初始化檢查
st.header("1. 連線檢查")
api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
google_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key or not google_key:
    st.error("❌ API Key 缺失！請檢查 Secrets。")
    st.stop()
else:
    st.success("✅ API Key 讀取成功")

try:
    pc = Pinecone(api_key=api_key)
    index = pc.Index("ibookle-004")
    stats = index.describe_index_stats()
    st.write(f"📊 **資料庫狀態 (ibookle-004)**:")
    st.json(stats)
    
    if stats.total_vector_count == 0:
        st.error("❌ 資料庫是空的！請重新執行上傳程式。")
        st.stop()
        
except Exception as e:
    st.error(f"❌ Pinecone 連線失敗: {e}")
    st.stop()

# 2. 檢查資料庫向量 (Data Vector Check)
st.header("2. 資料庫向量檢查 (是否全都一樣？)")
st.write("隨機抓取 3 本書，檢查它們的向量是否重複...")

try:
    # 用一個假向量隨便搜，目的是抓出存在的書
    dummy_vec = [0.1] * 768
    res = index.query(vector=dummy_vec, top_k=3, include_values=True, include_metadata=True)
    
    if len(res.matches) < 2:
        st.warning("⚠️ 資料不足 2 筆，無法比對。")
    else:
        vec_1 = res.matches[0].values
        vec_2 = res.matches[1].values
        title_1 = res.matches[0].metadata.get('Title', '未知')
        title_2 = res.matches[1].metadata.get('Title', '未知')
        
        # 計算相似度 (Dot Product)
        similarity = np.dot(vec_1, vec_2)
        
        st.write(f"📚 **書籍 A**: {title_1}")
        st.write(f"📚 **書籍 B**: {title_2}")
        st.metric(label="兩本書的向量相似度 (1.0 代表完全一樣)", value=f"{similarity:.6f}")
        
        if similarity > 0.999:
            st.error("🚨 **診斷結果：資料庫壞了！** 兩本不同的書竟然有完全一樣的向量。這代表上傳時發生了錯誤 (複製人 Bug)。")
            st.info("👉 解法：問題出在 `upload_004.py`，可能是批次上傳時變數沒更新，或是 Embedding API 回傳了空值。")
        else:
            st.success("✅ 資料庫正常：不同的書有不同的向量。")

except Exception as e:
    st.error(f"檢查過程發生錯誤: {e}")

# 3. 檢查 Embedding 模型 (Query Vector Check)
st.header("3. 搜尋模型檢查 (是否無法區分關鍵字？)")
st.write("測試 generating 兩個完全不同的關鍵字向量...")

try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",      
        google_api_key=google_key,
        task_type="retrieval_query",
        output_dimensionality=768
    )
    
    q1 = "天氣"
    q2 = "恐龍"
    
    v1 = embeddings.embed_query(q1)
    v2 = embeddings.embed_query(q2)
    
    q_similarity = np.dot(v1, v2)
    
    st.write(f"🔍 關鍵字 1: {q1}")
    st.write(f"🔍 關鍵字 2: {q2}")
    st.metric(label="兩個關鍵字的相似度", value=f"{q_similarity:.6f}")
    
    if q_similarity > 0.99:
        st.error("🚨 **診斷結果：Embedding 模型失效！** 天氣和恐龍竟然被視為一樣的意思。")
        st.info("👉 解法：這通常是 `langchain-google-genai` 版本與 `task_type` 參數不相容導致。")
    else:
        st.success("✅ 模型正常：能區分不同語意。")

except Exception as e:
    st.error(f"模型測試失敗: {e}")