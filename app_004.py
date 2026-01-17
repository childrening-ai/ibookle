import streamlit as st
import os
import json
import uuid
import requests  # <--- 關鍵：使用最基礎的請求工具
from pinecone import Pinecone
from dotenv import load_dotenv

# ================= 1. 初始化與設定 =================
load_dotenv()
st.set_page_config(page_title="ibookle 004 (REST API)", layout="wide", initial_sidebar_state="collapsed")

if "session_id" not in st.session_state: 
    st.session_state.session_id = str(uuid.uuid4())[:8]

# 讀取 API Keys
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("❌ API Key 未設定，請檢查 Streamlit Secrets")
    st.stop()

# ================= 2. 核心函式：手動呼叫 Google API =================
def get_embedding_via_rest(text, api_key):
    """
    不透過 SDK，直接發送 HTTP 請求給 Google。
    這能避開所有 Python 套件版本衝突的問題。
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "models/text-embedding-004",
        "content": {
            "parts": [{
                "text": text
            }]
        },
        # 004 的關鍵參數：告訴 API 這是「查詢」不是「文件」
        "taskType": "RETRIEVAL_QUERY"
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if response.status_code != 200:
            st.error(f"API 請求失敗: {response.text}")
            return None
            
        result = response.json()
        # 解析回傳的 JSON 取得向量
        return result['embedding']['values']
        
    except Exception as e:
        st.error(f"連線錯誤: {e}")
        return None

# ================= 3. 主流程 =================
def get_recommendations_rest(user_query):
    try:
        # 1. 初始化 Pinecone (這部分通常很穩定，不太會有版本問題)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("ibookle-004") 

        # 2. 生成向量 (使用 REST API)
        query_vector = get_embedding_via_rest(user_query, GOOGLE_API_KEY)
        
        if not query_vector:
            return [], False

        # 3. 執行搜尋
        res = index.query(
            vector=query_vector, 
            top_k=15, 
            include_metadata=True 
        )

        # 4. 處理結果 (邏輯保持不變)
        vague_keywords = ["推薦", "好書", "小學生", "繪本", "介紹", "童書"]
        is_vague = len(user_query.strip()) <= 4 or user_query.strip() in vague_keywords

        candidate_books = []
        for match in res.matches:
            meta = match.metadata or {}
            try:
                rating = float(meta.get('Expert_Rating', 0))
            except: 
                rating = 0.0

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
        st.error(f"搜尋過程發生錯誤: {e}")
        return [], False

# ================= 4. UI 介面 =================
st.title("🌐 ibookle 004 (REST API版)")
st.caption("繞過 SDK 版本衝突，直接連線 Google 核心")

user_query = st.text_input("", placeholder="輸入關鍵字測試 (例如: 天氣)...", key="main_search")

if user_query:
    with st.spinner("🔍 正在透過 API 直連搜尋..."):
        results, is_vague = get_recommendations_rest(user_query)
        
        if results:
            st.success(f"✅ 搜尋成功！(模式: {'模糊推薦' if is_vague else '精確搜尋'})")
            for b in results:
                with st.container():
                    st.subheader(f"《{b.get('Title', '無標題')}》")
                    st.caption(f"作者：{b.get('Author', '未知')} | ⭐ 評分：{b.get('Expert_Rating', 0)}")
                    st.info(b.get('Quick_Summary', '無簡介')[:100] + "...")
                    
                    with st.expander("查看導讀"):
                        st.write(b.get('Refine_Content', '無導讀資料'))
                        
                    st.divider()
        else:
            st.warning("找不到結果，請確認資料庫是否已有資料。")