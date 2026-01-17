import streamlit as st
import os
import sys
import subprocess
import time
import uuid
from dotenv import load_dotenv

# ==========================================
# 🚑【緊急修復】程式碼內建安裝 (Hot-fix)
# 目標：不修改 requirements.txt 也能跑新版 SDK
# 邏輯：這段會自動檢查環境，如果缺套件就當場安裝
# ==========================================
try:
    import google.generativeai as genai
    # 簡單檢查版本或功能
    genai.configure(api_key="test")
except Exception:
    st.toast("🔧 偵測到環境缺失，正在為 004 安裝專用套件...", icon="⚙️")
    try:
        # 強制安裝 google-generativeai 0.8.3 (相容性最好的版本)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai==0.8.3"])
        st.toast("✅ 環境安裝完成！正在重啟...", icon="🚀")
        time.sleep(2) # 給系統緩衝時間
        import google.generativeai as genai # 重新匯入
    except Exception as e:
        st.error(f"安裝失敗，請檢查 Log: {e}")
        st.stop()

# ==========================================
# 以下是主程式邏輯
# ==========================================

from pinecone import Pinecone

# 1. 初始化與設定
load_dotenv()
st.set_page_config(page_title="ibookle 004 (Hotfix)", layout="wide", initial_sidebar_state="collapsed")

if "session_id" not in st.session_state: 
    st.session_state.session_id = str(uuid.uuid4())[:8]

# 2. 讀取 API Keys
# 優先讀取 Streamlit Secrets，其次讀取環境變數 (本地測試用)
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("❌ API Key 未設定，請檢查 Streamlit Secrets 或 .env 檔案")
    st.stop()

# 設定 Google GenAI
genai.configure(api_key=GOOGLE_API_KEY)

# 3. 核心搜尋函式
def get_recommendations_hotfix(user_query):
    try:
        # 初始化 Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        # ⚠️ 請確認您的 Pinecone Index 名稱是否為 "ibookle-004"
        index = pc.Index("ibookle-004") 

        # 生成向量 (使用剛剛強制安裝的 SDK)
        # model="models/text-embedding-004" 是關鍵
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=user_query,
            task_type="retrieval_query" 
        )
        
        query_vector = result['embedding']

        # 執行搜尋
        res = index.query(
            vector=query_vector, 
            top_k=15, 
            include_metadata=True 
        )

        # 處理結果
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

        # 排序邏輯
        if is_vague:
            # 模糊：只看評分
            candidate_books.sort(key=lambda x: x['rating'], reverse=True)
        else:
            # 精確：先看評分，再看相關度分數
            candidate_books.sort(key=lambda x: (x['rating'], x['score']), reverse=True)

        # 回傳前 5 名
        return [item["meta"] for item in candidate_books[:5]], is_vague

    except Exception as e:
        st.error(f"搜尋過程發生錯誤: {e}")
        return [], False

# 4. UI 介面
st.title("🛡️ ibookle 004 (獨立環境版)")
st.caption("使用 Hot-fix 技術，確保不影響舊版運作")

user_query = st.text_input("", placeholder="輸入關鍵字測試 (例如: 天氣)...", key="main_search")

if user_query:
    with st.spinner("🔍 正在透過 004 模型搜尋..."):
        results, is_vague = get_recommendations_hotfix(user_query)
        
        if results:
            st.success(f"找到相關結果！(模式: {'模糊推薦' if is_vague else '精確搜尋'})")
            for b in results:
                with st.container():
                    st.subheader(f"《{b.get('Title', '無標題')}》")
                    st.caption(f"作者：{b.get('Author', '未知')} | ⭐ 評分：{b.get('Expert_Rating', 0)}")
                    st.info(b.get('Quick_Summary', '無簡介')[:100] + "...")
                    
                    # 顯示完整導讀供檢查
                    with st.expander("查看導讀內容"):
                        st.write(b.get('Refine_Content', '無導讀資料'))
                        
                    st.divider()
        else:
            st.warning("找不到結果，請確認 Pinecone 資料庫是否有資料。")