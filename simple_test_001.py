import streamlit as st
import os
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from rapidfuzz import process, fuzz

# ================= 1. 設定區 =================
load_dotenv()
st.set_page_config(page_title="L0 直通車 + L4 雙軌搜尋", layout="wide")

INDEX_NAME = "ibookle-dual-langchain-001" 
CSV_PATH = "ibookle_final_upload_ready.csv" 

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("❌ 缺少 API Key")
    st.stop()

# ================= 2. 初始化搜尋引擎 (L4 Backend) =================
@st.cache_resource
def get_search_engines():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_query",
        output_dimensionality=768
    )
    store_shell = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings, namespace="shell"
    )
    store_core = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings, namespace="core"
    )
    return store_shell, store_core

# ================= 3. Layer 0: 快取與直通車邏輯 =================
@st.cache_resource
def get_cache():
    """
    建立書名與 ISBN 的快取清單
    """
    cache = {"titles": [], "isbn_map": {}}
    try:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            
            # A. 建立書名列表 (用於模糊比對)
            if "Title" in df.columns:
                cache["titles"] = df["Title"].dropna().astype(str).tolist()
            
            # B. 建立 ISBN 對照表 (用於精準搜尋)
            if "ISBN" in df.columns and "Title" in df.columns:
                temp_df = df[["ISBN", "Title"]].dropna()
                for _, row in temp_df.iterrows():
                    # 清洗 ISBN
                    raw_isbn = str(row["ISBN"])
                    clean_isbn = raw_isbn.replace(".0", "").replace("-", "").replace(" ", "").strip()
                    book_title = str(row["Title"]).strip()
                    
                    if clean_isbn:
                        cache["isbn_map"][clean_isbn] = book_title
        else:
            st.warning(f"⚠️ 找不到 {CSV_PATH}，直通車功能受限。")
    except Exception as e:
        st.error(f"Cache Error: {e}")
    return cache

# 👇👇👇 [關鍵修正] 這行絕對不能少！也不能放在函式裡面！ 👇👇👇
CACHE = get_cache() 
# 👆👆👆 這一行負責把上面算好的結果存入全域變數 CACHE 👆👆👆


def layer_0_direct_hit(query):
    """
    Layer 0: 直通車 (修正版：支援部分關鍵字)
    """
    # 0. 防呆機制：如果 CACHE 沒載入成功，直接回傳 None
    if not CACHE or "titles" not in CACHE:
        return None

    # 清洗輸入
    clean_q = query.replace("-", "").replace(" ", "").strip()
    
    # 1. ISBN 精準比對
    if clean_q.isdigit() and "isbn_map" in CACHE and clean_q in CACHE["isbn_map"]:
        matched_title = CACHE["isbn_map"][clean_q]
        return matched_title, 1.0

    # 2. 書名模糊比對 (演算法：部分符合 partial_token_sort_ratio)
    if CACHE["titles"]:
        match = process.extractOne(
            query, 
            CACHE["titles"], 
            scorer=fuzz.partial_token_sort_ratio
        )
        
        if match:
            matched_title, score, index = match
            
            # [除錯] 顯示分數 (正式版可註解掉)
            st.sidebar.markdown(f"🔍 直通車: `{query}` vs `{matched_title}` = **{score}**")

            # 門檻建議 75~85
            if score >= 80: 
                return matched_title, score / 100.0
                
    return None

# ================= 4. Layer 4: 雙軌搜尋 (含直通車處理) =================
def dual_track_search(query, top_k=20, exact_title=None):
    candidates = {}
    
    # --- 狀況 A: 直通車模式 (Exact Match) ---
    if exact_title:
        # 直接用 Filter 鎖定書名，不用管語意相似度
        # 只搜 shell 即可，因為 metadata 一樣
        results = db_shell.similarity_search_with_score(
            exact_title, k=1, filter={"Title": exact_title}
        )
        
        final_list = []
        for doc, score in results:
            final_list.append({
                "doc": doc, 
                "score": 1.0,  # 強制滿分
                "matched_via": ["🚀 直通車"]
            })
        return final_list

    # --- 狀況 B: 一般語意搜尋模式 (Semantic Search) ---
    results_shell = db_shell.similarity_search_with_score(query, k=top_k)
    results_core = db_core.similarity_search_with_score(query, k=top_k)

    # Shell 處理
    for doc, score in results_shell:
        book_id = doc.metadata.get('ISBN') or doc.metadata.get('Title')
        if not book_id: continue
        candidates[book_id] = {"doc": doc, "score": score, "matched_via": ["殼 (故事)"]}

    # Core 處理 (Max Strategy)
    for doc, score in results_core:
        book_id = doc.metadata.get('ISBN') or doc.metadata.get('Title')
        if not book_id: continue

        if book_id in candidates:
            if score > candidates[book_id]["score"]: candidates[book_id]["score"] = score
            candidates[book_id]["matched_via"].append("核 (教育)")
        else:
            candidates[book_id] = {"doc": doc, "score": score, "matched_via": ["核 (教育)"]}

    final_list = list(candidates.values())
    final_list.sort(key=lambda x: x["score"], reverse=True)
    return final_list[:10]

# ================= 5. UI 與控制器 (Controller) =================
try:
    db_shell, db_core = get_search_engines()
except Exception as e:
    st.error(f"連線失敗: {e}")
    st.stop()

st.title("📚 ibookle 搜尋引擎 (L0 + L4)")
st.caption("Layer 0: ISBN/書名直通 | Layer 4: 雙軌語意搜尋")

query = st.text_input("輸入關鍵字 (試試輸入 ISBN、書名，或任何想找的主題)...", value="")

if st.button("🔍 搜尋"):
    if not query:
        st.warning("請輸入內容")
    else:
        results = []
        system_msg = ""
        
        with st.spinner("專家分析中..."):
            # --- Step 1: Layer 0 檢查 ---
            direct_hit = layer_0_direct_hit(query)
            
            if direct_hit:
                # 命中直通車
                matched_title, score = direct_hit
                type_label = "ISBN" if query.replace("-","").isdigit() else "書名"
                system_msg = f"🚀 [{type_label}直通車] 精準調閱：《{matched_title}》"
                results = dual_track_search(query, exact_title=matched_title)
            
            else:
                # 未命中 -> 走 L4 語意搜尋
                system_msg = "🔍 啟動雙軌語意搜尋..."
                results = dual_track_search(query)

        # --- 結果顯示 ---
        if system_msg: st.info(system_msg)

        if results:
            for rank, item in enumerate(results, 1):
                doc = item['doc']
                score = item['score']
                sources = item['matched_via']
                meta = doc.metadata
                
                # Metadata
                title = meta.get('Title', '未知')
                author = meta.get('Author', '未知')
                illustrator = meta.get('Illustrator', '')
                if str(illustrator).lower() == 'nan': illustrator = ""
                publisher = meta.get('出版社', '未知')
                book_format = meta.get('型式', '一般')
                age_range = meta.get('適讀年齡', '')
                pinyin_label = meta.get('注音標籤', '')
                link = meta.get('書店連結', '')
                
                # 圖片
                img_raw = str(meta.get('書封', ''))
                img_url = None
                if "http" in img_raw:
                    match = re.search(r'\((http[^)]+)\)', img_raw)
                    if match: img_url = match.group(1)
                    elif img_raw.startswith("http"): img_url = img_raw
                
                try: rating = float(meta.get('Expert_Rating', 0))
                except: rating = 0.0

                with st.container():
                    col_img, col_info = st.columns([1, 4])
                    with col_img:
                        if img_url: st.image(img_url, use_container_width=True)
                        else: st.markdown("📷")
                            
                    with col_info:
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            t_str = f"### {rank}. 《{title}》"
                            if rating >= 4.0: t_str += " 🏆"
                            st.markdown(t_str)
                        with c2:
                            st.metric("關聯度", f"{score:.3f}")

                        st.caption(f"**{author}** | {publisher} | {book_format}")
                        
                        tags = []
                        if age_range and age_range!='nan': tags.append(f"👶 {age_range}")
                        if "有注音" in str(pinyin_label): tags.append("✅ 有注音")
                        if tags: st.markdown(" ".join([f"`{t}`" for t in tags]))

                        summary = meta.get('Quick_Summary', '')
                        if str(summary)=='nan': summary="暫無"
                        st.markdown(f"**📖 摘要**：{summary[:100]}...")
                        
                        if link and str(link).startswith('http'):
                            st.link_button("🛒 博客來購書", link)

                    with st.expander("💡 專家導讀"):
                        st.info(meta.get('Refine_Content', ''))
                        st.caption(f"來源: {sources} | ISBN: {meta.get('ISBN')}")
                    st.divider()
        else:
            st.warning("找不到結果。")