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
st.set_page_config(page_title="L0+L4 混合搜尋版", layout="wide")

INDEX_NAME = "ibookle-dual-langchain-001" 
CSV_PATH = "ibookle_final_upload_ready.csv" 

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("❌ 缺少 API Key")
    st.stop()

# ================= 2. 初始化搜尋引擎 =================
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

# ================= 3. Layer 0: 快取與直通車 =================
@st.cache_resource
def get_cache():
    cache = {"titles": [], "isbn_map": {}}
    try:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            if "Title" in df.columns:
                cache["titles"] = df["Title"].dropna().astype(str).tolist()
            if "ISBN" in df.columns and "Title" in df.columns:
                temp_df = df[["ISBN", "Title"]].dropna()
                for _, row in temp_df.iterrows():
                    raw_isbn = str(row["ISBN"])
                    clean_isbn = raw_isbn.replace(".0", "").replace("-", "").replace(" ", "").strip()
                    book_title = str(row["Title"]).strip()
                    if clean_isbn:
                        cache["isbn_map"][clean_isbn] = book_title
    except Exception as e:
        st.error(f"Cache Error: {e}")
    return cache

CACHE = get_cache()

def layer_0_direct_hit(query):
    """
    Layer 0: 判斷是否為直通車請求
    """
    if not CACHE or "titles" not in CACHE: return None
    clean_q = query.replace("-", "").replace(" ", "").strip()
    
    # 1. ISBN (精準)
    if clean_q.isdigit() and "isbn_map" in CACHE and clean_q in CACHE["isbn_map"]:
        matched_title = CACHE["isbn_map"][clean_q]
        return matched_title, 1.0, "ISBN"

    # 2. 書名模糊比對 (Partial Ratio)
    if CACHE["titles"]:
        # 使用 partial_token_sort_ratio 支援部分關鍵字
        match = process.extractOne(
            query, 
            CACHE["titles"], 
            scorer=fuzz.partial_token_sort_ratio
        )
        if match:
            matched_title, score, index = match
            # 門檻設為 80，太低容易誤判通用詞
            if score >= 80: 
                return matched_title, score / 100.0, "TITLE"
                
    return None

# ================= 4. Layer 4: 雙軌搜尋 (彈性版) =================
def dual_track_search(query, top_k=20, exact_title_filter=None):
    """
    執行搜尋。
    如果有 exact_title_filter，則只回傳那一本書 (用於抓取置頂書)。
    否則回傳一般語意搜尋結果。
    """
    candidates = {}
    
    # --- 模式 A: 僅抓取特定書名 (為了置頂用) ---
    if exact_title_filter:
        results = db_shell.similarity_search_with_score(
            exact_title_filter, k=1, filter={"Title": exact_title_filter}
        )
        final_list = []
        for doc, score in results:
            # 加上特殊標記 is_direct_hit
            final_list.append({
                "doc": doc, "score": 1.0, 
                "matched_via": ["🚀 直通車"], 
                "is_direct_hit": True
            })
        return final_list

    # --- 模式 B: 一般語意搜尋 ---
    # 注意：這裡不做 filter，讓它廣泛搜尋
    results_shell = db_shell.similarity_search_with_score(query, k=top_k)
    results_core = db_core.similarity_search_with_score(query, k=top_k)

    # Shell 處理
    for doc, score in results_shell:
        book_id = doc.metadata.get('ISBN') or doc.metadata.get('Title')
        if not book_id: continue
        candidates[book_id] = {"doc": doc, "score": score, "matched_via": ["殼 (故事)"], "is_direct_hit": False}

    # Core 處理
    for doc, score in results_core:
        book_id = doc.metadata.get('ISBN') or doc.metadata.get('Title')
        if not book_id: continue
        if book_id in candidates:
            if score > candidates[book_id]["score"]: candidates[book_id]["score"] = score
            candidates[book_id]["matched_via"].append("核 (教育)")
        else:
            candidates[book_id] = {"doc": doc, "score": score, "matched_via": ["核 (教育)"], "is_direct_hit": False}

    final_list = list(candidates.values())
    final_list.sort(key=lambda x: x["score"], reverse=True)
    return final_list[:15] # 取前 15 名

# ================= 5. UI 與控制器 (混合邏輯核心) =================
try:
    db_shell, db_core = get_search_engines()
except Exception as e:
    st.error(f"連線失敗: {e}")
    st.stop()

st.title("📚 ibookle 搜尋引擎 (智慧混合版)")
st.caption("同時滿足「找這本書」與「找這類書」的需求")

query = st.text_input("輸入關鍵字 (例如：時間管理、情緒、雷斯瑪雅)...", value="")

if st.button("🔍 搜尋"):
    if not query:
        st.warning("請輸入內容")
    else:
        final_display_list = []
        direct_hit_title = None
        system_msg = ""
        
        with st.spinner("專家分析中..."):
            # 1. 先執行 Layer 0 檢查
            hit_result = layer_0_direct_hit(query)
            
            # 2. 執行 Layer 4 廣泛搜尋 (不管有沒有命中，都先搜一輪相關的)
            semantic_results = dual_track_search(query)
            
            # 3. 混合邏輯
            if hit_result:
                direct_hit_title, score, hit_type = hit_result
                
                # 檢查這本「命中書」是否已經在廣泛搜尋結果裡？
                # 如果在，把它提到第一位；如果不在，特地去把它抓出來
                
                found_in_list = False
                for item in semantic_results:
                    # 比對書名
                    if item['doc'].metadata.get('Title') == direct_hit_title:
                        item['is_direct_hit'] = True # 標記它
                        item['score'] = 1.0          # 給滿分
                        item['matched_via'] = [f"🚀 {hit_type}直通"] # 改標籤
                        
                        # 把這本書移出列表，準備放在最前面
                        semantic_results.remove(item)
                        final_display_list.append(item)
                        found_in_list = True
                        break
                
                # 如果沒在語意搜尋裡找到 (可能排名太後面被截斷)，我們強制抓它
                if not found_in_list:
                    specific_book = dual_track_search(query, exact_title_filter=direct_hit_title)
                    if specific_book:
                        final_display_list.extend(specific_book)

                system_msg = f"已為您鎖定：《{direct_hit_title}》，並列出其他相關推薦。"
            
            else:
                system_msg = "啟動雙軌語意搜尋..."

            # 4. 把剩下的語意搜尋結果接在後面
            final_display_list.extend(semantic_results)

        # --- 顯示結果 ---
        if system_msg: st.info(system_msg)

        if final_display_list:
            for rank, item in enumerate(final_display_list, 1):
                doc = item['doc']
                score = item['score']
                sources = item['matched_via']
                is_pinned = item.get('is_direct_hit', False)
                meta = doc.metadata
                
                # Metadata 處理
                title = meta.get('Title', '未知')
                author = meta.get('Author', '未知')
                illustrator = meta.get('Illustrator', '')
                if str(illustrator).lower() == 'nan': illustrator = ""
                publisher = meta.get('出版社', '未知')
                book_format = meta.get('型式', '一般')
                age_range = meta.get('適讀年齡', '')
                pinyin_label = meta.get('注音標籤', '')
                link = meta.get('書店連結', '')
                
                # 圖片處理
                img_raw = str(meta.get('書封', ''))
                img_url = None
                if "http" in img_raw:
                    match = re.search(r'\((http[^)]+)\)', img_raw)
                    if match: img_url = match.group(1)
                    elif img_raw.startswith("http"): img_url = img_raw

                # UI 顯示：如果是直通車命中，給它一個特殊的背景或標示
                container = st.container()
                if is_pinned:
                    container.markdown("#### 🎯 精準命中")
                    container.success(f"您似乎在找這本書？")

                with container:
                    col_img, col_info = st.columns([1, 4])
                    with col_img:
                        if img_url: st.image(img_url, use_container_width=True)
                        else: st.markdown("📷")
                            
                    with col_info:
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.markdown(f"### {rank}. 《{title}》")
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
                        st.caption(f"來源: {sources}")
                    st.divider()
        else:
            st.warning("找不到結果。")