import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ================= 1. 設定區 =================
# app_step1_v3.py
load_dotenv()
st.set_page_config(page_title="Layer 4 精準顯示版 (型式修正)", layout="wide")

INDEX_NAME = "ibookle-dual-langchain-001" 

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("❌ 缺少 API Key，請檢查 .env")
    st.stop()

# ================= 2. 初始化核心 =================
@st.cache_resource
def get_search_engines():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_query",
        output_dimensionality=768
    )

    store_shell = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace="shell"
    )

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

# ================= 3. 搜尋邏輯 =================
def dual_track_search(query, top_k=20):
    results_shell = db_shell.similarity_search_with_score(query, k=top_k)
    results_core = db_core.similarity_search_with_score(query, k=top_k)

    candidates = {}

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
            if score > candidates[book_id]["score"]:
                candidates[book_id]["score"] = score
            candidates[book_id]["matched_via"].append("核 (教育)")
        else:
            candidates[book_id] = {"doc": doc, "score": score, "matched_via": ["核 (教育)"]}

    final_list = list(candidates.values())
    final_list.sort(key=lambda x: x["score"], reverse=True)
    return final_list[:10]

# ================= 4. UI 介面 (✨ 書籍型式修正) =================
st.title("📚 ibookle 搜尋引擎 (Metadata 校正)")
st.caption(f"已更新：分類 -> 書籍型式 (Format)")

query = st.text_input("輸入關鍵字...", value="建立自信")

if st.button("🔍 搜尋"):
    if not query:
        st.warning("請輸入內容")
    else:
        with st.spinner("搜尋中..."):
            results = dual_track_search(query)

        if results:
            st.success(f"找到 {len(results)} 本書：")
            
            for rank, item in enumerate(results, 1):
                doc = item['doc']
                score = item['score']
                sources = item['matched_via']
                meta = doc.metadata
                
                # --- 資料清洗 ---
                title = meta.get('Title', '未知標題')
                author = meta.get('Author', '未知')
                illustrator = meta.get('Illustrator', '')
                if str(illustrator).lower() == 'nan': illustrator = ""
                publisher = meta.get('出版社', '未知出版社')
                
                # [修正] 書籍型式 (Book Format)
                # 對應 CSV 的 '型式' 欄位
                book_format = meta.get('型式', '一般')
                
                age_range = meta.get('適讀年齡', '未知')
                pinyin_label = meta.get('注音標籤', '') 
                link = meta.get('書店連結', '')
                
                # 圖片處理
                img_raw = str(meta.get('書封', ''))
                img_url = None
                if "http" in img_raw:
                    match = re.search(r'\((http[^)]+)\)', img_raw)
                    if match:
                        img_url = match.group(1)
                    elif img_raw.startswith("http"):
                        img_url = img_raw

                try: rating = float(meta.get('Expert_Rating', 0))
                except: rating = 0.0

                # --- UI 卡片顯示 ---
                with st.container():
                    col_img, col_info = st.columns([1, 4])
                    
                    with col_img:
                        if img_url:
                            st.image(img_url, use_container_width=True)
                        else:
                            st.markdown("📷 *(無封面)*")
                            
                    with col_info:
                        # 標題區
                        t_col1, t_col2 = st.columns([4, 1])
                        with t_col1:
                            title_display = f"### {rank}. 《{title}》"
                            if rating >= 4.0: title_display += " 🏆"
                            st.markdown(title_display)
                        with t_col2:
                            st.metric("關聯度", f"{score:.3f}")

                        # 資訊行 (已更新書籍型式)
                        # 顯示格式：作者 | 出版社 | 書籍型式
                        st.caption(f"**{author}** | {publisher} | {book_format}")
                        
                        tags = []
                        if age_range and age_range != 'nan': tags.append(f"👶 {age_range}")
                        if "有注音" in str(pinyin_label): tags.append("✅ 有注音")
                        if tags: st.markdown(" ".join([f"`{t}`" for t in tags]))

                        # 摘要
                        summary = meta.get('Quick_Summary', '')
                        if str(summary) == 'nan': summary = "暫無摘要"
                        st.markdown(f"**📖 內容摘要**：\n{summary[:120]}...")

                        # 購書連結
                        if link and str(link) != 'nan' and str(link).startswith('http'):
                            st.link_button("🛒 前往博客來購書", link)

                    with st.expander("💡 專家導讀與教育功能 (Refine Content)"):
                        refine = meta.get('Refine_Content', '')
                        if str(refine) == 'nan': refine = "暫無導讀資料"
                        st.info(refine)
                        st.caption(f"🔎 匹配來源: {', '.join(sources)} | ISBN: {meta.get('ISBN')}")

                    st.divider()
        else:
            st.warning("找不到結果。")