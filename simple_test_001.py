import streamlit as st
import os
import re
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from rapidfuzz import process, fuzz
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ================= 1. 設定區 =================
load_dotenv()
st.set_page_config(page_title="Layer 3 AI 大腦整合版", layout="wide")

INDEX_NAME = "ibookle-dual-langchain-001" 
CSV_PATH = "ibookle_final_upload_ready.csv" 

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("❌ 缺少 API Key")
    st.stop()

# --- [新增] 定義資料庫的 Metadata 白名單 ---
# --- [已更新] 根據資料庫實際內容清洗後的白名單 ---
VALID_METADATA = {
    "age_groups": [
        "3～6歲",
        "6～9歲", 
        "9～12歲", 
        "13～18歲", 
        "18歲以上或成人書"
    ],
    "formats": [
        "繪本・圖畫書", 
        "百科・圖鑑", 
        "漫畫", 
        "讀本・小說・文字書", 
        "操作書・立體書・實驗書", 
        "玩具",
        "其他"
    ],
    "pinyin": [
        "有注音", 
        "無注音"
    ]
}

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
        index_name=INDEX_NAME, embedding=embeddings, namespace="shell"
    )
    store_core = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings, namespace="core"
    )
    # [新增] 初始化 Layer 3 的分析大腦 (使用快速的 Flash 模型)
    llm_brain = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0, # 設定為 0 讓他精準不亂掰
        google_api_key=GOOGLE_API_KEY
    )
    return store_shell, store_core, llm_brain

# ================= 3. Layer 0: 直通車 (維持不變) =================
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
    except: pass
    return cache

CACHE = get_cache()

def layer_0_direct_hit(query):
    if not CACHE or "titles" not in CACHE: return None
    clean_q = query.replace("-", "").replace(" ", "").strip()
    
    if clean_q.isdigit() and "isbn_map" in CACHE and clean_q in CACHE["isbn_map"]:
        return CACHE["isbn_map"][clean_q], 1.0, "ISBN"

    if CACHE["titles"]:
        match = process.extractOne(query, CACHE["titles"], scorer=fuzz.partial_token_sort_ratio)
        if match:
            matched_title, score, index = match
            if score >= 80: # 門檻
                return matched_title, score / 100.0, "TITLE"
    return None

# ================= 4. [新增] Layer 3: AI 意圖分析 =================
def layer_3_analyze_intent(user_query, llm_model):
    """
    使用 LLM 分析使用者查詢，提取關鍵字與 Metadata 篩選條件
    """
    system_prompt = """
    你是一個專業的兒童圖書搜尋助手。你的任務是將使用者的自然語言查詢轉換為結構化的搜尋條件。
    
    請分析使用者的輸入，並輸出 JSON 格式，包含以下欄位：
    1. "search_keywords": (字串) 優化後的搜尋關鍵字。
       - **重要規則**：如果使用者輸入的詞是「書籍型式」(如：漫畫、繪本、橋樑書)，請**不要**將其保留在關鍵字中，以免誤搜到關於該物品的內容（除非使用者明確說「畫繪本教學」或「橋樑的構造」）。
       - 範例：輸入「找恐龍橋樑書」 -> 關鍵字：「恐龍」(刪除橋樑書)
    
    2. "filters": (物件) 根據以下白名單提取篩選條件。
       - 如果使用者提到「橋樑書」，請對應到型式：「讀本・小說・文字書」。
       - 如果使用者沒提到該條件，則該欄位留空(null)。
    
    【Metadata 白名單】
    - 適讀年齡: {age_groups}
    - 型式: {formats}
    - 注音標籤: {pinyin}
    
    【範例 1：型式誤判防範】
    輸入: "我想找橋樑書"
    輸出: {{
        "search_keywords": "閱讀素養 故事",  (AI 自動補上適合橋樑書的通用詞，而不是"橋樑結構")
        "filters": {{
            "適讀年齡": "6～9歲",          (橋樑書通常對應 6-9 歲，AI 可自動推論)
            "型式": "讀本・小說・文字書",   (自動對應到正確分類)
            "注音標籤": "有注音"           (橋樑書通常有注音)
        }}
    }}
    
   【範例 2：複合查詢】
    輸入: "找一本適合小一的恐龍漫畫，要有注音"
    輸出: {{
        "search_keywords": "恐龍 古生物 冒險",
        "filters": {{
            "適讀年齡": "6-9歲",
            "型式": "漫畫",
            "注音標籤": "有注音"
        }}
    }}
    【範例 3：自然語意】
    輸入: "小孩最近不想上學"
    輸出: {{
        "search_keywords": "校園適應 分離焦慮 拒學 同儕相處",
        "filters": {{
            "適讀年齡": null,
            "型式": null,
            "注音標籤": null
        }}
    }}
    
    請只回傳 JSON，不要有 Markdown 標記。
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    
    # 填入變數
    chain = prompt | llm_model | JsonOutputParser()
    
    try:
        result = chain.invoke({
            "query": user_query,
            "age_groups": ", ".join(VALID_METADATA["age_groups"]),
            "formats": ", ".join(VALID_METADATA["formats"]),
            "pinyin": ", ".join(VALID_METADATA["pinyin"])
        })
        return result
    except Exception as e:
        print(f"Layer 3 Error: {e}")
        # 如果 AI 失敗，回傳原始查詢且不篩選
        return {"search_keywords": user_query, "filters": {}}

# ================= [修正] Layer 4: 雙軌搜尋 (修復無謂的放寬提示) =================
def dual_track_search(query, top_k=20, exact_title_filter=None, metadata_filter=None):
    # --- 模式 A: 直通車 (維持不變) ---
    if exact_title_filter:
        results = db_shell.similarity_search_with_score(
            exact_title_filter, k=1, filter={"Title": exact_title_filter}
        )
        return [{"doc": doc, "score": 1.0, "matched_via": ["🚀 直通車"], "is_direct_hit": True} for doc, score in results]

    # --- [關鍵修正 1] 預先清洗 Filter，只保留有效值 ---
    # 把 value 為 None 的鍵值對直接刪掉，這樣 active_filters 才是乾淨的
    active_filters = {k: v for k, v in metadata_filter.items() if v} if metadata_filter else {}

    # --- 內部函式：執行搜尋 ---
    def run_search(current_filter):
        # 因為外面已經清洗過了，這裡可以直接用
        candidates = {}
        try:
            r_shell = db_shell.similarity_search_with_score(query, k=top_k, filter=current_filter)
            r_core = db_core.similarity_search_with_score(query, k=top_k, filter=current_filter)
        except: return []

        for doc, score in r_shell:
            bid = doc.metadata.get('ISBN') or doc.metadata.get('Title')
            if not bid: continue
            candidates[bid] = {"doc": doc, "score": score, "matched_via": ["殼"], "is_direct_hit": False}

        for doc, score in r_core:
            bid = doc.metadata.get('ISBN') or doc.metadata.get('Title')
            if not bid: continue
            if bid in candidates:
                if score > candidates[bid]["score"]: candidates[bid]["score"] = score
                candidates[bid]["matched_via"].append("核")
            else:
                candidates[bid] = {"doc": doc, "score": score, "matched_via": ["核"], "is_direct_hit": False}
        
        final = list(candidates.values())
        final.sort(key=lambda x: x["score"], reverse=True)
        return final

    # ==========================
    # 🚀 兩階段搜尋邏輯
    # ==========================
    
    # 1. 第一輪：嚴格搜尋 (使用 active_filters)
    results_strict = run_search(active_filters)
    
    # 2. 判斷是否需要放寬
    high_quality_count = sum(1 for item in results_strict if item['score'] > 0.72)
    final_results = results_strict

    # --- [關鍵修正 2] 精準的觸發條件 ---
    # 只有當同時滿足以下兩個條件才放寬：
    # (A) 高分結果太少 (< 3)
    # (B) 目前的篩選條件裡，真的包含「型式」或「注音」(這才是我們要捨棄的對象)
    # 如果使用者只指定了「年齡」或根本沒指定，就不應該放寬。
    
    droppable_keys = ["型式", "注音", "注音標籤"]
    has_droppable_filters = any(key in active_filters for key in droppable_keys)

    if high_quality_count < 3 and has_droppable_filters:
        
        # 3. 準備放寬條件：保留「適讀年齡」，刪除其他
        relaxed_filter = {}
        if "適讀年齡" in active_filters:
            relaxed_filter["適讀年齡"] = active_filters["適讀年齡"]
        
        # 只有當真的有東西被拿掉時，才執行
        if len(relaxed_filter) < len(active_filters):
            st.toast("⚠️ 嚴格條件下書目不足，系統已自動放寬搜尋範圍 (保留年齡，放寬型式)...")
            
            # 4. 第二輪：放寬搜尋
            results_broad = run_search(relaxed_filter)
            
            # 5. 合併
            seen_isbns = set(item['doc'].metadata.get('ISBN') for item in results_strict)
            for item in results_broad:
                isbn = item['doc'].metadata.get('ISBN')
                if isbn not in seen_isbns:
                    item['matched_via'].append("(放寬)") 
                    final_results.append(item)
    
    # 6. 排序與過濾
    final_results.sort(key=lambda x: x["score"], reverse=True)
    
    # 底線過濾 (0.68)
    filtered_list = [item for item in final_results if item['score'] >= 0.68]
    
    if not filtered_list and final_results:
        return final_results[:3]
        
    return filtered_list[:15]

# ================= 6. UI 與控制器 (Layer 3 整合) =================
try:
    db_shell, db_core, llm_brain = get_search_engines()
except Exception as e:
    st.error(f"連線失敗: {e}")
    st.stop()

st.title("📚 ibookle 搜尋引擎 (Layer 3 智慧版)")
st.caption("Layer 3: AI 幫你拆解關鍵字與篩選條件")

query = st.text_input("輸入關鍵字 (試試：小一的恐龍漫畫、小孩做事拖拖拉拉)...", value="")

if st.button("🔍 搜尋"):
    if not query:
        st.warning("請輸入內容")
    else:
        final_display_list = []
        
        with st.spinner("🧠 Layer 3: AI 正在分析您的意圖..."):
            # --- [Step 1] Layer 0 直通車 ---
            hit_result = layer_0_direct_hit(query)
            
            # --- [Step 2] Layer 3 AI 分析 ---
            # 呼叫 AI 幫忙分析語意和提取篩選條件
            ai_analysis = layer_3_analyze_intent(query, llm_brain)
            
            # 取得 AI 建議的關鍵字 (如果沒有變更，就用原字)
            refined_query = ai_analysis.get("search_keywords", query)
            # 取得 AI 建議的 Filter
            extracted_filters = ai_analysis.get("filters", {})
            
            # [UI 顯示] 讓您看到 AI 到底做了什麼
            with st.expander("🤖 查看 AI 大腦的分析結果", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**🔍 優化後關鍵字**：\n`{refined_query}`")
                with c2:
                    st.markdown(f"**🛡️ 提取篩選條件**：\n`{extracted_filters}`")

            # --- [Step 3] Layer 4 執行搜尋 (帶入參數) ---
            semantic_results = dual_track_search(
                query=refined_query,           # 用變聰明的關鍵字
                metadata_filter=extracted_filters # 用 AI 提取的篩選條件
            )
            
            # --- [Step 4] 混合直通車與語意搜尋結果 ---
            if hit_result:
                direct_hit_title, score, hit_type = hit_result
                found_in_list = False
                for item in semantic_results:
                    if item['doc'].metadata.get('Title') == direct_hit_title:
                        item['is_direct_hit'] = True 
                        item['score'] = 1.0          
                        item['matched_via'] = [f"🚀 {hit_type}直通"] 
                        semantic_results.remove(item)
                        final_display_list.append(item)
                        found_in_list = True
                        break
                
                if not found_in_list:
                    specific_book = dual_track_search(query, exact_title_filter=direct_hit_title)
                    if specific_book: final_display_list.extend(specific_book)
            
            final_display_list.extend(semantic_results)

        # --- 結果顯示 ---
        if final_display_list:
            st.success(f"根據 AI 分析，為您找到 {len(final_display_list)} 本符合條件的書：")
            
            for rank, item in enumerate(final_display_list, 1):
                doc = item['doc']
                score = item['score']
                sources = item['matched_via']
                is_pinned = item.get('is_direct_hit', False)
                meta = doc.metadata
                
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
                        st.caption(f"來源: {sources} | ISBN: {meta.get('ISBN')}")
                    st.divider()
        else:
            st.warning("找不到結果。可能是篩選條件太嚴格，或是資料庫中沒有符合的書。")