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
    Layer 3: AI 大腦 - 整合自然語意理解與資料庫標籤對齊
    """
    system_prompt = """
    你是一個精通兒童圖書與家長心理的「ibookle 智慧搜尋助手」。
    你的任務是將使用者的自然語言，精準對齊資料庫的 Metadata 白名單 JSON。

    【重要規則：標籤對齊】
    1. 資料庫過濾器極其嚴格，你輸出的 "filters" 必須「完全符合」以下白名單字串：
    - 適讀年齡: "3～6歲", "6～9歲", "9～12歲", "13～18歲" (注意：波浪號必須為全形「～」)
    - 型式: "繪本・圖畫書", "百科・圖鑑", "漫畫", "讀本・小說・文字書", "操作書・立體書・實驗書"
    - 注音標籤: "有注音", "無注音"
    2. **不准過度推論**：除非明確提到，否則標籤保持 null。使用者搜「恐龍百科」，代表他只要「恐龍」關鍵字與「百科」型式，不代表他一定要有注音或特定年齡。
    3. **陣列輸出規範 (重要)**：
       - 若需求涵蓋多個標籤，請以「陣列 [ ]」輸出。
       - **範例 A (跨年齡)**：搜尋「小學生」 -> 適讀年齡: ["6～9歲", "9～12歲"]。
       - **範例 B (多重型式)**：搜尋「百科或漫畫」 -> 型式: ["百科・圖鑑", "漫畫"]。

    【範例 1：型式誤判防範與標籤對應】
    輸入: "我想找橋樑書"
    輸出: {{
        "search_keywords": "閱讀素養 故事 自主閱讀",
        "filters": {{
            "適讀年齡": "6～9歲",
            "型式": "讀本・小說・文字書",
            "注音標籤": "有注音"
        }}
    }}

    【範例 2：複合查詢與標籤對應】
    輸入: "找一本適合小一的恐龍漫畫，要有注音"
    輸出: {{
        "search_keywords": "恐龍 古生物 冒險",
        "filters": {{
            "適讀年齡": "6～9歲",
            "型式": "漫畫",
            "注音標籤": "有注音"
        }}
    }}

    【範例 3：自然語意擴散 (ibookle 核心價值)】
    輸入: "小孩最近不想上學"
    輸出: {{
        "search_keywords": "校園適應 分離焦慮 拒學 同儕相處 情感支持",
        "filters": {{
            "適讀年齡": null,
            "型式": null,
            "注音標籤": null
        }}
    }}

    【輸出要求】
    1. search_keywords 請排除「型式名詞」(如漫畫、繪本)，專注於主題。
    2. 僅回傳 JSON，不要有任何 Markdown 標記或額外解釋。
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    
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
        # 如果 AI 失敗，回傳原始查詢且不篩選
        return {"search_keywords": user_query, "filters": {}}

# ================= [修正] Layer 4: 分層排序 (Tiered Sorting) =================
def dual_track_search(query, top_k=20, exact_title_filter=None, metadata_filter=None):
    # --- 模式 A: 直通車 (維持不變) ---
    if exact_title_filter:
        results = db_shell.similarity_search_with_score(
            exact_title_filter, k=1, filter={"Title": exact_title_filter}
        )
        return [{"doc": doc, "score": 1.0, "matched_via": ["🚀 直通車"], "is_direct_hit": True} for doc, score in results]

    # --- [關鍵修正] 清洗與轉換 Filter，支援陣列 OR 搜尋 ($in) ---
    active_filters = {}
    if metadata_filter:
        for k, v in metadata_filter.items():
            if v:  # 確保標籤不是 None 或空值
                if isinstance(v, list):
                    # 如果 AI 傳回的是陣列 (例如：['6～9歲', '9～12歲'])
                    # 我們將其轉為 Pinecone 的 $in 運算子，這代表資料庫會執行 OR 邏輯
                    active_filters[k] = {"$in": v}
                else:
                    # 如果是單一字串 (例如：'漫畫')，維持原樣進行精準比對
                    active_filters[k] = v

    # --- 內部搜尋函式 ---
    def run_search(current_filter):
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
        
        # 這裡只做內部排序
        final = list(candidates.values())
        final.sort(key=lambda x: x["score"], reverse=True)
        return final

    # ==========================
    # 🚀 兩階段搜尋 + 分層排序
    # ==========================
    
    # 1. 第一輪：嚴格搜尋 (Strict Search)
    results_strict = run_search(active_filters)
    
    # 為第一輪結果加上標記，證明它們是「完全符合」
    for item in results_strict:
        item['is_strict_match'] = True

    # 2. 判斷是否需要放寬
    # 這裡的邏輯維持：如果嚴格符合的好書太少 (<3)，且有型式/注音限制，就放寬
    high_quality_count = sum(1 for item in results_strict if item['score'] > 0.72)
    
    results_broad = [] # 初始化第二輪結果
    
    droppable_keys = ["型式", "注音", "注音標籤"]
    has_droppable_filters = any(key in active_filters for key in droppable_keys)

    if high_quality_count < 3 and has_droppable_filters:
        relaxed_filter = {}
        if "適讀年齡" in active_filters:
            relaxed_filter["適讀年齡"] = active_filters["適讀年齡"]
        
        if len(relaxed_filter) < len(active_filters):
            st.toast("⚠️ 嚴格條件下書目不足，已自動補入相關主題書籍...")
            
            # 第二輪：放寬搜尋
            results_broad = run_search(relaxed_filter)

    # 3. [關鍵修正] 分層合併 (Tiered Merge)
    # 我們不把兩包混在一起 sort，而是「先排 Strict，再排 Broad」
    
    final_results = []
    
    # A. 先放：嚴格符合的書 (Strict)
    # 這裡面已經按分數排好了
    seen_isbns = set()
    for item in results_strict:
        # 即使是嚴格符合，分數也不能太難看 (例如 < 0.68)
        if item['score'] >= 0.68:
            final_results.append(item)
            seen_isbns.add(item['doc'].metadata.get('ISBN'))

    # B. 後放：放寬條件的書 (Broad)
    # 只有當嚴格符合的書還沒塞滿 15 本時，才從後面補
    if results_broad:
        for item in results_broad:
            isbn = item['doc'].metadata.get('ISBN')
            # 1. 沒重複 2. 分數及格
            if isbn not in seen_isbns and item['score'] >= 0.68:
                # 標記這是放寬的
                item['matched_via'].append("(延伸推薦)") 
                item['is_strict_match'] = False
                final_results.append(item)
    
    # 防呆：如果真的都沒書，放寬底線勉強顯示前 3 名
    if not final_results:
        # 嘗試從 broad 裡撈前三名 (不管分數)
        if results_broad:
            return results_broad[:3]
        # 還是沒有，就回傳 strict 的原始結果
        return results_strict[:3]

    return final_results[:15]

# ================= 5. Layer 5: 閱讀夥伴導讀生成器 =================
def layer_5_generate_report(user_query, ai_analysis, search_results, llm_model):
    """
    Layer 5: 領讀人模式 - 在清單前產出暖心導讀
    """
    if not search_results:
        return "您好！我翻遍了書櫃暫時沒看到完全吻合的書。要不要試著換個說法，或是跟我說說孩子的年級？我一定再幫您找找看。"

    # --- [判定邏輯] 模糊 vs 精確 ---
    # 模糊定義：沒有篩選條件 (例如只搜「天氣」) 或 只有對象沒有主題
    keywords = ai_analysis.get("search_keywords", "").strip()
    filters = ai_analysis.get("filters", {})
    active_filter_count = sum(1 for v in filters.values() if v)
    
    is_vague = True if active_filter_count == 0 else False

    # --- [素材準備] 擷取前 5 本書的專家導讀 (Refine_Content) ---
    context_materials = ""
    for i, item in enumerate(search_results[:5], 1):
        meta = item['doc'].metadata
        title = meta.get('Title', '未知')
        # 夥伴說 Refine_Content 字數固定，直接全量帶入
        expert_view = meta.get('Refine_Content', '這本書深受專家推薦，值得一讀。')
        context_materials += f"【書名{i}】：{title}\n【專家導讀點】：{expert_view}\n\n"

    # --- [Prompt 設定] 父母圈夥伴語氣 ---
    if is_vague:
        prompt_text = f"""
        你是一位在父母圈裡熱心、溫暖且專業的「閱讀夥伴」。
        使用者提問："{user_query}" (這是一個較廣的需求)
        
        【任務指令】
        1. 開頭請說「您好！」，語氣要像老朋友聊天，不要像 AI。
        2. 說明這個主題很棒，你先從 ibookle 口袋名單挑了幾本「專家三星首選」的經典神書。
        3. 根據下方素材，誠懇地介紹這幾本書：
        {context_materials}
        4. 溫柔地追問更多細節（如：孩子的年級、特定興趣），例如：「對了，不知道你們家孩子幾年級了？或是他對這主題有什麼特別好奇的地方嗎？」
        5. 禁止表情符號，約 150-200 字，繁體中文。
        """
    else:
        prompt_text = f"""
        你是一位與家長並肩作戰、懂書也懂孩子的「閱讀夥伴」。
        使用者需求："{user_query}"
        
        【任務指令】
        1. 開頭請說「您好！」，展現出收到需求後「讓我們一起來解決」的熱情。
        2. 根據下方的專家導讀素材，將這幾本書串成一個「共讀建議」：
        {context_materials}
        3. 說明為什麼這組書能精準滿足對方的需求，強調這是結合 ibookle 專家觀點的精選。
        4. 禁止表情符號，語氣平易近人。約 200 字，繁體中文。
        """

    prompt = ChatPromptTemplate.from_messages([("human", prompt_text)])
    chain = prompt | llm_model
    try:
        return chain.invoke({}).content
    except Exception as e:
        return f"您好！剛才我的思緒稍微斷了一下，不過下方的書單都是我為您挑出的寶藏，您可以先看看喔！"

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
            # =========== [導讀置頂] ===========
            st.markdown("### 🕊️ ibookle 夥伴的共讀建議")
            # 確保這裡傳入了 ai_analysis
            report_text = layer_5_generate_report(query, ai_analysis, final_display_list, llm_brain)
            
            with st.chat_message("assistant"):
                st.write(report_text)
            
            st.divider()
            # =================================

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
                
    # [關鍵修正] 這裡定義 rating，下面才不會報錯
                try: 
                    rating = float(meta.get('Expert_Rating', 0))
                except: 
                    rating = 0.0

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
                            # =========== [修改開始] 標題顯示邏輯 ===========
                            # 判斷這本書是「嚴格符合」還是「放寬推薦」
                            is_strict = item.get('is_strict_match', True) 
                            
                            # 基礎標題字串
                            title_display = f"### {rank}. 《{title}》"
                            
                            # 如果有高分評分，加個獎盃
                            if rating >= 2.0: 
                                title_display += " 🏆"

                            if is_strict:
                                # 1. 嚴格符合：正常顯示
                                st.markdown(title_display)
                            else:
                                # 2. 放寬/延伸推薦：加上灰色小字標示，讓使用者知道為什麼它排在後面
                                st.markdown(f"{title_display} <small style='color:gray; font-weight:normal'>(延伸推薦)</small>", unsafe_allow_html=True)
                            # =========== [修改結束] ===========
                        with c2:
                            st.metric("關聯度", f"{score:.3f}")

                        st.caption(f"**{author}** | {publisher} | {book_format}")
                        
                        tags = []
                        if age_range and age_range!='nan': tags.append(f"👶 {age_range}")
                        if "有注音" in str(pinyin_label): tags.append("✅ 有注音")
                        if tags: st.markdown(" ".join([f"`{t}`" for t in tags]))

                        summary = meta.get('Quick_Summary', '')
                        if str(summary)=='nan': summary="暫無"
                        st.markdown(f"**📖 摘要**：{summary}")
                        if link and str(link).startswith('http'):
                            st.link_button("🛒 博客來購書", link)
                        
                    with st.expander("💡 專家導讀"):
                        st.info(meta.get('Refine_Content', ''))
                        st.caption(f"來源: {sources} | ISBN: {meta.get('ISBN')}")
                    st.divider()
        else:
            st.warning("找不到結果。可能是篩選條件太嚴格，或是資料庫中沒有符合的書。")