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

# ================= 4. [新增] Layer 3: AI 意圖分析（更新20260217） =================
def layer_3_analyze_intent(user_query, llm_model):
    # 整合了「實體保護」、「寬鬆檢索」與「50% 權重平衡」後的 System Prompt
    system_prompt = """
    你是一個精通兒童圖書與家長心理的「ibookle 智慧搜尋助手」。
    你的任務是將使用者的自然語言，精準對齊資料庫的 Metadata 白名單 JSON。

    【🚨 實體保護與負面屏蔽規則 (優先等級最高)】
    1. **實體不可拆解**：若提問包含專有名詞（如：麵包小偷、野貓軍團、神奇柑仔店），請將其視為不可分割的實體。**絕對禁止**拆解字面意義（例如：禁止將『麵包小偷』提取為「偷竊」或「犯罪」主題）。
    2. **類型詞保留**：如「武俠」、「橋樑書」、「漫畫」，這些是強烈的檢索信號，必須完整保留，不可轉化為抽象詞彙。
    3. **禁止負面詞彙**：`search_keywords` 絕對禁止出現「偷竊、小偷、陰謀、背叛、欺騙、死板、暴力」等具備負面引導的詞彙。
    4. **特定情境轉換**：只有當「吵鬧/不乖」與「睡前/睡覺」同時出現時，代表家長需求是安定情緒的睡前故事。

    【🚨 注音標籤特殊處理規則】
    1. **字面檢查原則**：只有當使用者提問中「確實出現了」『注音』這兩個字時，"注音標籤" 才能設為 "有注音"。
    2. **嚴禁常識推論**：絕對禁止因為出現「屁屁偵探、小學生、低年級」就自動推想要有注音。
    3. **預設寬鬆**：只要提問中沒出現「注音」二字，欄位必須保持為 null。

    【核心規則：保持寬鬆與關鍵字配比】
    1. **標籤極簡原則**：除非使用者「明確提出」要求（如：我要找XX歲、特定型式），否則標籤（filters）請一律保持 null。
    2. **適讀年齡提取限制**：只有出現明確「幾歲」或「幾年級」時才提取；暗示性字眼（如：適合小朋友）則保持 null。
    3. **關鍵字雙軌配比**：`search_keywords` 提供 3-5 個詞。
        - **50% 核心詞/主題**：關於知識、學習、能力發展（如：觀察力、科學、行為養成）。以及保留書名、作者或類型（如：麵包小偷、柴田啟子、武俠小說）。
        - **50% 風格詞**：關於故事氣氛、趣味、體感（如：幽默、冒險、奇幻、溫馨）。
        - 這樣能確保搜尋結果在「教育功能」與「故事趣味」之間取得平衡。

   【重要規則：標籤對齊】
    1. 資料庫過濾器極其嚴格，你輸出的 "filters" 必須「完全符合」以下白名單字串：
        - 適讀年齡: "3～6歲", "6～9歲", "9～12歲", "13～18歲" (注意：波浪號必須為全形「～」)
        - 型式: "繪本・圖畫書", "百科・圖鑑", "漫畫", "讀本・小說・文字書", "操作書・立體書・實驗書"
        - 注音標籤: "有注音", "無注音"
    2. **陣列輸出規範 (重要)**：
       - 若需求涵蓋多個標籤，請以「陣列 [ ]」輸出。
       - **範例 A (跨年齡)**：搜尋「小學生」 -> 適讀年齡: ["6～9歲", "9～12歲"]。
       - **範例 B (多重型式)**：搜尋「百科或漫畫」 -> 型式: ["百科・圖鑑", "漫畫"]。

    【範例修正：落實寬鬆政策】
    輸入: "找適合小一的偵探書" (註：沒提到注音，所以標籤為 null)
    輸出: {{
        "search_keywords": "偵探 觀察力 幽默 冒險 屁屁偵探",
        "filters": {{ "適讀年齡": "6～9歲", "型式": null, "注音標籤": null }}
    }}    
    【範例 1：實體保護與關聯（麵包小偷）】
    輸入: "麵包小偷 新的"
    輸出: {{
        "search_keywords": "麵包小偷 柴田啟子 烘焙 幽默",
        "filters": {{ "適讀年齡": null, "型式": "繪本・圖畫書", "注音標籤": null }}
    }}

    【範例 2：類型保留（武俠）】
    輸入: "武俠 小說 高中生"
    輸出: {{
        "search_keywords": "武俠小說 少年文學 冒險 經典",
        "filters": {{ "適讀年齡": "13～18歲", "型式": "讀本・小說・文字書", "注音標籤": null }}
    }}

    【範例 3：情境組合轉換（睡前）】
    輸入: "找畫畫溫柔一點，適合睡前講，可以讓小孩不要一直吵鬧的"
    輸出: {{
        "search_keywords": "睡前故事 溫柔繪畫 安定情緒 療癒",
        "filters": {{ "適讀年齡": "3～6歲", "型式": "繪本・圖畫書", "注音標籤": null }}
    }}

    【範例 4：自然語意擴散】
    輸入: "小孩最近不想上學"
    輸出: {{
        "search_keywords": "校園適應 分離焦慮 拒學 同儕相處 情感支持",
        "filters": {{ "適讀年齡": null, "型式": null, "注音標籤": null }}
    }}

    
    
    【範例 5：型式誤判防範與標籤對應】
    輸入: "我想找橋樑書"
    輸出: {{
        "search_keywords": "閱讀素養 故事 自主閱讀",
        "filters": {{
            "適讀年齡": "6～9歲",
            "型式": "讀本・小說・文字書",
            "注音標籤": null
        }}
    }}

    【範例 6：複合查詢與標籤對應】
    輸入: "找一本適合小一的恐龍漫畫，要有注音"
    輸出: {{
        "search_keywords": "恐龍 古生物 冒險",
        "filters": {{
            "適讀年齡": "6～9歲",
            "型式": "漫畫",
            "注音標籤": "有注音"
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
        # [確認項目]：這裡傳遞的變數需與 simple_test_001.py 呼叫端對齊
        return chain.invoke({"query": user_query})
    except Exception as e:
        return {"search_keywords": user_query, "filters": {}}

# ================= [修正] Layer 4: 分層排序 (Tiered Sorting) 更新20260217=================
def dual_track_search(query, top_k=20, exact_title_filter=None, metadata_filter=None):
    # --- 模式 A: 直通車 (維持不變) ---
    if exact_title_filter:
        results = db_shell.similarity_search_with_score(
            exact_title_filter, k=1, filter={"Title": exact_title_filter}
        )
        # 為直通車返回預設值，並標註 is_relaxed 為 False
        return [{"doc": doc, "score": 1.0, "matched_via": ["🚀 直通車"], "is_direct_hit": True, "is_strict_match": True} for doc, score in results], False

    # --- [關鍵修正] 清洗與轉換 Filter，支援陣列 OR 搜尋 ($in) ---
    active_filters = {}
    if metadata_filter:
        for k, v in metadata_filter.items():
            if v:  
                if isinstance(v, list):
                    active_filters[k] = {"$in": v}
                else:
                    active_filters[k] = v

    # --- 內部搜尋函式 ---
    # --- 內部搜尋函式 (修正版：加入標籤追蹤) ---
    def fetch(f):
        temp_candidates = {}
        for db, label in [(db_shell, "殼"), (db_core, "核")]:
            try:
                res = db.similarity_search_with_score(query, k=10, filter=f)
                for doc, score in res:
                    bid = doc.metadata.get('ISBN') or doc.metadata.get('Title')
                    if not bid: continue
                    
                    if bid in temp_candidates:
                        # 如果這本書已經在另一個庫被搜到過
                        if score > temp_candidates[bid]["score"]: 
                            temp_candidates[bid]["score"] = score
                        # 將新的來源標籤加入列表 (例如變成 ["殼", "核"])
                        if label not in temp_candidates[bid]["matched_via"]:
                            temp_candidates[bid]["matched_via"].append(label)
                    else:
                        # 第一次搜到這本書
                        temp_candidates[bid] = {
                            "doc": doc, 
                            "score": score, 
                            "is_strict_match": True,
                            "matched_via": [label] # 👈 關鍵：初始化來源標籤
                        }
            except Exception as e:
                print(f"Search Error: {e}")
        return temp_candidates

    # --- 1. 執行第一輪：嚴格搜尋 (Strict Search) ---
    strict_results = fetch(active_filters)
    strict_list = sorted(list(strict_results.values()), key=lambda x: x["score"], reverse=True)
    
    # 判定高品質命中數量 (門檻 0.75)
    high_quality_count = sum(1 for item in strict_list if item['score'] > 0.75)
    
    # --- 2. 判定是否啟動放寬搜尋 (Relaxation) ---
    is_relaxed = False
    broad_results = {}
    droppable_keys = ["型式", "注音", "注音標籤"]
    has_droppable = any(k in active_filters for k in droppable_keys)

    if high_quality_count < 3 and has_droppable:
        is_relaxed = True
        # 放寬篩選：僅保留「適讀年齡」
        relaxed_filter = {"適讀年齡": active_filters["適讀年齡"]} if "適讀年齡" in active_filters else {}
        broad_results = fetch(relaxed_filter)

    # --- 3. [核心移植] 全域排序邏輯 ---
    combined_results = []
    seen_isbns = set()

    # 合併 Strict (門檻 0.68)
    for item in strict_list:
        if item['score'] >= 0.68:
            item['is_strict_match'] = True
            combined_results.append(item)
            seen_isbns.add(item['doc'].metadata.get('ISBN'))

    # 合併 Broad (補書，門檻 0.68)
    if is_relaxed:
        broad_list = sorted(list(broad_results.values()), key=lambda x: x["score"], reverse=True)
        for item in broad_list:
            isbn = item['doc'].metadata.get('ISBN')
            if isbn not in seen_isbns and item['score'] >= 0.68:
                item['is_strict_match'] = False
                combined_results.append(item)
                seen_isbns.add(isbn)
    
    # 全局按分數重新排序，這確保了「誰準誰排第一」
    final_results = sorted(combined_results, key=lambda x: x["score"], reverse=True)

    # 返回結果清單與放寬標記
    return final_results[:15], is_relaxed

# ================= 5. Layer 5: 閱讀夥伴導讀生成器 (正確語法加固版) 更新20260217=================
def layer_5_generate_report(user_query, ai_analysis, search_results, is_relaxed_triggered, llm_model):
    """
    Layer 5: 領讀人模式 - 整合誠實機制、結構化分段與正向防火牆
    """
    if not search_results:
        return "目前書櫃暫時沒有完全適合的書籍，我們正在努力補足中！請您稍等幾天。"

    try:
        # --- [Step 1: 判定提問是否模糊] ---
        safe_filters = ai_analysis.get("filters", {}) if isinstance(ai_analysis, dict) else {}
        active_filter_count = sum(1 for v in safe_filters.values() if v)
        is_vague = True if active_filter_count == 0 else False
        
        # 獲取第一名分數，用來決定開場白語氣
        top_score = search_results[0].get('score', 0)

        # --- [Step 2: 判定開場白 (補書預防針邏輯)] ---
        if is_relaxed_triggered:
            intro_announcement = "目前完全符合需求的書較少，所以我額外推薦類似主題的優選書單，推薦給你和孩子嘗試看看。同時，我也會努力擴充書櫃，再麻煩您稍待。"
        else:
            if top_score > 0.75:
                intro_announcement = "這些是我為您與孩子挑選的優質好書，希望能符合您的需求。"
            else:
                intro_announcement = "目前書櫃暫時沒有完全適合的書籍，但我先挑選一些類似的書單，請參考看看。我們也正在努力補足書櫃中。"

        # --- [Step 3: 素材分發 - 分成核心與驚喜兩群] ---
        core_materials = ""
        surprise_materials = ""
        
        for i, item in enumerate(search_results[:8], 1): 
            meta = item['doc'].metadata
            title = meta.get('Title', '未知')
            expert_view = meta.get('Refine_Content', '這本書深受專家推薦。')
            
            if i <= 5:
                core_materials += f"【書目{i}：{title}】專家導讀：{expert_view}\n"
            else:
                surprise_materials += f"【發現：{title}】\n"

        # --- [新增邏輯：動態生成驚喜區段指令] ---
        surprise_section = ""
        if surprise_materials:
            surprise_section = f"""
           ### 🎨 驚喜發現
           - 簡單以一篇短文說明剩餘書籍的推薦原因，每本書名使用《》包裝：{surprise_materials}
           - 短文格式：每本書名使用書名號《》包裝，每本書的特點用句號分隔。
           - 短文約 100 字。
           """
        # --- [Step 4: 結構化 Prompt 封裝] ---
        # 這裡包含您要求的 250 字短文指令與正向防火牆
        core_instructions = f"""
        【🚨 導讀結構規範】
        1. 正向防火牆：禁止討論偷竊、犯罪等負面意義。請聚焦於書籍的「核」（內在教育或學習價值與心理意義）與「殼」（外在故事情節特色與視覺設計風格）**想要傳達的內容重點。
        2. 結構化分段（務必使用以下 ### 標題）：
           ### 🌟 優選書單
            - **核心任務**：以兒童閱讀專家角度，將素材中排名前五名的書，串聯成一段溫暖且節奏明快的導讀短文。
            - **寫作限制**：
            1. **精簡至上**：每本書僅提取「一個最核心特點」，用一兩句話精準呈現，嚴禁冗長贅述。
            2. **格式要求**：書名請加《》，書與書之間以句號分隔，不需分點編號。
            3. **嚴格邊界**：僅限使用 {core_materials}。有多少寫多少，若只有一本則深入介紹該書。
            4. **字數控制**：這段文字總產出務必控制在 **200-250 字** 以內。
       
          {surprise_section}
        """

        # 根據是否模糊決定導讀模板
        if is_vague:
            prompt_text = f"""
            你是一位在父母圈裡熱心、專業的「閱讀暖心夥伴」，除了幫助有選書困擾的家長解決問題，也用溫暖的文字安慰家長。
            使用者提問："{user_query}" (這是一個較廣的需求)
            
            【任務指令】
            1. **溫暖開場**：請先用一兩句話回應、肯定或安慰家長的擔心或提問，讓家長感受到被理解。
            2. **系統說明**：接續第一點後，請務必帶入這段話：『{intro_announcement}』
            3. **核心推薦**：{core_instructions}
            4. **結尾與互動**：提供一兩句話鼓勵家長，並溫柔追問篩選條件缺少的細節（如孩子年級、興趣、需不需要注音等）。
            5. **限制**：禁止表情符號（除標題外），繁體中文。
            """
        else:
            prompt_text = f"""
            你是一位在父母圈裡熱心、專業的「閱讀暖心夥伴」，除了幫助有選書困擾的家長解決問題，也用溫暖的文字安慰家長。
            使用者需求："{user_query}"
            
            【任務指令】
            1. **溫暖開場**：請先用一兩句話回應、肯定或安慰家長的擔心，展現同理心。
            2. **系統說明**：接續第一點後，請務必帶入這段話：『{intro_announcement}』
            3. **核心推薦**：{core_instructions}
            4. **溫暖總結**：最後提供一兩句話鼓勵與安慰家長。
            5. **限制**：禁止表情符號（除標題外），繁體中文。
            """
        # --- [Step 5: 執行呼叫] ---
        prompt = ChatPromptTemplate.from_messages([("human", "{text}")])
        chain = prompt | llm_model 
        response = chain.invoke({"text": prompt_text})
        return response.content

    except Exception as e:
        return f"🚨 導讀生成失敗：{str(e)}"

# ================= 6. UI 與控制器 (Layer 3 整合) =================
try:
    db_shell, db_core, llm_brain = get_search_engines()
except Exception as e:
    st.error(f"連線失敗: {e}")
    st.stop()

st.title("📚 ibookle 搜尋引擎 (Layer 5)")
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
            semantic_results, is_relaxed_triggered = dual_track_search(
                query=refined_query,
                metadata_filter=extracted_filters
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
            report_text = layer_5_generate_report(
                user_query=query, 
                ai_analysis=ai_analysis, 
                search_results=final_display_list, 
                is_relaxed_triggered=is_relaxed_triggered, 
                llm_model=llm_brain
            )
            
            with st.chat_message("assistant"):
                st.write(report_text)
            
            st.divider()
            # =================================

            st.success(f"根據 AI 分析，為您找到 {len(final_display_list)} 本符合條件的書：")
            
            for rank, item in enumerate(final_display_list, 1):
                doc = item['doc']
                score = item['score']
                # [優化點 1] 來源標籤邏輯：結合核心/外殼與嚴格標籤
                is_strict = item.get('is_strict_match', True)
                raw_sources = item.get('matched_via', [])
            
                # 如果是空清單，給予基礎標籤
                if not raw_sources:
                    sources = ["精準匹配"] if is_strict else ["延伸推薦"]
                else:
                    sources = raw_sources
                    if not is_strict and "(延伸推薦)" not in sources:
                        sources.append("(延伸推薦)")
            
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
                        # [優化點 2] 讓後台資訊更透明
                        source_tags = " | ".join(sources)
                        st.caption(f"數據追蹤標籤: {source_tags} | 分數: {score:.3f} | ISBN: {meta.get('ISBN')}")
                    st.divider()
        else:
            st.warning("找不到結果。可能是篩選條件太嚴格，或是資料庫中沒有符合的書。")