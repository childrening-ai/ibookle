import streamlit as st
import jieba
import json, os, datetime, gspread, uuid, pytz, re, time
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from pinecone import Pinecone
from rapidfuzz import process, fuzz

# ================= 1. 初始化與環境配置 =================
load_dotenv()

st.set_page_config(page_title="ibookle 童書專家", layout="wide", initial_sidebar_state="collapsed")

# 初始化 Session State
if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())[:8]
if "search_results" not in st.session_state: st.session_state.search_results = None
if "last_row_idx" not in st.session_state: st.session_state.last_row_idx = None
if "prev_query" not in st.session_state: st.session_state.prev_query = ""

# API Key 設定
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))

if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)
else:
    st.error("❌ 未偵測到 GOOGLE_API_KEY")
    st.stop()

# ================= 2. 資料快取 (Layer 0 & 1) =================

@st.cache_resource
def get_cache():
    """載入 CSV 並自動挖掘關鍵字建立強力白名單"""
    cache = {"whitelist_tags": set(), "all_book_titles": [], "all_creators": []}
    try:
        if os.path.exists("book_data.csv"):
            df = pd.read_csv("book_data.csv")
            
            # 1. 基礎與進階挖掘 (Jieba)
            tags = set()
            # 合併所有可能的文字欄位
            content_text = ""
            if "Merged_Keywords" in df.columns:
                 content_text += " ".join(df["Merged_Keywords"].dropna().astype(str).tolist()) + " "
            if "Vector_Story_Fun" in df.columns:
                content_text += " ".join(df["Vector_Story_Fun"].dropna().astype(str).tolist()) + " "
            if "Vector_Edu_Function" in df.columns:
                content_text += " ".join(df["Vector_Edu_Function"].dropna().astype(str).tolist())
            
            # 定義停用詞
            stop_words = {
                "跟著", "就此", "而是", "只是", "還有", "讓人", "不僅", "作為", 
                "透過", "雖然", "但是", "因為", "所以", "如果", "其實", "然後",
                "書中", "本書", "內容", "描繪", "介紹", "帶領", "展開"
            }

            if content_text:
                words = jieba.cut(content_text)
                for w in words:
                    if len(w) > 1 and w.strip() and w not in stop_words: 
                        tags.add(w)

            # 加入人工補強通用詞
            tags.update(["恐龍", "友誼", "上學", "科學", "宇宙", "昆蟲", "繪本", "橋樑書", "漫畫", "好書", "推薦", "注音", "大班", "中班", "小班"])
            cache["whitelist_tags"] = tags
            print(f"✅ 白名單建立完成，共 {len(tags)} 個詞")

            # 2. 建立書名與創作者清單
            if "Title" in df.columns:
                cache["all_book_titles"] = df["Title"].dropna().astype(str).tolist()
            
            creators = set()
            if "Author" in df.columns: creators.update(df["Author"].dropna().astype(str).tolist())
            if "Illustrator" in df.columns: creators.update(df["Illustrator"].dropna().astype(str).tolist())
            cache["all_creators"] = list(creators)
        else:
            cache["whitelist_tags"] = {"恐龍", "學校", "繪本"}
    except Exception as e:
        print(f"Cache Error: {e}")
    return cache

CACHE = get_cache()

# ================= 3. Google Sheet 紀錄 =================

def get_google_sheet():
    try:
        if "GOOGLE_CREDENTIALS" in st.secrets:
            raw_json = st.secrets["GOOGLE_CREDENTIALS"]
            creds_info = json.loads(raw_json.strip(), strict=False)
            scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
            return gspread.authorize(creds).open("AI_User_Logs").worksheet("Brief_Logs")
    except: return None

def save_to_log(user_input, ai_response, recommended_books, result_type="BOOK_LIST"):
    try:
        sheet = get_google_sheet()
        if sheet:
            tw_tz = pytz.timezone('Asia/Taipei')
            now_tw = datetime.datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M:%S")
            new_row = [now_tw, st.session_state.session_id, user_input, ai_response, recommended_books, "", result_type]
            sheet.append_row(new_row)
            return len(sheet.get_all_values())
    except: return None

def update_log_feedback():
    row_idx = st.session_state.last_row_idx
    fb_key = f"fb_key_{row_idx}"
    if row_idx and fb_key in st.session_state:
        score = st.session_state[fb_key]
        if score is not None:
            try:
                sheet = get_google_sheet()
                sheet.update_cell(row_idx, 6, "👍" if score == 1 else "👎")
                st.toast("感謝您的鼓勵！" if score == 1 else "感謝回饋！", icon="❤️" if score == 1 else "📝")
            except: pass

# ================= 4. 核心搜尋邏輯 (Layer 0 - 4) =================

def check_age_overlap(user_range, book_age_str):
    if not user_range or not book_age_str: return True
    try:
        nums = re.findall(r"\d+", str(book_age_str))
        if not nums: return True
        b_min = int(nums[0])
        b_max = int(nums[1]) if len(nums) > 1 else 99
        u_min, u_max = user_range
        return not (u_max < b_min or u_min > b_max)
    except: return True

def extract_constraints_with_ai(query):
    """Layer 3: 使用 Gemini 進行精準語意解析"""
    prompt = f"""
    你是 ibookle 的圖書館管理員。請分析：「{query}」
    回傳純 JSON：
    1. age_range (list[int] | null): 轉為數字區間 [min, max]。如 "小二"->[7,8], "幼兒"->[3,6]。
    2. pinyin (bool | null): 明確要注音->true, 不要/無/沒注音->false, 未提->null。
    3. category (str | null): 僅限輸出: "繪本", "漫畫", "橋樑書", "科普圖鑑", "少年小說"。請自動歸類。
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash', contents=prompt,
            config=GenerateContentConfig(response_mime_type="application/json")
        )
        c = json.loads(response.text)
        if c.get("age_range"): c["age_range"] = tuple(c["age_range"])
        return c
    except: return {"age_range": None, "pinyin": None, "category": None}

def layer_0_direct_hit(query):
    """Layer 0: 直通車"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("gemini768")
    
    # ISBN
    clean_query = query.replace("-", "").strip()
    if clean_query.isdigit() and len(clean_query) in [10, 13]:
        res = index.query(vector=[0]*768, filter={"ISBN": {"$eq": clean_query}}, top_k=1, namespace="shell", include_metadata=True)
        if res.matches: return [res.matches[0]]

    # 創作者直通車 (回傳 None 讓 Layer 4 處理)
    if query in CACHE["all_creators"]: return None 
        
    # 書名模糊比對
    if CACHE["all_book_titles"]:
        match = process.extractOne(query, CACHE["all_book_titles"], scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 90:
            res = index.query(vector=[0]*768, filter={"Title": {"$eq": match[0]}}, top_k=1, namespace="shell", include_metadata=True)
            if res.matches: return [res.matches[0]]
    return None

def check_is_functional_pattern(query):
    """Layer 1 輔助：檢查是否為功能性指令 (省 API)"""
    # 年齡特徵
    if re.search(r"(\d+歲)|(小[一二三四五六])|(低年級|中年級|高年級)|(國中|幼兒)|(大班|中班|小班)", query): return True
    # 注音特徵
    if "注音" in query: return True
    # 型式特徵
    if any(k in query for k in ["繪本", "漫畫", "橋樑書", "圖鑑", "小說", "百科"]): return True
    return False

def layer_1_gatekeeper(query):
    """Layer 1: 守門員"""
    # Route A: 課綱
    if re.search(r"小[一二三四五六]|(?:[一二三四五六]年級)", query) and \
       re.search(r"國語|數學|社會|自然|生活|物理|化學|歷史", query):
        return "ROUTE_CURRICULUM"
    
    # Route B: 功能性指令 OR 白名單
    if check_is_functional_pattern(query): return "ROUTE_WHITELIST"
    for tag in CACHE["whitelist_tags"]:
        if tag in query: return "ROUTE_WHITELIST"
            
    return "ROUTE_UNKNOWN"

def layer_2_google_verification(query):
    """Layer 2: Google 驗證"""
    prompt = f"""
    使用者查詢：「{query}」。請利用 Google Search 判斷。
    回傳 JSON: {{ "type": "ambiguous"|"external_book"|"concept", "options": [...], "book_info": {{...}} }}
    若為普通概念或館藏可能有的書，回傳 "type": "concept"。
    """
    try:
        tools = [Tool(google_search=GoogleSearch())]
        response = client.models.generate_content(
            model='gemini-2.0-flash', contents=prompt, config=GenerateContentConfig(tools=tools)
        )
        return json.loads(response.text.replace("```json", "").replace("```", ""))
    except: return {"type": "concept"}

def layer_4_vector_search(query, constraints):
    """Layer 4: 雙軌搜尋 + 規格過濾"""
    # 1. 產生向量
    for _ in range(3):
        try:
            emb = client.models.embed_content(model="models/text-embedding-004", contents=query)
            q_vec = emb.embeddings[0].values
            break
        except: time.sleep(1)
    else: return [], "AI 連線忙碌中"

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("gemini768")
    
    # 2. 雙軌搜尋 (Shell + Core)
    res_shell = index.query(vector=q_vec, top_k=40, namespace="shell", include_metadata=True)
    res_core = index.query(vector=q_vec, top_k=40, namespace="core", include_metadata=True)
    
    # 3. 加權融合
    candidates = {}
    for match in res_shell.matches:
        candidates[match.id] = {"doc": match, "score": match.score * 0.5}
    for match in res_core.matches:
        if match.id in candidates: candidates[match.id]["score"] += match.score * 0.5
        else: candidates[match.id] = {"doc": match, "score": match.score * 0.5}

    all_books = list(candidates.values())
    
    # 4. 應用 Layer 3 濾網 (Smart Filter)
    filtered_books = []
    for item in all_books:
        meta = item["doc"].metadata or {}
        
        # 年齡過濾
        if constraints["age_range"]:
            if not check_age_overlap(constraints["age_range"], meta.get("適讀年齡", "")): continue

        # 注音過濾
        if constraints["pinyin"] is not None:
            has_pinyin = (meta.get("注音標籤") == "有注音")
            if constraints["pinyin"] != has_pinyin: continue

        # 分類過濾
        if constraints["category"]:
            book_cat = str(meta.get("型式", "")) + str(meta.get("Category", ""))
            if constraints["category"] not in book_cat: continue
            
        filtered_books.append(item)
    
    # 5. 例外處理
    final_list = filtered_books
    system_msg = ""
    if not final_list:
        system_msg = "（找不到完全符合條件的書籍，已為您放寬搜尋條件）"
        final_list = all_books # Fallback

    final_list.sort(key=lambda x: x["score"], reverse=True)
    return [x["doc"] for x in final_list[:5]], system_msg

# ================= 5. 主控制器 =================

def get_recommendations_vFinal(user_query):
    # L0
    direct_hit = layer_0_direct_hit(user_query)
    if direct_hit: return direct_hit, "為您找到這本書！", "BOOK_LIST"
        
    # L1
    route = layer_1_gatekeeper(user_query)
    
    # L3 解析 (無論走哪條路，先解析規格總是好的，除非是 Route C 需要先驗證)
    constraints = extract_constraints_with_ai(user_query)

    if route == "ROUTE_CURRICULUM":
        books, msg = layer_4_vector_search(user_query, constraints) 
        return books, "這是配合學校課程的推薦書單：", "CURRICULUM"

    if route == "ROUTE_WHITELIST":
        books, msg = layer_4_vector_search(user_query, constraints)
        return books, msg, "BOOK_LIST"
        
    if route == "ROUTE_UNKNOWN":
        g_res = layer_2_google_verification(user_query)
        if g_res.get("type") == "ambiguous": return g_res, "發現不同含義", "AMBIGUOUS"
        elif g_res.get("type") == "external_book": return g_res, "館藏無此書", "EXTERNAL"
        else:
            books, msg = layer_4_vector_search(user_query, constraints)
            return books, msg, "BOOK_LIST"

    return [], "Error", "ERROR"

# ================= 6. UI 介面 =================

st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    div[data-testid="stStatusWidget"], .stAppViewFooter { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] {
        background-color: #E67E22 !important; border-radius: 50% !important;
        width: 40px !important; height: 40px !important; display: flex !important; justify-content: center !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg { fill: white !important; }
    .expert-suggestion-text { margin: 20px 0; line-height: 1.8; color: #34495E; font-size: 1.05rem; }
    .stTextInput input { border: 2px solid #E67E22 !important; border-radius: 25px !important; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("📊 ibookle 統計")
    total = "---"
    sheet = get_google_sheet()
    if sheet: total = len(sheet.get_all_values()) - 1
    st.metric("已解答", f"{total} 次")
    st.divider()
    st.link_button("📝 問卷回饋", "https://childrening.pse.is/8jjrrl", use_container_width=True)

st.title("💡 ibookle 童書共讀專家")
st.markdown("##### *為每一本好書，找到懂它的家長；為每一個孩子，挑選最好的陪伴。*")

user_query = st.text_input("", placeholder="🔍 輸入關鍵字：小三 自然、恐龍、十朝...", key="main_search")

if user_query and (not st.session_state.search_results or st.session_state.prev_query != user_query):
    with st.spinner("🔍 專家正在分析您的需求..."):
        results, sys_msg, result_type = get_recommendations_vFinal(user_query)
        
        search_data = {"type": result_type, "query": user_query, "sys_msg": sys_msg, "data": results}
        
        if result_type in ["BOOK_LIST", "CURRICULUM"] and results:
            titles = [d.metadata.get('Title','未知') for d in results]
            titles_str = ", ".join(titles)
            prompt = f"""
            你現在是 ibookle 的共讀夥伴。
            使用者查詢："{user_query}" ({sys_msg})
            推薦書單：{titles_str}
            請用溫暖語氣，針對這個主題寫一段約 100 字的導讀建議。
            若系統訊息有提到「放寬條件」，請溫柔解釋。
            """
            try:
                ai_resp = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
                search_data["ai_response"] = ai_resp.text
            except: search_data["ai_response"] = "專家連線忙碌中。"
        
        st.session_state.search_results = search_data
        st.session_state.prev_query = user_query
        save_to_log(user_query, str(result_type), str(results)[:100], result_type)

if st.session_state.search_results:
    res = st.session_state.search_results
    data = res["data"]
    
    if res["type"] in ["BOOK_LIST", "CURRICULUM"]:
        if "ai_response" in res: st.markdown(f'<div class="expert-suggestion-text"><b>💡 共讀夥伴分享：</b><br>{res["ai_response"]}</div>', unsafe_allow_html=True)
        if res["sys_msg"]: st.caption(f"ℹ️ {res['sys_msg']}")
        
        if not data: st.warning("抱歉，找不到符合條件的書籍。")
        else:
            for b in data:
                meta = b.metadata or {}
                with st.container():
                    st.subheader(f"《{meta.get('Title')}》")
                    st.caption(f"✍️ {meta.get('Author')} | ⭐ {meta.get('Expert_Rating')} | 🏷️ {meta.get('Category')}")
                    if meta.get('Quick_Summary'): st.info(meta.get('Quick_Summary'))
                    if meta.get('Link'): st.link_button("🛒 購書連結", meta.get('Link'))
                st.divider()
                
    elif res["type"] == "AMBIGUOUS":
        st.warning(f"🤔 針對「{res['query']}」，發現不同含義：")
        opts = data.get("options", [])
        c1, c2 = st.columns(2)
        if len(opts) >= 2:
            with c1: st.info(f"**{opts[0]['label']}**\n\n{opts[0]['desc']}")
            with c2: st.info(f"**{opts[1]['label']}**\n\n{opts[1]['desc']}")
                
    elif res["type"] == "EXTERNAL":
        info = data.get("book_info", {})
        st.markdown(f"### 🌐 網路資源：{info.get('title')}")
        st.write(info.get('summary'))
        st.markdown("---")
        st.write("雖然館藏無此書，但您可以參考網路資訊。")

    if st.session_state.last_row_idx:
        fb_key = f"fb_key_{st.session_state.last_row_idx}"
        st.write("🌟 滿意這次的搜尋結果嗎？")
        st.feedback("thumbs", key=fb_key, on_change=update_log_feedback)
else:
    st.caption("© 2026 ibookle")