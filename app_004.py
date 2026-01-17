import streamlit as st
import json, os, datetime, gspread, uuid, pytz
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

# [修改 1] 引入必要的 SDK
# 舊版使用 langchain_google_genai 和 langchain_pinecone
# 新版直接使用原生 SDK，以支援 004 的進階參數 (task_type)
from google import genai
from google.genai import types
from pinecone import Pinecone

# ================= 1. 初始化與環境配置 =================
load_dotenv()

st.set_page_config(page_title="ibookle 004 測試版", layout="wide", initial_sidebar_state="collapsed")

if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())[:8]
if "search_results" not in st.session_state: st.session_state.search_results = None
if "last_row_idx" not in st.session_state: st.session_state.last_row_idx = None
if "prev_query" not in st.session_state: st.session_state.prev_query = ""

# 初始化 Client
# [修改 2] 統一使用 google.genai.Client
if "GOOGLE_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    # 本地測試容錯
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ================= 2. 核心函式定義 =================

def get_google_sheet():
    """穩定連線 Google Sheets (保持不變)"""
    try:
        # 優先讀取 secrets，本地測試可讀取 os.getenv
        if "GOOGLE_CREDENTIALS" in st.secrets:
            raw_json = st.secrets["GOOGLE_CREDENTIALS"]
        else:
            raw_json = os.getenv("GOOGLE_CREDENTIALS_JSON") # 本地測試用
            
        if not raw_json: return None

        creds_info = json.loads(raw_json.strip(), strict=False)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
        return gspread.authorize(creds).open("AI_User_Logs").worksheet("Brief_Logs")
    except: return None

def save_to_log(user_input, ai_response, recommended_books):
    """寫入 Log (保持不變)"""
    try:
        sheet = get_google_sheet()
        if sheet:
            tw_tz = pytz.timezone('Asia/Taipei')
            now_tw = datetime.datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M:%S")
            new_row = [now_tw, st.session_state.session_id, user_input, ai_response, recommended_books, ""]
            sheet.append_row(new_row)
            return len(sheet.get_all_values())
    except: return None

def update_log_feedback():
    """處理回饋 (保持不變)"""
    row_idx = st.session_state.last_row_idx
    fb_key = f"fb_key_{row_idx}"
    if row_idx and fb_key in st.session_state:
        score = st.session_state[fb_key]
        if score is not None:
            try:
                sheet = get_google_sheet()
                sheet.update_cell(row_idx, 6, "👍" if score == 1 else "👎")
                if score == 1: st.toast("感謝鼓勵！🌟", icon="❤️")
                else: st.toast("感謝回饋。", icon="📝")
            except: pass

def get_recommendations_004(user_query):
    """
    [修改 3] 搜尋核心邏輯重寫：適配 004 模型
    """
    try:
        # 1. 初始化 Pinecone (使用原生 SDK)
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY") or st.secrets["PINECONE_API_KEY"])
        # [關鍵] 連接新的 004 專用 Index
        index = pc.Index("ibookle-004") 

        # 2. 生成查詢向量 (Query Embedding)
        # [關鍵] 指定 model='text-embedding-004'
        # [關鍵] 指定 task_type='RETRIEVAL_QUERY' (這是 004 搜尋準確的關鍵)
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=user_query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        # 取得 768 維向量 (API 會自動根據 output_dimensionality 裁切，或預設回傳)
        # 這裡要注意：如果上傳時有指定 768，這裡回傳的預設也是 768 (因為 004 支援 Matryoshka)
        # 但為了保險，我們取前 768 維
        q_vec = response.embeddings[0].values[:768]

        # 3. 執行搜尋 (Pinecone Native Query)
        # 不再使用 LangChain 的 similarity_search，改用 index.query
        # 邏輯：先抓 Top 15，再依照您的「星等排序邏輯」處理
        res = index.query(
            vector=q_vec, 
            top_k=15, 
            include_metadata=True 
        )

        # 4. 處理結果 (邏輯與舊版相同：模糊 vs 精確)
        vague_keywords = ["推薦", "好書", "小學生", "繪本", "介紹", "童書"]
        is_vague = len(user_query.strip()) <= 4 or user_query.strip() in vague_keywords

        candidate_books = []
        for match in res.matches:
            # [注意] 原生 Pinecone 回傳的是 match 物件，不是 Document
            # match.metadata 是一個字典
            meta = match.metadata or {}
            score = match.score
            
            # 處理 Rating (防呆)
            try:
                rating = float(meta.get('Expert_Rating', 0))
            except: rating = 0.0

            candidate_books.append({
                "meta": meta,   # 這裡存 metadata
                "score": score,
                "rating": rating
            })

        if is_vague:
            # 模糊策略：星等優先
            candidate_books.sort(key=lambda x: x['rating'], reverse=True)
        else:
            # 精確策略：星等優先，分數次之 (您原本的邏輯)
            candidate_books.sort(key=lambda x: (x['rating'], x['score']), reverse=True)

        # 回傳前 5 名的 metadata 列表
        return [item["meta"] for item in candidate_books[:5]], is_vague

    except Exception as e:
        st.error(f"004 檢索異常: {e}")
        return [], False

# ================= 3. UI 介面 (保持不變，只改標題以示區別) =================

st.markdown("""
    <style>
    /* CSS 樣式保持與舊版一致，省略以節省篇幅，請直接複製舊版 CSS */
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    div[data-testid="stStatusWidget"], .stAppViewFooter { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { background-color: #9B59B6 !important; } /* 改紫色區分 */
    .stTextInput input:focus { border-color: #9B59B6 !important; box-shadow: 0 0 0 1px #9B59B6 !important; }
    .stTextInput input { border: 2px solid #9B59B6 !important; border-radius: 25px !important; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("🧪 004 測試版")
    st.info("目前使用模型：text-embedding-004")
    st.divider()

st.title("🧪 ibookle 實驗室 (004版)")
st.caption("核心引擎已升級，請測試搜尋準確度")

user_query = st.text_input("", placeholder="🔍 測試看看：天氣、恐龍、情緒...", key="main_search")

# ================= 4. 搜尋與生成 (串接新函式) =================

if user_query and (not st.session_state.search_results or st.session_state.prev_query != user_query):
    with st.spinner("🔍 004 引擎運算中..."):
        # [修改 4] 呼叫新的搜尋函式
        results, is_vague_mode = get_recommendations_004(user_query)
        
        if results:
            # 資料結構微調：results 現在是 metadata 的 list
            titles = [m.get('Title', '未知') for m in results]
            titles_str = ", ".join(titles)

            # Prompt 邏輯保持不變
            if is_vague_mode:
                prompt = f"""使用者問模糊問題"{user_query}"。已選經典書：{titles_str}。請以專家身分回覆：1.開頭說「您好！」2.說明問題廣，先推薦首選。3.詢問細節。4.語氣親切150字。"""
            else:
                prompt = f"""使用者需求：{user_query}。精選書：{titles_str}。請以專家身分回覆：1.開頭說「您好！」2.簡述適合原因。3.強調專家導讀。4.語氣親切150字。"""
            
            try:
                # 生成回應
                ai_resp = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
                
                # 存入 Session (結構對齊 UI)
                st.session_state.search_results = {
                    "ai_response": ai_resp.text,
                    "books": results # 直接存 metadata 列表
                }
                st.session_state.prev_query = user_query
                save_to_log(user_query, ai_resp.text, titles_str)
            except Exception as e:
                st.error(f"AI 生成回應失敗: {e}")

# ================= 5. 結果顯示 (適配新的資料結構) =================

if st.session_state.search_results:
    res = st.session_state.search_results
    st.markdown(f'<div class="expert-suggestion-text"><b>🤖 004 專家建議：</b><br>{res["ai_response"]}</div>', unsafe_allow_html=True)
    
    st.markdown("### 📖 搜尋結果")
    for b in res["books"]:
        with st.container():
            # 取值方式改為直接從 dict 取
            rating = float(b.get('Expert_Rating', 0) or 0)
            header = f"《{b.get('Title')}》" + (" ✨ [專家首選]" if rating >= 3.0 else "")
            
            st.subheader(header)
            st.caption(f"✍️ 作者：{b.get('Author')} | ⭐ {rating}")
            
            if b.get('Quick_Summary'): st.info(b.get('Quick_Summary'))
            
            with st.expander("🔍 專家導讀"):
                st.markdown(b.get('Refine_Content', '無內容'))
                # 顯示 Score (僅測試用，正式版可隱藏)
                # st.caption(f"Debug Score: {b.get('score', 'N/A')}") 
            
            if b.get('Link') and b.get('Link') != 'nan':
                st.link_button("🛒 購書", b.get('Link'), use_container_width=True)
        st.divider()

    # (分享功能區塊省略，邏輯相同)