import streamlit as st
import json, os, datetime, gspread, uuid, pytz
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ================= 1. 初始化與環境配置 =================
load_dotenv()

# 設定頁面屬性 (必須是第一個 Streamlit 指令)
st.set_page_config(page_title="ibookle", layout="wide", initial_sidebar_state="expanded")

# 初始化 Session State
if "session_id" not in st.session_state: 
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "last_row_idx" not in st.session_state:
    st.session_state.last_row_idx = None

# 初始化 AI Client
if "GOOGLE_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    client = None

# ================= 2. 函式定義 (放在主程式呼叫前) =================

def get_google_sheet():
    """終極清洗邏輯，確保連線不中斷"""
    try:
        raw_json = st.secrets["GOOGLE_CREDENTIALS"]
        try:
            creds_info = json.loads(raw_json.strip(), strict=False)
        except:
            clean_json = raw_json.replace('\n', '\\n').replace('\r', '\\r')
            creds_info = json.loads(clean_json, strict=False)
            
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
        client_gs = gspread.authorize(creds)
        return client_gs.open("AI_User_Logs").worksheet("Brief_Logs")
    except Exception as e:
        return None

def save_to_log(user_input, ai_response, recommended_books):
    """將搜尋紀錄存入 Google Sheets"""
    try:
        sheet = get_google_sheet()
        if sheet:
            tw_tz = pytz.timezone('Asia/Taipei')
            now_tw = datetime.datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M:%S")
            new_row = [now_tw, st.session_state.session_id, user_input, ai_response, recommended_books, ""]
            sheet.append_row(new_row)
            return len(sheet.get_all_values())
        return None
    except:
        return None

def update_log_feedback():
    """處理 👍/👎 回饋"""
    row_idx = st.session_state.last_row_idx
    if row_idx:
        score = st.session_state.get(f"fb_key_{row_idx}")
        if score is not None:
            try:
                sheet = get_google_sheet()
                feedback_text = "👍" if score == 1 else "👎"
                sheet.update_cell(row_idx, 6, feedback_text)
            except:
                pass

def get_recommendations(user_query):
    """手動截斷維度 (Dimension Fixer) 的搜尋函數"""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        pinecone_key = st.secrets["PINECONE_API_KEY"]
        
        # 原始 Embedding 模型
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key, 
            task_type="retrieval_query"
        )
        
        # 維度修正器：強制將 3072 維切成 768 維
        class DimensionFixer:
            def __init__(self, model): self.model = model
            def embed_query(self, text): return self.model.embed_query(text)[:768]
            def embed_documents(self, texts): return [v[:768] for v in self.model.embed_documents(texts)]

        fixed_embeddings = DimensionFixer(embeddings_model)
        
        vectorstore = PineconeVectorStore(
            index_name="gemini768", 
            embedding=fixed_embeddings, 
            pinecone_api_key=pinecone_key
        )
        
        return vectorstore.similarity_search(user_query, k=5)
    except Exception as e:
        st.error(f"🔍 搜尋引擎連線異常: {e}")
        return None

# ================= 3. 主程式邏輯 =================

# 這裡呼叫 get_google_sheet 就不會報 NameError 了
total_answers = "---"
sheet_for_count = get_google_sheet()
if sheet_for_count:
    try:
        total_answers = len(sheet_for_count.get_all_values()) - 1
    except:
        pass

# CSS 樣式
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    .stTextInput input { border: 2px solid #E67E22 !important; border-radius: 25px !important; }
    .expert-box { margin: 20px 0; padding: 20px; background-color: #FEF9E7; border-left: 5px solid #F39C12; border-radius: 10px; line-height: 1.8; }
    </style>
    """, unsafe_allow_html=True)

# 側邊欄
with st.sidebar:
    st.markdown("## 💡 ibookle 統計")
    st.metric("📊 已解答家長疑問", f"{total_answers} 次")
    st.divider()
    st.info("若有任何建議，歡迎點擊下方按鈕告知我們。")

# 主頁面
st.title("💡 ibookle 繪本共讀專家")
user_query = st.text_input("", placeholder="🔍 輸入孩子最近的狀況 (例如：孩子不愛收玩具...)", key="main_search")

# 觸發搜尋
if user_query and (not st.session_state.search_results or st.session_state.get("prev_query") != user_query):
    with st.spinner("專家正在挑選繪本..."):
        results = get_recommendations(user_query)
        if results:
            titles_str = ", ".join([d.metadata.get('Title','未知') for d in results])
            prompt = f"問題：{user_query}\n書籍：{titles_str}\n請以親子專家口吻簡述推薦原因。約150字。"
            
            try:
                response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
                ai_response = response.text
                
                st.session_state.search_results = {
                    "ai_response": ai_response, 
                    "books": [{
                        "Title": d.metadata.get('Title', '未知'), 
                        "Author": d.metadata.get('Author', '未知'), 
                        "Illustrator": d.metadata.get('Illustrator', '未知'), 
                        "Quick_Summary": d.metadata.get('Quick_Summary', ''), 
                        "Refine_Content": d.metadata.get('Refine_Content', '暫無導讀'), 
                        "Link": d.metadata.get('Link', '')
                    } for d in results]
                }
                st.session_state.prev_query = user_query
                st.session_state.last_row_idx = save_to_log(user_query, ai_response, titles_str)
            except:
                st.error("AI 專家暫時無法回應。")

# 顯示搜尋結果
if st.session_state.search_results:
    res = st.session_state.search_results
    st.markdown(f'<div class="expert-box">{res["ai_response"]}</div>', unsafe_allow_html=True)
    
    st.markdown("### 📖 推薦書單")
    for b in res["books"]:
        with st.container():
            st.subheader(f"《{b['Title']}》")
            st.write(f"作者：{b['Author']} | 繪者：{b['Illustrator']}")
            if b['Quick_Summary']: st.info(b['Quick_Summary'])
            with st.expander("🔍 查看詳細導讀"):
                st.write(b['Refine_Content'])
                if b['Link']: st.link_button("🛒 購書連結", b['Link'])
        st.divider()

    if st.session_state.last_row_idx:
        st.feedback("thumbs", key=f"fb_key_{st.session_state.last_row_idx}", on_change=update_log_feedback)
else:
    st.info("👋 歡迎！請在上方輸入框描述孩子的情況，我將為您推薦合適的共讀繪本。")