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

# 設定頁面屬性
st.set_page_config(page_title="ibookle 童書專家", layout="wide", initial_sidebar_state="collapsed")

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

# ================= 2. 核心函式定義 =================

def get_google_sheet():
    """穩定連線 Google Sheets"""
    try:
        raw_json = st.secrets["GOOGLE_CREDENTIALS"]
        creds_info = json.loads(raw_json.strip(), strict=False)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
        client_gs = gspread.authorize(creds)
        return client_gs.open("AI_User_Logs").worksheet("Brief_Logs")
    except:
        return None

def save_to_log(user_input, ai_response, recommended_books):
    """依照 Time, SessionID, Input, AI, Books, Feedback 順序寫入"""
    try:
        sheet = get_google_sheet()
        if sheet:
            tw_tz = pytz.timezone('Asia/Taipei')
            now_tw = datetime.datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M:%S")
            # 寫入新紀錄，Feedback 預設為空
            new_row = [now_tw, st.session_state.session_id, user_input, ai_response, recommended_books, ""]
            sheet.append_row(new_row)
            return len(sheet.get_all_values())
        return None
    except:
        return None

def update_log_feedback():
    """處理 👍/👎 回饋與即時感謝通知"""
    row_idx = st.session_state.last_row_idx
    fb_key = f"fb_key_{row_idx}"
    if row_idx and fb_key in st.session_state:
        score = st.session_state[fb_key]
        if score is not None:
            try:
                sheet = get_google_sheet()
                feedback_text = "👍" if score == 1 else "👎"
                sheet.update_cell(row_idx, 6, feedback_text)
                
                # 手機版即時感謝通知 (Toast)
                if score == 1:
                    st.toast("感謝您的鼓勵！我們會繼續挑選好書。🌟", icon="❤️")
                else:
                    st.toast("感謝您的回饋，我們會持續改進建議品質。", icon="📝")
            except:
                pass

def get_recommendations(user_query):
    """維度修正器：將 3072 維轉為 768 維以對齊 Pinecone"""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        pinecone_key = st.secrets["PINECONE_API_KEY"]
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key, 
            task_type="retrieval_query"
        )
        class DimensionFixer:
            def __init__(self, model): self.model = model
            def embed_query(self, text): return self.model.embed_query(text)[:768]
            def embed_documents(self, texts): return [v[:768] for v in self.model.embed_documents(texts)]
        fixed_embeddings = DimensionFixer(embeddings_model)
        vectorstore = PineconeVectorStore(index_name="gemini768", embedding=fixed_embeddings, pinecone_api_key=pinecone_key)
        return vectorstore.similarity_search(user_query, k=5)
    except:
        return None

# ================= 3. UI 介面樣式 (手機響應式) =================

st.markdown("""
    <style>
    /* 隱藏預設元件 */
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    div[data-testid="stStatusWidget"], .stAppViewFooter, [data-testid="stDecoration"], [data-testid="stHeader"] { display: none !important; }
    button[title="View fullscreen"] { display: none !important; }
    
    /* 專家回覆框 */
    .expert-box { 
        margin: 20px 0; 
        padding: 25px; 
        background-color: #FEF9E7; 
        border-left: 5px solid #F39C12; 
        border-radius: 10px; 
        line-height: 1.8; 
        color: #5D4037; 
    }
    
    /* 搜尋框樣式 */
    .stTextInput input { border: 2px solid #E67E22 !important; border-radius: 25px !important; }

    /* 問卷區塊背景 */
    .feedback-container {
        background-color: #F8F9F9;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        border: 1px solid #EBEDEF;
    }

    /* 手機版適配 */
    @media (max-width: 640px) {
        .expert-box { padding: 15px !important; font-size: 0.95rem; }
    }
    </style>
    """, unsafe_allow_html=True)

# 側邊欄：統計與連線燈號
with st.sidebar:
    st.header("📊 ibookle 統計")
    total_answers = "---"
    system_status = "🔴 系統連線中..."
    sheet_data = get_google_sheet()
    if sheet_data:
        try:
            total_answers = len(sheet_data.get_all_values()) - 1
            system_status = "🟢 系統正常運作"
        except:
            system_status = "🟡 系統忙碌中"
    st.metric("已解答家長疑問", f"{total_answers} 次")
    st.write(system_status)
    st.divider()
    st.caption("© 2026 ibookle")

# 主標題
st.title("💡 ibookle 童書共讀專家")
st.markdown("##### *為每一本好書，找到懂它的家長；為每一個孩子，挑選最好的陪伴。*")
st.write("你好！我是你的共讀專家。輸入孩子的狀況或想找的主題，我會為你挑選最適合的童書。")

user_query = st.text_input("", placeholder="🔍 例如：想找關於學習分享的童書...", key="main_search")

# ================= 4. 搜尋與生成邏輯 =================

if user_query and (not st.session_state.search_results or st.session_state.get("prev_query") != user_query):
    with st.spinner("🔍 正在為您翻閱書櫃並整理建議..."):
        results = get_recommendations(user_query)
        if results:
            book_titles = [d.metadata.get('Title','未知') for d in results]
            titles_str = ", ".join(book_titles)
            
            # 童書專家語境 Prompt
            prompt = f"使用者問題：{user_query}\n相關童書：{titles_str}\n請以專家口吻給予溫暖建議。約150字，禁表情符號。"
            
            try:
                response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
                ai_response = response.text
                
                st.session_state.search_results = {
                    "ai_response": ai_response, 
                    "books": [{
                        "Title": d.metadata.get('Title', '未知'), 
                        "Author": d.metadata.get('Author', '未知'), 
                        "Illustrator": d.metadata.get('Illustrator', '未知'), 
                        "Category": d.metadata.get('Category', '一般'),
                        "Quick_Summary": d.metadata.get('Quick_Summary', ''), 
                        "Refine_Content": d.metadata.get('Refine_Content', '暫無導讀'), 
                        "Link": d.metadata.get('Link', '')
                    } for d in results]
                }
                st.session_state.prev_query = user_query
                # 存入紀錄並獲取行號
                st.session_state.last_row_idx = save_to_log(user_query, ai_response, titles_str)
            except:
                st.error("AI 專家目前連線不穩，請稍候。")

# ================= 5. 結果顯示 (手機優化佈局) =================

if st.session_state.search_results:
    res = st.session_state.search_results
    st.markdown(f'<div class="expert-box"><b>🤖 專家建議：</b><br>{res["ai_response"]}</div>', unsafe_allow_html=True)
    
    st.markdown("### 📖 精選推薦清單")
    for b in res["books"]:
        with st.container():
            st.subheader(f"《{b['Title']}》")
            st.caption(f"✍️ 作者：{b['Author']} | 🎨 繪者：{b['Illustrator']} | 🏷️ 分類：{b['Category']}")
            
            if b['Quick_Summary']: st.info(b['Quick_Summary'])
                
            with st.expander("🔍 點擊查看專家深度導讀"):
                st.markdown(b['Refine_Content'])
            
            # 獨立購書按鈕 (手機全寬)
            if b['Link']: 
                st.link_button(f"🛒 前往購買《{b['Title']}》", b['Link'], use_container_width=True)
        
        st.write("") 
        st.divider()

    # 互動回饋問卷區
    if st.session_state.last_row_idx:
        fb_key = f"fb_key_{st.session_state.last_row_idx}"
        st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
        
        if fb_key not in st.session_state or st.session_state[fb_key] is None:
            st.write("🌟 這份建議對您有幫助嗎？")
        else:
            st.write("✅ 感謝您的回饋，讓 ibookle 變得更好！")
            
        st.feedback("thumbs", key=fb_key, on_change=update_log_feedback)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("---")
    st.caption("👋 歡迎使用 ibookle！請描述孩子的情況，我們將為您推薦最適合的童書。")