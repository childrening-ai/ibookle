import streamlit as st
import pandas as pd
import json, gspread, datetime, re
from oauth2client.service_account import ServiceAccountCredentials
from google import genai

# ================= 1. 初始化與密碼鎖定 (維持原樣) =================

def check_password():
    def password_entered():
        correct_password = st.secrets.get("ADMIN_PASSWORD", "ibookle_admin")
        if st.session_state["password"] == correct_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("🔐 ibookle 管理員登入")
    st.text_input("請輸入管理員密碼", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 密碼錯誤。")
    return False

if not check_password():
    st.stop()

# ================= 2. 資料連線與解析工具 =================

def get_google_sheet_standalone():
    try:
        raw_json = st.secrets["GOOGLE_CREDENTIALS"]
        creds_info = json.loads(raw_json.strip(), strict=False)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
        client_gs = gspread.authorize(creds)
        # --- 修改點：連向新的試算表與工作表名稱 ---
        return client_gs.open("AI_User_Logs").worksheet("Dual_Logs")
    except Exception as e:
        st.error(f"❌ 試算表連線失敗: {e}")
        return None

def parse_book_list(book_string):
    """
    解析 2.0 版書單格式: 書名(分數)[標籤] ; 書名(分數)[標籤]
    """
    if not book_string or str(book_string) == 'nan':
        return []
    books = []
    # 使用分號分割每本書
    items = str(book_string).split(" ; ")
    for item in items:
        # 正則表達式抓取：書名(分數)[標籤]
        match = re.search(r"(.+)\(([\d\.]+)\)\[(.+)\]", item)
        if match:
            books.append({
                "title": match.group(1),
                "score": float(match.group(2)),
                "tag": match.group(3)
            })
    return books

# ================= 3. 主程式介面與資料過濾 =================

st.set_page_config(page_title="ibookle 管理後台", layout="wide")
st.title("📊 ibookle 營運戰情室 2.0")

if "ai_analysis_result" not in st.session_state:
    st.session_state.ai_analysis_result = ""

ai_client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"]) if "GOOGLE_API_KEY" in st.secrets else None

with st.sidebar:
    st.header("⚙️ 管理面版")
    if st.button("🔄 刷新數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    sheet = get_google_sheet_standalone()
    if sheet:
        data = sheet.get_all_records()
        if not data:
            st.warning("目前尚無資料。")
            st.stop()
        
        df = pd.DataFrame(data)
        # 轉換時間並處理序號
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        df = df.dropna(subset=['Time']).sort_values(by="Time", ascending=True)
        df.insert(0, '序號', range(1, len(df) + 1))
        
        # 時間篩選
        st.subheader("📅 時間範圍")
        min_date, max_date = df['Time'].dt.date.min(), df['Time'].dt.date.max()
        date_range = st.date_input("選擇區間", value=(min_date, max_date))
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range

        st.divider()
        st.subheader("🤖 AI 分析筆數")
        analysis_count = st.slider("分析最近幾筆？", 5, 100, 20)
    else:
        st.stop()

# 篩選資料
mask = (df['Time'].dt.date >= start_date) & (df['Time'].dt.date <= end_date)
filtered_df = df.loc[mask].copy().sort_values(by="Time", ascending=False)

# ================= 4. 戰情指標看板 (2.0 新增) =================

st.subheader("🚀 系統關鍵指標")
c1, c2, c3, c4 = st.columns(4)

total_runs = len(filtered_df)
# 解析回饋率 (假設 👍 為 1)
thumbs_up = len(filtered_df[filtered_df['Feedback'] == 'thumbs_up'])
thumbs_down = len(filtered_df[filtered_df['Feedback'] == 'thumbs_down'])
satisfaction = (thumbs_up / (thumbs_up + thumbs_down) * 100) if (thumbs_up + thumbs_down) > 0 else 0

# 解析核殼比例 (從結果書單中分析)
all_books = []
for bs in filtered_df['Recommended_Books']:
    all_books.extend(parse_book_list(bs))
book_df = pd.DataFrame(all_books)

with c1:
    st.metric("累積諮詢次數", f"{total_runs} 次")
with c2:
    st.metric("使用者滿意度", f"{satisfaction:.1f} %", delta=f"👍{thumbs_up} / 👎{thumbs_down}")
with c3:
    if not book_df.empty:
        strict_pct = (len(book_df[book_df['tag'].str.contains("精準|核心")]) / len(book_df) * 100)
        st.metric("核心庫存命中率", f"{strict_pct:.1f} %")
    else:
        st.metric("核心庫存命中率", "0 %")
with c4:
    relaxed_count = len(filtered_df[filtered_df['Is_Relaxed'] == True])
    st.metric("觸發放寬搜尋", f"{relaxed_count} 次")

st.divider()

# ================= 5. AI 診斷區 =================

st.subheader("🤖 AI 營運診斷")
col_btn1, col_btn2 = st.columns([1, 5])

with col_btn1:
    if st.button("🚀 啟動分析", type="primary"):
        if ai_client and not filtered_df.empty:
            with st.spinner("分析中..."):
                # 改為分析 原始提問 + AI 關鍵字 的對比
                sample = filtered_df.head(analysis_count)
                text_list = []
                for _, row in sample.iterrows():
                    text_list.append(f"提問: {row['User_Input']} -> AI優化關鍵字: {row['AI_Keywords']}")
                
                query_text = "\n".join(text_list)
                prompt = f"你是一位專業教育顧問，請分析以下 {analysis_count} 筆搜尋紀錄：\n{query_text}\n\n請提供：1.家長近期關注的主題趨勢 2.AI關鍵字優化是否準確 3.建議新增的書籍類別。"
                
                try:
                    response = ai_client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
                    st.session_state.ai_analysis_result = response.text
                except Exception as e:
                    st.error(f"分析失敗: {e}")

if st.session_state.ai_analysis_result:
    st.info("💡 AI 診斷報告")
    st.markdown(st.session_state.ai_analysis_result)

st.divider()

# ================= 6. 詳細紀錄清單 (欄位優化) =================

st.subheader("📝 詳細紀錄清單")

# 定義 2.0 版的顯示欄位
# 欄位順序：序號, 時間, 原始提問, AI 關鍵字, 導讀內容, 推薦書單, 狀態, 回饋
display_cols = ['序號', 'Time', 'User_Input', 'AI_Keywords', 'AI_Response', 'Recommended_Books', 'Is_Relaxed', 'Feedback']

st.dataframe(
    filtered_df[display_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "序號": st.column_config.NumberColumn(width="small"),
        "Time": st.column_config.DatetimeColumn("時間", format="MM-DD HH:mm"),
        "User_Input": st.column_config.TextColumn("家長提問", width="medium"),
        "AI_Keywords": st.column_config.TextColumn("AI 優化字", width="small"),
        "AI_Response": st.column_config.TextColumn("導讀報告", width="large"),
        "Recommended_Books": st.column_config.TextColumn("解析後書單", width="medium"),
        "Is_Relaxed": st.column_config.CheckboxColumn("放寬搜尋?"),
        "Feedback": st.column_config.TextColumn("回饋", width="small")
    }
)

# ================= 7. 熱門搜尋書單 (2.0 新增) =================
if not book_df.empty:
    st.subheader("🔥 熱門被推薦書籍排行")
    top_books = book_df['title'].value_counts().head(10).reset_index()
    top_books.columns = ['書名', '被推薦次數']
    st.table(top_books)