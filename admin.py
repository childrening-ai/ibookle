import streamlit as st
import pandas as pd
import json, gspread, datetime, re
from oauth2client.service_account import ServiceAccountCredentials
from google import genai

# ================= 1. 初始化與密碼鎖定 =================

def check_password():
    def password_entered():
        # ADMIN_PASSWORD 建議設定在 Streamlit Secrets 中，預設為 ibookle_admin
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
    """連線至 Google Sheets 讀取 Dual_Logs"""
    try:
        raw_json = st.secrets["GOOGLE_CREDENTIALS"]
        creds_info = json.loads(raw_json.strip(), strict=False)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
        client_gs = gspread.authorize(creds)
        # 開啟試算表並指定 Dual_Logs 工作表
        return client_gs.open("AI_User_Logs").worksheet("Dual_Logs")
    except Exception as e:
        st.error(f"❌ 試算表連線失敗: {e}")
        return None

def parse_book_list(book_string):
    """解析 2.0 版書單格式: 書名(分數)[標籤] ; 書名(分數)[標籤]"""
    if not book_string or str(book_string) == 'nan' or book_string == "":
        return []
    books = []
    # 使用分號分割
    items = str(book_string).split(" ; ")
    for item in items:
        # 使用正則表達式提取內容
        match = re.search(r"(.+)\(([\d\.]+)\)\[(.+)\]", item)
        if match:
            books.append({
                "title": match.group(1).strip(),
                "score": float(match.group(2)),
                "tag": match.group(3).strip()
            })
    return books

# ================= 3. 主程式介面設定 =================

st.set_page_config(page_title="ibookle 營運戰情室", layout="wide")
st.title("📊 ibookle 營運戰情室 2.0")

if "ai_analysis_result" not in st.session_state:
    st.session_state.ai_analysis_result = ""

# 初始化 Gemini Client
ai_client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"]) if "GOOGLE_API_KEY" in st.secrets else None

# --- 資料讀取與預處理 ---
sheet = get_google_sheet_standalone()
if sheet:
    data = sheet.get_all_records()
    if not data:
        st.warning("📅 目前 Dual_Logs 中尚無資料。")
        st.stop()
    
    df = pd.DataFrame(data)
    
    # 校對欄位名稱：確保「時間」欄位存在並轉換格式
    if '時間' in df.columns:
        df['時間'] = pd.to_datetime(df['時間'], errors='coerce')
        df = df.dropna(subset=['時間']).sort_values(by="時間", ascending=True)
        # 建立序號
        df.insert(0, '序號', range(1, len(df) + 1))
    else:
        st.error(f"⚠️ 找不到 '時間' 欄位。目前的欄位有：{list(df.columns)}")
        st.stop()

    # --- 側邊欄控制面版 ---
    with st.sidebar:
        st.header("⚙️ 管理面版")
        if st.button("🔄 刷新數據", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        st.subheader("📅 時間篩選")
        min_date, max_date = df['時間'].dt.date.min(), df['時間'].dt.date.max()
        # 避免只有單一日期時產生的錯誤
        if min_date == max_date:
            date_range = (min_date, max_date)
            st.info(f"當前資料日期：{min_date}")
        else:
            date_range = st.date_input("選擇統計區間", value=(min_date, max_date))
        
        st.divider()
        st.subheader("🤖 AI 診斷設定")
        analysis_count = st.slider("分析最近幾筆紀錄？", 5, 100, 20)
        
    # 執行時間篩選
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
        mask = (df['時間'].dt.date >= start_d) & (df['時間'].dt.date <= end_d)
        filtered_df = df.loc[mask].copy().sort_values(by="時間", ascending=False)
    else:
        filtered_df = df.copy().sort_values(by="時間", ascending=False)

    # ================= 4. 戰情指標看板 (KPI) =================

    st.subheader("🚀 系統營運關鍵指標")
    c1, c2, c3, c4 = st.columns(4)

    # 指標計算
    total_logs = len(filtered_df)
    thumbs_up = len(filtered_df[filtered_df['Feedback'] == '👍'])
    thumbs_down = len(filtered_df[filtered_df['Feedback'] == '👎'])
    satisfaction = (thumbs_up / (thumbs_up + thumbs_down) * 100) if (thumbs_up + thumbs_down) > 0 else 0

    # 書單與標籤深度解析
    all_books = []
    for bs in filtered_df['推薦書單']:
        all_books.extend(parse_book_list(bs))
    book_df = pd.DataFrame(all_books)

    with c1:
        st.metric("累積諮詢次數", f"{total_logs} 次")
    with c2:
        st.metric("滿意度 (👍)", f"{satisfaction:.1f} %", delta=f"👍{thumbs_up} / 👎{thumbs_down}")
    with c3:
        if not book_df.empty:
            # 統計包含關鍵字「精準」或「直通」或火箭符號的核心命中
            strict_hits = len(book_df[book_df['tag'].str.contains("精準|直通|🚀", na=False)])
            strict_pct = (strict_hits / len(book_df) * 100)
            st.metric("核心庫存命中率", f"{strict_pct:.1f} %")
        else:
            st.metric("核心庫存命中率", "0 %")
    with c4:
        # 統計補書標記為 True (代表觸發放寬搜尋)
        relaxed_count = len(filtered_df[filtered_df['補書標記'] == True])
        st.metric("放寬搜尋次數", f"{relaxed_count} 次")

    st.divider()

    # ================= 5. AI 營運診斷報告 =================

    st.subheader("🤖 AI 自動化診斷")
    col_btn1, col_btn2 = st.columns([1, 5])

    with col_btn1:
        if st.button("🚀 啟動分析", type="primary"):
            if ai_client and not filtered_df.empty:
                with st.spinner(f"正在分析最近 {analysis_count} 筆紀錄..."):
                    sample = filtered_df.head(analysis_count)
                    text_data = []
                    for _, row in sample.iterrows():
                        text_data.append(f"家長提問: {row['原始提問']} | AI優化: {row['AI優化關鍵字']}")
                    
                    prompt = f"""
                    你是一位專業的圖書館營運顧問，請分析以下 {analysis_count} 筆 ibookle 搜尋紀錄：
                    {chr(10).join(text_data)}
                    
                    請給出：
                    1. 使用者需求趨勢分析 (家長最近關心什麼？)
                    2. AI 轉譯精準度評價。
                    3. 內容庫存建議 (根據提問，資料庫可能缺哪些類型的書？)。
                    """
                    try:
                        response = ai_client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
                        st.session_state.ai_analysis_result = response.text
                    except Exception as e:
                        st.error(f"AI 分析失敗: {e}")

    with col_btn2:
        if st.button("🧹 清除報告"):
            st.session_state.ai_analysis_result = ""
            st.rerun()

    if st.session_state.ai_analysis_result:
        st.info("💡 管理員專屬診斷報告")
        st.markdown(st.session_state.ai_analysis_result)

    st.divider()

    # ================= 6. 詳細紀錄清單 =================

    st.subheader("📝 詳細數據清單")

    # 定義顯示欄位順序 (中文欄位)
    display_cols = ['序號', '時間', '原始提問', 'AI優化關鍵字', '專家建議內容', '推薦書單', '補書標記', 'Feedback']

    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "序號": st.column_config.NumberColumn(width="small"),
            "時間": st.column_config.DatetimeColumn("時間", format="MM-DD HH:mm"),
            "原始提問": st.column_config.TextColumn("家長提問", width="medium"),
            "AI優化關鍵字": st.column_config.TextColumn("AI 優化", width="small"),
            "專家建議內容": st.column_config.TextColumn("導讀報告", width="large"),
            "推薦書單": st.column_config.TextColumn("書單詳情", width="medium"),
            "補書標記": st.column_config.CheckboxColumn("放寬?"),
            "Feedback": st.column_config.TextColumn("回饋", width="small")
        }
    )

    # ================= 7. 熱門推薦排行分析 =================
    if not book_df.empty:
        st.divider()
        st.subheader("🔥 本月熱門推薦書籍 Top 10")
        top_books = book_df['title'].value_counts().head(10).reset_index()
        top_books.columns = ['書名', '被推薦次數']
        
        c_table, c_chart = st.columns([1, 1])
        with c_table:
            st.table(top_books)
        with c_chart:
            # 簡單的長條圖呈現
            st.bar_chart(data=top_books.set_index('書名'))

else:
    st.error("無法載入試算表，請確認連線設定與 Secrets 設定。")