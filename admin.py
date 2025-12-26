import streamlit as st
import pandas as pd
import json, gspread, datetime
from oauth2client.service_account import ServiceAccountCredentials
from google import genai

# ================= 1. 初始化與密碼鎖定 =================

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

# ================= 2. 資料連線與環境設定 =================

def get_google_sheet_standalone():
    try:
        raw_json = st.secrets["GOOGLE_CREDENTIALS"]
        creds_info = json.loads(raw_json.strip(), strict=False)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
        client_gs = gspread.authorize(creds)
        return client_gs.open("AI_User_Logs").worksheet("Brief_Logs")
    except Exception as e:
        st.error(f"❌ 試算表連線失敗: {e}")
        return None

if "ai_analysis_result" not in st.session_state:
    st.session_state.ai_analysis_result = ""

ai_client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"]) if "GOOGLE_API_KEY" in st.secrets else None

# ================= 3. 主程式介面與資料處理 =================

st.title("📊 ibookle 營運戰情室")

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
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        df = df.dropna(subset=['Time'])
        
        # --- 序號邏輯修正 ---
        # 先按時間「從小到大」排，給予永久序號，確保序號 1 是最舊的資料
        df = df.sort_values(by="Time", ascending=True)
        df.insert(0, '序號', range(1, len(df) + 1))
        
        # 時間篩選
        st.subheader("📅 時間範圍")
        min_date, max_date = df['Time'].dt.date.min(), df['Time'].dt.date.max()
        date_range = st.date_input("選擇區間", value=(min_date, max_date))
        
        # 處理日期範圍
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range

        st.divider()
        
        # --- AI 分析筆數設定 ---
        st.subheader("🤖 AI 分析設定")
        analysis_count = st.slider("分析最近幾筆資料？", min_value=5, max_value=100, value=20)

        st.divider()
        st.subheader("👁️ 顯示欄位")
        all_cols = [c for c in df.columns if c != '序號']
        selected_cols = st.multiselect("勾選欄位", options=all_cols, default=all_cols)
    else:
        st.stop()

# --- 資料篩選與排序 (表格顯示最新在上面) ---
mask = (df['Time'].dt.date >= start_date) & (df['Time'].dt.date <= end_date)
filtered_df = df.loc[mask].copy().sort_values(by="Time", ascending=False)

# ================= 4. AI 診斷區 =================

st.subheader("🤖 AI 營運診斷")
col_btn1, col_btn2 = st.columns([1, 5])

with col_btn1:
    if st.button("🚀 啟動分析", type="primary"):
        if ai_client and not filtered_df.empty:
            with st.spinner(f"AI 正在分析最近 {analysis_count} 筆紀錄..."):
                # 根據用戶設定的筆數抓取資料
                sample_queries = filtered_df['Input'].head(analysis_count).tolist()
                query_text = "\n".join([f"- {q}" for q in sample_queries])
                
                prompt = f"你是一位專業教育數據分析師，請分析以下 {analysis_count} 筆家長提問：\n{query_text}\n\n請提供：1.核心需求 2.建議標籤 3.內容缺口 4.社群文案方向。"
                
                try:
                    response = ai_client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
                    st.session_state.ai_analysis_result = response.text
                except Exception as e:
                    st.error(f"分析失敗: {e}")

with col_btn2:
    if st.button("🧹 清除分析"):
        st.session_state.ai_analysis_result = ""
        st.rerun()

if st.session_state.ai_analysis_result:
    st.info("💡 AI 診斷報告")
    st.write(st.session_state.ai_analysis_result)

st.divider()

# ================= 5. 紀錄清單 (橫向捲軸優化) =================

st.subheader("📝 詳細紀錄清單")

# 根據您提供的欄位名稱設定理想順序
ideal_order = ['序號', 'Time', 'SessionID', 'Input', 'AI', 'Books', 'Feedback']
final_display_cols = [c for c in ideal_order if c in selected_cols or c == '序號']

if final_display_cols:
    st.dataframe(
        filtered_df[final_display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "序號": st.column_config.NumberColumn(width="small"),
            "Time": st.column_config.DatetimeColumn("時間", format="MM-DD HH:mm", width="medium"),
            "Input": st.column_config.TextColumn("家長提問", width="large"),
            "AI": st.column_config.TextColumn("AI回覆", width="large"),
            "Books": st.column_config.TextColumn("推薦書單", width="medium"),
        }
    )