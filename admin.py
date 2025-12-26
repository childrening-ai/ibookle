import streamlit as st
import pandas as pd
import json, gspread, datetime
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px

# ================= 1. 初始化與連線 =================

def get_google_sheet_standalone():
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
        st.error(f"❌ 試算表連線失敗: {e}")
        return None

# 初始化 AI 分析結果的 Session State
if "ai_analysis_result" not in st.session_state:
    st.session_state.ai_analysis_result = ""

# ================= 2. 頁面配置 =================

st.set_page_config(page_title="ibookle 戰情室", layout="wide")
st.title("📊 ibookle 營運戰情室")

# --- 側邊欄控制區 ---
with st.sidebar:
    st.header("⚙️ 管理面版")
    
    if st.button("🔄 刷新最新數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    
    sheet = get_google_sheet_standalone()
    if sheet:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        # 轉換時間 (請確保你的標題是 Timestamp)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        st.subheader("📅 時間篩選")
        min_date = df['Timestamp'].min().date()
        max_date = df['Timestamp'].max().date()
        date_range = st.date_input("選擇日期範圍", value=(min_date, max_date))
        
        # 處理日期選擇（避免只選一個日期時報錯）
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range

        st.divider()
        st.subheader("👁️ 欄位顯示")
        selected_cols = st.multiselect("選擇顯示資訊", options=df.columns.tolist(), default=df.columns.tolist())
    else:
        st.stop()

# ================= 3. 資料篩選與 AI 分析 =================

if not df.empty:
    mask = (df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)
    filtered_df = df.loc[mask]

    # --- KPI 區塊 ---
    c1, c2, c3 = st.columns(3)
    c1.metric("總搜尋次數", len(filtered_df))
    c2.metric("不重複使用者", filtered_df.iloc[:, 1].nunique() if len(filtered_df)>0 else 0)
    # 假設第 6 欄是回饋
    pos_fb = len(filtered_df[filtered_df.iloc[:, 5] == "👍"]) if filtered_df.shape[1] > 5 else 0
    c3.metric("滿意回饋", pos_fb)

    st.divider()

    # --- AI 診斷分析區 ---
    st.subheader("🤖 AI 營運診斷")
    col_ai_btn1, col_ai_btn2 = st.columns([1, 5])
    
    with col_ai_btn1:
        if st.button("🚀 啟動分析", type="primary"):
            # 這裡簡單模擬 AI 分析邏輯，你可以接入你的 Gemini Client
            recent_queries = filtered_df.iloc[:10, 2].tolist() # 抓前10筆提問
            analysis_text = f"【診斷報告 - {datetime.datetime.now().strftime('%H:%M')}】\n\n"
            analysis_text += f"1. 搜尋熱度：目前選定區間共有 {len(filtered_df)} 筆資料。\n"
            analysis_text += f"2. 用戶關注點：從最近的提問「{', '.join(recent_queries[:3])}」來看，家長主要關心行為習慣與情緒引導。\n"
            analysis_text += "3. 優化建議：可以增加更多關於「情緒繪本」的標籤，這類搜尋轉化率較高。"
            st.session_state.ai_analysis_result = analysis_text

    with col_ai_btn2:
        if st.button("🧹 清除分析內容"):
            st.session_state.ai_analysis_result = ""
            st.rerun()

    # 顯示 AI 分析內容 (純文字模式，無背景設計)
    if st.session_state.ai_analysis_result:
        st.text_area("AI 分析結果", value=st.session_state.ai_analysis_result, height=200, disabled=True)
        # 或者使用 st.write(st.session_state.ai_analysis_result) 若不需要框框

    st.divider()

    # --- 資料表格 ---
    st.subheader("📝 詳細紀錄清單")
    if selected_cols:
        display_df = filtered_df[selected_cols].sort_values(by="Timestamp", ascending=False)
        st.dataframe(display_df, use_container_width=True)
    
    # --- 趨勢圖 ---
    st.divider()
    st.subheader("📈 每日搜尋量")
    trend_df = filtered_df.resample('D', on='Timestamp').size().reset_index(name='次數')
    fig = px.line(trend_df, x='Timestamp', y='次數')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("該區間尚無搜尋紀錄。")