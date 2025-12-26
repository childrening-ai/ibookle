import streamlit as st
import pandas as pd
import json, gspread, datetime, pytz
from oauth2client.service_account import ServiceAccountCredentials
from google import genai
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

# 初始化 Gemini Client (用於 AI 分析)
if "GOOGLE_API_KEY" in st.secrets:
    ai_client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    ai_client = None

if "ai_analysis_result" not in st.session_state:
    st.session_state.ai_analysis_result = ""

# ================= 2. 頁面配置與側邊欄 =================

st.set_page_config(page_title="ibookle 戰情室", layout="wide")
st.title("📊 ibookle 營運戰情室")

with st.sidebar:
    st.header("⚙️ 管理面版")
    
    if st.button("🔄 刷新數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    
    sheet = get_google_sheet_standalone()
    if sheet:
        data = sheet.get_all_records()
        if not data:
            st.warning("目前尚無資料。")
            st.stop()
            
        df = pd.DataFrame(data)
        # 強制轉換時間格式
        df['Time'] = pd.to_datetime(df['Time'])
        
        # 時間篩選
        st.subheader("📅 時間範圍")
        min_date = df['Time'].min().date()
        max_date = df['Time'].max().date()
        date_range = st.date_input("選擇區間", value=(min_date, max_date))
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range

        st.divider()
        
        # 欄位顯示自訂
        st.subheader("👁️ 顯示設定")
        # 這裡列出你試算表真正的欄位名稱
        all_cols = df.columns.tolist()
        selected_cols = st.multiselect("勾選想看的欄位", options=all_cols, default=all_cols)
    else:
        st.stop()

# ================= 3. 資料處理 (核心邏輯：排序與序號) =================

# 1. 篩選日期
mask = (df['Time'].dt.date >= start_date) & (df['Time'].dt.date <= end_date)
filtered_df = df.loc[mask].copy()

# 2. 核心排序：無論是否顯示 Time 欄位，內部都先依時間排序
filtered_df = filtered_df.sort_values(by="Time", ascending=False)

# 3. 生成序號：從 1 開始，與排序後的順序一致
filtered_df.insert(0, '序號', range(1, len(filtered_df) + 1))

# --- KPI 顯示 ---
c1, c2, c3 = st.columns(3)
c1.metric("總搜尋量", len(filtered_df))
c2.metric("不重複用戶", filtered_df.iloc[:, 2].nunique() if len(filtered_df)>0 else 0) # 假設 ID 在第 2 欄
# 假設 Feedback 在第 6 欄
pos_count = len(filtered_df[filtered_df.iloc[:, -1] == "👍"]) if filtered_df.shape[1] > 5 else 0
c3.metric("滿意回饋", pos_count)

st.divider()

# ================= 4. AI 營運診斷區 =================

st.subheader("🤖 AI 營運診斷")
col_btn1, col_btn2 = st.columns([1, 5])

with col_btn1:
    if st.button("🚀 啟動分析", type="primary"):
        if ai_client and not filtered_df.empty:
            with st.spinner("AI 正在閱讀最近紀錄..."):
                # 抓取最近 20 筆提問作為分析素材
                sample_queries = filtered_df['User_Input'].head(20).tolist()
                query_text = "\n".join([f"- {q}" for q in sample_queries])
                
                prompt = f"""你是一位專業的兒童教育與數據分析專家。
                請分析以下家長提問數據並提供精煉診斷：
                
                數據內容：
                {query_text}
                
                請依照格式回覆：
                1. 核心需求熱點：家長最集中的煩惱是什麼？
                2. 搜尋關鍵字建議：建議增加哪些標籤？
                3. 內容缺口預警：有哪些主題目前較難應對？
                4. 社群文案方向：一句能打動這群家長的文案。
                """
                
                try:
                    response = ai_client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
                    st.session_state.ai_analysis_result = response.text
                except Exception as e:
                    st.error(f"AI 分析失敗: {e}")
        else:
            st.warning("無數據可供分析。")

with col_btn2:
    if st.button("🧹 清除分析"):
        st.session_state.ai_analysis_result = ""
        st.rerun()

if st.session_state.ai_analysis_result:
    # 使用 st.write 以純文字自然呈現，解決禁止符號問題
    st.info("💡 AI 診斷報告")
    st.write(st.session_state.ai_analysis_result)
    st.divider()

# ================= 5. 資料表格 (固定順序與防錯) =================

st.subheader("📝 詳細紀錄清單")

# 定義你的「理想顯示順序」(請確保名稱與試算表標題完全一致)
# 我加入了 '序號'，因為它是我們剛剛手動插入的
ideal_order = ['序號', 'Time', 'Session_ID', 'User_Input', 'AI_Response', 'Recommended_Books', 'Feedback']

# 交叉比對：只顯示「使用者勾選」且「理想順序中存在」的欄位
final_display_cols = [c for c in ideal_order if c in selected_cols or c == '序號']

if final_display_cols:
    st.dataframe(
        filtered_df[final_display_cols],
        use_container_width=True,
        hide_index=True, # 隱藏原生的 0, 1, 2 索引，改看我們自製的 1, 2, 3 序號
        column_config={
            "Time": st.column_config.DatetimeColumn("搜尋時間", format="MM-DD HH:mm"),
            "Feedback": "回饋"
        }
    )
else:
    st.warning("請在左側至少勾選一個欄位。")

# 簡單圖表
st.divider()
trend_df = filtered_df.resample('D', on='Time').size().reset_index(name='次數')
st.plotly_chart(px.line(trend_df, x='Time', y='次數', title="每日搜尋趨勢"), use_container_width=True)