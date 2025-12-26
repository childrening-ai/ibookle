import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai  # 使用最新 SDK
import os
import re
from app import get_google_sheet

# ========================================================
# 1. 頁面基本設定與安全檢查
# ========================================================
st.set_page_config(page_title="ibookle 數位戰情室", layout="wide")

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if st.session_state.password_correct:
        return True

    st.title("🔐 ibookle 管理員登入")
    admin_pwd = st.secrets.get("ADMIN_PASSWORD", "admin123")
    pwd = st.text_input("後台授權密碼", type="password")
    if st.button("確認進入"):
        if pwd == admin_pwd:
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("❌ 密碼錯誤")
    return False

# ========================================================
# 2. 數據抓取 (含緩存機制)
# ========================================================
@st.cache_data(ttl=600)
def fetch_logs():
    try:
        sheet_obj = get_google_sheet() # 調用 app.py 的連線
        data = sheet_obj.get_all_records()
        df = pd.DataFrame(data)
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
        return df
    except Exception as e:
        st.error(f"❌ 資料對接失敗: {e}")
        return pd.DataFrame()

# ========================================================
# 3. 戰情室主介面
# ========================================================
if check_password():
    st.title("📊 ibookle 數位戰情室")
    
    tab_ops, tab_health = st.tabs(["📈 營運用戶分析", "🛡️ 資料庫健康度"])

    with tab_ops:
        df_logs = fetch_logs()
        
        if not df_logs.empty:
            # --- KPI 看板 ---
            c1, c2, c3 = st.columns(3)
            c1.metric("累計搜尋", len(df_logs))
            c2.metric("總點讚數", len(df_logs[df_logs['Feedback'] == '👍']))
            rate = (len(df_logs[df_logs['Feedback'].isin(['👍', '👎'])]) / len(df_logs) * 100) if len(df_logs) > 0 else 0
            c3.metric("用戶互動率", f"{rate:.1f}%")

            st.divider()

            # --- 🤖 Gemini 童書專家分析 (最新 SDK 版) ---
            st.subheader("💡 童書專家營運深度診斷")
            if st.button("啟動 AI 專家分析"):
                with st.spinner("專家正在審閱最近 50 筆日誌..."):
                    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
                    
                    # 準備數據 (包含 SessionID, Input, AI 推薦語, Feedback)
                    recent_data = df_logs[['Time', 'SessionID', 'Input', 'AI', 'Feedback']].tail(50).to_string()
                    
                    prompt = f"""
                    你是 ibookle 首席童書專家。請分析以下 50 筆搜尋紀錄：
                    {recent_data}
                    
                    請撰寫一份專家報告，結構如下：
                    [報告標題]: (請給這份分析一個標題)
                    [痛點分析]: (分析家長的搜尋意圖與年齡層)
                    [推薦稽核]: (評估 AI 推薦語是否具備專家溫度)
                    [優化建議]: (針對負評或空白結果給出具體行動建議)
                    """
                    
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.0-flash',
                            contents=prompt
                        )
                        # 清洗 Markdown 代碼塊
                        result = response.text.replace("```markdown", "").replace("```", "").strip()
                        st.markdown(f'<div style="background-color:#F0F2F6; padding:25px; border-radius:12px; border-left: 5px solid #E67E22;">{result}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"AI 分析執行失敗: {e}")

            st.divider()

            # --- 流量趨勢圖 ---
            st.subheader("📈 搜尋流量趨勢")
            daily_trend = df_logs.resample('D', on='Time').size().reset_index(name='次數')
            fig = px.area(daily_trend, x='Time', y='次數', color_discrete_sequence=['#E67E22'])
            st.plotly_chart(fig, use_container_width=True)

            # --- 資料明細 ---
            st.subheader("📋 最近搜尋明細")
            st.dataframe(df_logs.tail(20), use_container_width=True)
        else:
            st.info("目前還沒有日誌數據喔！")

    with tab_health:
        st.subheader("🛡️ 資料庫健康診斷預留區")
        st.info("這裡是未來擴充的功能，例如自動檢查 ISBN 遺漏或爬蟲失敗率。")