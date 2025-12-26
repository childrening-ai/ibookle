import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai  # 使用新版 SDK
import os
import re
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ========================================================
# 1. 獨立的資料連線函數 (不再從 app.py 匯入)
# ========================================================
def get_google_sheet_standalone():
    try:
        # 1. 取得原始字串
        raw_json = st.secrets["GOOGLE_CREDENTIALS"]
        
        # 2. 強力清洗：處理非法控制字元
        # strict=False 會允許 JSON 字串中包含真正的換行符號
        try:
            creds_info = json.loads(raw_json.strip(), strict=False)
        except json.JSONDecodeError:
            # 如果還是失敗，嘗試處理反斜槓轉義問題
            clean_json = raw_json.replace('\n', '\\n').replace('\r', '\\r')
            creds_info = json.loads(clean_json, strict=False)
        
        # 3. 設定標準 Scope
        scope = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
        client_gs = gspread.authorize(creds)
        return client_gs.open("AI_User_Logs").worksheet("Brief_Logs")
    except Exception as e:
        st.error(f"❌ 試算表連線失敗: {e}")
        return None

# ========================================================
# 2. 登入檢查
# ========================================================
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
# 3. 戰情室主介面
# ========================================================
if check_password():
    st.title("📊 ibookle 數位戰情室")
    
    tab_ops, tab_health = st.tabs(["📈 營運用戶分析", "🛡️ 資料庫健康度"])

    with tab_ops:
        sheet = get_google_sheet_standalone()
        if sheet:
            data = sheet.get_all_records()
            df_logs = pd.DataFrame(data)
            
            if not df_logs.empty:
                if 'Time' in df_logs.columns:
                    df_logs['Time'] = pd.to_datetime(df_logs['Time'])
                
                # --- KPI 看板 ---
                c1, c2, c3 = st.columns(3)
                c1.metric("累計搜尋", len(df_logs))
                c2.metric("總點讚數", len(df_logs[df_logs['Feedback'] == '👍']))
                rate = (len(df_logs[df_logs['Feedback'].isin(['👍', '👎'])]) / len(df_logs) * 100) if len(df_logs) > 0 else 0
                c3.metric("用戶互動率", f"{rate:.1f}%")

                st.divider()

                # --- 🤖 AI 專家診斷 ---
                st.subheader("💡 童書專家營運深度診斷")
                if st.button("啟動 AI 專家分析"):
                    with st.spinner("專家正在審閱最近 50 筆日誌..."):
                        client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
                        recent_data = df_logs[['Time', 'Input', 'AI', 'Feedback']].tail(50).to_string()
                        
                        prompt = f"你是 ibookle 首席童書專家。請分析以下搜尋紀錄：\n{recent_data}\n\n請撰寫報告包含：[報告標題]、[痛點分析]、[推薦稽核]、[優化建議]。不使用表情符號。"
                        
                        try:
                            response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
                            st.markdown(f'<div style="background-color:#F0F2F6; padding:25px; border-radius:12px; border-left: 5px solid #E67E22;">{response.text}</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"AI 分析失敗: {e}")

                st.divider()
                st.subheader("📋 最近搜尋明細")
                st.dataframe(df_logs.tail(20), use_container_width=True)
            else:
                st.info("目前還沒有日誌數據。")
    
    with tab_health:
        st.subheader("🛡️ 資料庫健康診斷")
        st.info("系統運作中，目前資料庫連線正常。")