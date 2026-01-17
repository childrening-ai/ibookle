import streamlit as st
import os
import json
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="模型救援驗證", layout="centered")

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ API Key 未設定")
    st.stop()

def get_vector_rest(model_name, text):
    """
    通用 REST API 呼叫函式
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:embedContent?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": f"models/{model_name}",
        "content": {"parts": [{"text": str(text)}]}
    }
    
    # 004 需要額外指定 outputDimensionality 避免維度跑掉
    if "004" in model_name:
        payload["outputDimensionality"] = 768

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()['embedding']['values']
        else:
            return f"Error {response.status_code}"
    except Exception as e:
        return f"Exception {e}"

st.title("🚑 模型救援驗證")
st.write("測試：001 與 004 誰能產出「不同」的向量？")

if st.button("開始比對"):
    with st.spinner("正在對決中..."):
        
        # --- 測試 1: embedding-001 ---
        st.subheader("選手 1: embedding-001")
        v1_a = get_vector_rest("embedding-001", "天氣")
        v1_b = get_vector_rest("embedding-001", "恐龍")
        
        if isinstance(v1_a, list) and isinstance(v1_b, list):
            sim_1 = np.dot(v1_a, v1_b)
            st.write(f"天氣 vs 恐龍 相似度: `{sim_1:.6f}`")
            if sim_1 > 0.99:
                st.error("❌ 001 失敗：向量完全一樣 (塌陷)")
            else:
                st.success("✅ 001 成功：向量不同")
        else:
            st.error(f"❌ 001 呼叫失敗: {v1_a}")

        st.divider()

        # --- 測試 2: text-embedding-004 ---
        st.subheader("選手 2: text-embedding-004")
        v2_a = get_vector_rest("text-embedding-004", "天氣")
        v2_b = get_vector_rest("text-embedding-004", "恐龍")
        
        if isinstance(v2_a, list) and isinstance(v2_b, list):
            sim_2 = np.dot(v2_a, v2_b)
            st.write(f"天氣 vs 恐龍 相似度: `{sim_2:.6f}`")
            if sim_2 > 0.99:
                st.error("❌ 004 失敗：向量完全一樣")
            else:
                st.success("✅ 004 成功：向量不同 (數值應在 0.4~0.7 之間)")
        else:
            st.error(f"❌ 004 呼叫失敗: {v2_a}")