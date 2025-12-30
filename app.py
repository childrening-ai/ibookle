import streamlit as st
import json, os, datetime, gspread, uuid, pytz, io
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from PIL import Image, ImageDraw, ImageFont  # 新增圖片處理

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

# ... [保留原本的 get_google_sheet, save_to_log, update_log_feedback, get_recommendations 函式，內容不變] ...

def generate_share_image(query, ai_res, books):
    """生成一張帶有版權字樣的分享圖"""
    # 寬度固定 800，高度根據書籍數量動態計算
    width = 800
    header_h = 250
    item_h = 160
    footer_h = 80
    total_h = header_h + (len(books) * item_h) + footer_h
    
    # 建立畫布 (米色背景增加專業質感)
    img = Image.new('RGB', (width, total_h), color=(252, 251, 247))
    draw = ImageDraw.Draw(img)
    
    # 嘗試載入字體 (若在 Linux Server，請確保路徑正確)
    try:
        # 建議在專案目錄放入一個中文字體檔，如 NotoSansTC-Regular.otf
        font_title = ImageFont.truetype("Arial.ttf", 36)
        font_text = ImageFont.truetype("Arial.ttf", 20)
        font_copy = ImageFont.truetype("Arial.ttf", 16)
    except:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_copy = ImageFont.load_default()

    # 繪製標題與背景裝飾
    draw.rectangle([0, 0, width, 180], fill=(230, 126, 34)) # 橘色頂欄
    draw.text((40, 50), "ibookle 專家選書報告", fill=(255, 255, 255), font=font_title)
    draw.text((40, 110), f"諮詢主題：{query[:30]}...", fill=(255, 255, 255), font=font_text)
    
    # 繪製 AI 觀點
    y = 210
    draw.text((40, y), "💡 專家導讀建議：", fill=(230, 126, 34), font=font_text)
    y += 40
    # 簡單自動換行處理 (AI 回覆取前 120 字)
    wrapped_ai = [ai_res[i:i+35] for i in range(0, min(len(ai_res), 140), 35)]
    for line in wrapped_ai:
        draw.text((40, y), line, fill=(52, 73, 94), font=font_text)
        y += 30
    
    # 繪製書籍卡片
    y += 20
    for b in books:
        draw.rectangle([30, y, 770, y+140], fill=(255, 255, 255), outline=(236, 240, 241), width=2)
        draw.text((50, y+25), f"《{b['Title']}》", fill=(44, 62, 80), font=font_text)
        draw.text((50, y+65), f"⭐ 專家評分：{b['Rating']} / 3.0", fill=(241, 196, 15), font=font_text)
        draw.text((50, y+95), f"推薦理由：{b['Quick_Summary'][:35]}...", fill=(127, 140, 141), font=font_text)
        y += 160

    # --- 加入版權浮水印標籤 (底欄) ---
    draw.rectangle([0, total_h- footer_h, width, total_h], fill=(44, 62, 80))
    draw.text((220, total_h-50), "© 2026 ibookle 專業 AI 導讀系統 - 轉載請註明出處", fill=(255, 255, 255), font=font_copy)

    # 輸出為 Bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ================= 3. UI 介面樣式 (新增浮動按鈕) =================

st.markdown("""
    <style>
    /* ... [保留原本的隱藏元件與按鈕樣式] ... */

    /* 新增：右下角浮動分享按鈕 */
    .float-share-btn {
        position: fixed;
        bottom: 30px;
        right: 30px;
        background-color: #E67E22;
        color: white !important;
        padding: 12px 24px;
        border-radius: 30px;
        text-decoration: none;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        z-index: 1000;
        transition: 0.3s;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .float-share-btn:hover {
        background-color: #D35400;
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    }
    </style>
    
    <a class="float-share-btn" href="#share_zone">
        <span>📤 分享專家報告</span>
    </a>
    """, unsafe_allow_html=True)

# ... [中間側邊欄與搜尋邏輯保持不變] ...

# ================= 5. 結果顯示與分享功能 =================

if st.session_state.search_results:
    res = st.session_state.search_results
    st.markdown(f'<div class="expert-suggestion-text"><b>🤖 專家建議：</b><br>{res["ai_response"]}</div>', unsafe_allow_html=True)
    
    st.markdown("### 📖 精選推薦清單")
    for b in res["books"]:
        with st.container():
            header_text = f"《{b['Title']}》"
            if float(b['Rating']) >= 3.0:
                header_text += " ✨ [專家首選]"
            
            st.subheader(header_text)
            st.caption(f"✍️ 作者：{b['Author']} | 🏷️ 分類：{b['Category']} | ⭐ 推薦指數：{b['Rating']}")
            
            if b['Quick_Summary']: 
                st.info(b['Quick_Summary'])
                
            with st.expander("🔍 點擊查看專家深度導讀"):
                st.markdown(b['Refine_Content'])
            
            if b['Link']: 
                st.link_button(f"🛒 前往購買《{b['Title']}》", b['Link'], use_container_width=True)
        st.divider()

    # --- 分享報告區塊 (新增圖片生成) ---
    st.markdown('<div id="share_zone"></div>', unsafe_allow_html=True) # 錨點
    st.subheader("📤 儲存與分享本次報告")
    st.write("您可以將專家建議保存為精美圖片或文字，方便分享給其他家長。")

    col_img, col_txt = st.columns(2)
    
    with col_img:
        # 僅在按下時生成圖片，節省資源
        if st.button("🖼️ 生成圖片報告 (含版權字樣)", use_container_width=True):
            with st.spinner("正在繪製您的專屬報告圖片..."):
                img_data = generate_share_image(st.session_state.prev_query, res["ai_response"], res["books"])
                st.image(img_data, caption="圖片預覽 (長按圖片可儲存)")
                st.download_button(
                    label="💾 下載圖片報告",
                    data=img_data,
                    file_name=f"ibookle_report_{datetime.date.today().strftime('%m%d')}.png",
                    mime="image/png",
                    use_container_width=True
                )

    with col_txt:
        # 原有的純文字分享
        share_content = f"🌟 ibookle 專家選書報告 🌟\n需求：{st.session_state.prev_query}\n\n💡 專家建議：\n{res['ai_response']}\n\n"
        for i, book in enumerate(res["books"], 1):
            share_content += f"{i}. 《{book['Title']}》 (★{book['Rating']})\n"
        share_content += "\n© 2026 ibookle - 專業童書共讀導讀系統"

        if st.button("📋 生成分享文字", use_container_width=True):
            st.code(share_content, language=None)
            st.toast("文字已生成，可複製分享！", icon="✨")

    # 預留 Pro 功能 (增加計畫書價值)
    with st.expander("🔒 進階功能 (Pro 版預覽)"):
        st.write("✨ **一鍵加入圖書館借閱清單** / **同步至我的書房筆記**")

# ... [保留後續回饋區與 footer] ...