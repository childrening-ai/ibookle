import os
import re
import json
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ================= 1. 環境初始化 =================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "ibookle-dual-langchain-001"

# 初始化模型
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
    task_type="retrieval_query",
    output_dimensionality=768
)
db_shell = PineconeVectorStore.from_existing_index(INDEX_NAME, embedding=embeddings, namespace="shell")
db_core = PineconeVectorStore.from_existing_index(INDEX_NAME, embedding=embeddings, namespace="core")
llm_brain = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=GOOGLE_API_KEY)

# ================= 2. 測試集定義 (包含 2026 寒假口語化提語) =================
test_set = [
     # --- 經典直通 ---
    "除了工頭太郎，還有哪些火車迷必看的書",
    "像神奇柑仔店那種有點奇幻又有點啟發的系列",
    "有沒有跟野貓軍團一樣好笑的漫畫",
    "找找看 威利在哪裡 幼兒版",
    "霸王龍系列 推薦"
]

# ================= 3. 核心邏輯函式 (與主程式完全對齊) =================

def run_layer_3(query):
    # 整合了「寬鬆檢索」與「權重平衡」後的 System Prompt
    system_prompt = """
    你是一個精通兒童圖書與家長心理的「ibookle 智慧搜尋助手」。
    你的任務是將使用者的自然語言，精準對齊資料庫的 Metadata 白名單 JSON。

    【核心規則：保持寬鬆 (Broad Search)】
    1. **標籤極簡原則**：為了避免過度篩選，除非使用者「明確提出」要求（如：我要找XX歲、我要有注音的），否則標籤（filters）請一律保持 null。
    2. **注音標籤禁止自動推論**：
       - 禁止因為搜尋「小一、低年級」就自動提取「有注音」。
       - 只有當提問包含「注音」二字時才提取該標籤。
    3. **適讀年齡提取限制**：
       - 只有當提問出現明確「幾歲」或「幾年級」時才提取年齡標籤。
       - 若只是暗示（如：屁屁偵探、橋樑書、適合小朋友），年齡標籤請保持 null。
    4. **關鍵字必須兼顧「核」與「殼」**：
       - `search_keywords` 請固定提供 3-5 個詞。
       - **50% 核心詞**：關於知識、學習、能力發展（如：觀察力、科學、行為養成）。
       - **50% 風格詞**：關於故事氣氛、趣味、體感（如：幽默、冒險、奇幻、溫馨）。
       - 這樣能確保搜尋結果在「教育功能」與「故事趣味」之間取得平衡。

    【重要規則：標籤對齊】
    1. 資料庫過濾器極其嚴格，你輸出的 "filters" 必須「完全符合」以下白名單字串：
        - 適讀年齡: "3～6歲", "6～9歲", "9～12歲", "13～18歲" (注意：波浪號必須為全形「～」)
        - 型式: "繪本・圖畫書", "百科・圖鑑", "漫畫", "讀本・小說・文字書", "操作書・立體書・實驗書"
        - 注音標籤: "有注音", "無注音"
    2. **陣列輸出規範 (重要)**：
       - 若需求涵蓋多個標籤，請以「陣列 [ ]」輸出。
       - **範例 A (跨年齡)**：搜尋「小學生」 -> 適讀年齡: ["6～9歲", "9～12歲"]。
       - **範例 B (多重型式)**：搜尋「百科或漫畫」 -> 型式: ["百科・圖鑑", "漫畫"]。

    【範例 1：型式誤判防範與標籤對應】
    輸入: "我想找橋樑書"
    輸出: {{
        "search_keywords": "閱讀素養 故事 自主閱讀",
        "filters": {{
            "適讀年齡": "6～9歲",
            "型式": "讀本・小說・文字書",
            "注音標籤": "有注音"
        }}
    }}

    【範例 2：複合查詢與標籤對應】
    輸入: "找一本適合小一的恐龍漫畫，要有注音"
    輸出: {{
        "search_keywords": "恐龍 古生物 冒險",
        "filters": {{
            "適讀年齡": "6～9歲",
            "型式": "漫畫",
            "注音標籤": "有注音"
        }}
    }}

    【範例 3：自然語意擴散 (ibookle 核心價值)】
    輸入: "小孩最近不想上學"
    輸出: {{
        "search_keywords": "校園適應 分離焦慮 拒學 同儕相處 情感支持",
        "filters": {{
            "適讀年齡": null,
            "型式": null,
            "注音標籤": null
        }}
    }}

    【輸出要求】
    1. search_keywords 請排除「型式名詞」(如漫畫、繪本)，專注於主題。
    2. 僅回傳 JSON，不要有任何 Markdown 標記或額外解釋。
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    
    chain = prompt | llm_brain | JsonOutputParser()
    
    try:
        # 這裡會完整產出你在主程式看到的優化關鍵字與標籤
        return chain.invoke({"query": query})
    except Exception as e:
        return {"search_keywords": query, "filters": {}}

def run_layer_4(query, filters):
    # 建立嚴格過濾器
    active_filters = {k: {"$in": v} if isinstance(v, list) else v for k, v in filters.items() if v}
    
    def fetch(f):
        temp_candidates = {}
        for db, label in [(db_shell, "殼"), (db_core, "核")]:
            res = db.similarity_search_with_score(query, k=10, filter=f)
            for doc, score in res:
                bid = doc.metadata.get('ISBN') or doc.metadata.get('Title')
                if bid in temp_candidates:
                    if score > temp_candidates[bid]["score"]: temp_candidates[bid]["score"] = score
                else:
                    temp_candidates[bid] = {"doc": doc, "score": score, "is_strict_match": True}
        return temp_candidates

    # 1. 執行第一輪：嚴格搜尋
    strict_results = fetch(active_filters)
    
    # 2. 判斷是否補書 (比照主程式邏輯)
    is_relaxed = False
    final_results = sorted(list(strict_results.values()), key=lambda x: x["score"], reverse=True)
    
    high_quality_count = sum(1 for item in final_results if item['score'] > 0.72)
    droppable_keys = ["型式", "注音", "注音標籤"]
    has_droppable = any(k in active_filters for k in droppable_keys)

    if high_quality_count < 3 and has_droppable:
        is_relaxed = True
        # 放寬篩選：只留年齡
        relaxed_filter = {"適讀年齡": active_filters["適讀年齡"]} if "適讀年齡" in active_filters else {}
        broad_results = fetch(relaxed_filter)
        
        # 合併並標註為 Relaxed
        seen_isbns = {item['doc'].metadata.get('ISBN') for item in final_results}
        for bid, item in broad_results.items():
            if item['doc'].metadata.get('ISBN') not in seen_isbns:
                item['is_strict_match'] = False
                final_results.append(item)
    
    return sorted(final_results, key=lambda x: x["score"], reverse=True)[:10], is_relaxed

def run_layer_5(user_query, ai_analysis, search_results):
    """
    Layer 5: 測試程式專用 - 完整重現主程式導讀邏輯
    """
    if not search_results:
        return "您好！我翻遍了書櫃暫時沒看到完全吻合的書。要不要試著換個說法？"

    # --- [Step 1: 判定邏輯] 確保與主程式一致 ---
    safe_filters = ai_analysis.get("filters", {}) if isinstance(ai_analysis, dict) else {}
    active_filter_count = sum(1 for v in safe_filters.values() if v)
    is_vague = True if active_filter_count == 0 else False

    # --- [Step 2: 素材準備] 定義 context_materials ---
    context_materials = ""
    for i, item in enumerate(search_results[:5], 1):
        meta = item['doc'].metadata
        title = meta.get('Title', '未知')
        expert_view = meta.get('Refine_Content', '這本書深受專家推薦，值得一讀。')
        context_materials += f"【書名{i}】：{title}\n【專家導讀點】：{expert_view}\n\n"

    # --- [Step 3: Prompt 設定] 引用變數 {user_query} 與 {context_materials} ---
    if is_vague:
        prompt_text = f"""
        你是一位在父母圈裡熱心、溫暖且專業的「閱讀夥伴」。
        使用者提問："{user_query}" (這是一個較廣的需求)
        
        【任務指令】
        1. 開頭請說「您好！」，語氣要像老朋友聊天，不要像 AI。
        2. 說明這個主題很棒，你先從 ibookle 口袋名單挑了幾本「專家三星首選」的經典神書。
        3. 根據下方素材，誠懇地介紹這幾本書：
        {context_materials}
        4. 溫柔地追問更多細節（如：孩子的年級、特定興趣），例如：「對了，不知道你們家孩子幾年級了？或是他對這主題有什麼特別好奇的地方嗎？」
        5. 禁止表情符號，約 150-200 字，繁體中文。
        """
    else:
        prompt_text = f"""
        你是一位與家長並肩作戰、懂書也懂孩子的「閱讀夥伴」。
        使用者需求："{user_query}"
        
        【任務指令】
        1. 開頭請說「您好！」，展現出收到需求後「讓我們一起來解決」的熱情。
        2. 根據下方的專家導讀素材，將這幾本書串成一個「共讀建議」：
        {context_materials}
        3. 說明為什麼這組書能精準滿足對方的需求，強調這是結合 ibookle 專家觀點的精選。
        4. 禁止表情符號，語氣平易近人。約 200 字，繁體中文。
        """

    # --- [Step 4: 執行呼叫] ---
    prompt = ChatPromptTemplate.from_messages([("human", "{text}")])
    chain = prompt | llm_brain # 這裡使用你在測試程式開頭定義的 llm_brain
    
    try:
        # 將組好的 prompt_text 傳入
        response = chain.invoke({"text": prompt_text})
        return response.content
    except Exception as e:
        return f"🚨 導讀生成失敗：{str(e)}"

# ================= 4. 自動化執行與數據封裝 =================

def start_batch_test():
    all_data = []
    print(f"🚀 ibookle 測試啟動...")

    for query in tqdm(test_set):
        try:
            start_time = time.time()
            
            # Step 1: AI 分析
            ai_ana = run_layer_3(query)
            refined_q = ai_ana.get("search_keywords", query)
            filters = ai_ana.get("filters", {})

            # Step 2: 檢索 (修正：接收兩個回傳值)
            results, is_relaxed_triggered = run_layer_4(refined_q, filters)
            
            # Step 3: 導讀
            report = run_layer_5(user_query=query, ai_analysis=ai_ana, search_results=results)
            
            # --- 新增：判定模糊模式 (供 CSV 記錄使用) ---
            active_filter_count = sum(1 for v in filters.values() if v)
            is_vague_mode = "是" if active_filter_count == 0 else "否"

            # 數據收集
            book_list = []
            for item in results:
                t = item['doc'].metadata.get('Title', '未知')
                s = round(item['score'], 3)
                # 標註這本書是嚴格命中還是補書
                mark = "[S]" if item.get('is_strict_match', True) else "[R]"
                book_list.append(f"{t}({s}){mark}")

            all_data.append({
                "原始提問": query,
                "AI優化關鍵字": refined_q,
                "AI提取條件": json.dumps(filters, ensure_ascii=False),
                "是否為模糊模式": is_vague_mode, 
                "是否觸發補書": "是" if is_relaxed_triggered else "否",
                "第一名分數": results[0]['score'] if results else 0,
                "前10名平均分": sum(i['score'] for i in results) / len(results) if results else 0,
                "前10本書清單": " | ".join(book_list),
                "導讀報告內容": report,
                "執行耗時": round(time.time() - start_time, 2)
            })
                
            # 防禦性暫停，避免 API 429 錯誤
            time.sleep(1.5) 
            
        except Exception as e:
            print(f"❌ 處理 '{query}' 時發生錯誤: {e}")

    # 輸出結果
    df = pd.DataFrame(all_data)

    # 檢查檔案是否已存在，如果不存在，則寫入標題 (Header)；如果已存在，則不寫標題並續寫
    file_path = "ibookle_test_report.csv"
    file_exists = os.path.isfile(file_path)

    # index=False 不儲存列索引, header=not file_exists 代表只在第一次寫入標題
    df.to_csv(file_path, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
    
    print(f"\n✅ 測試完成！數據已成功追加至 {file_path}")

if __name__ == "__main__":
    start_batch_test()