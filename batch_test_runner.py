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
    
    # --- 第 1 類：影視連動與時事 ---
    "看完屁屁偵探星星與月亮電影，小孩一直在問還有沒有類似的偵探書",
    "神奇柑仔店電影版那個紅子，想找裡面有出現的魔法點心書",
    "過年要到了，有沒有那種教小朋友除夕習俗但不要太死板的繪本",
    "寒假出國想在飛機上看的，不要太重、有很多小機關可以讓小孩安靜很久的那種",
    "2026寒假推薦書單小一",
    
    # --- 第 2 類：長語句與焦慮抱怨 ---
    "我兒子現在小二，寒假只會一直看電視，有沒有那種內容很吸引人讓他願意放下遙控器的橋樑書",
    "拜託推薦我那種可以讓小孩專注力變好一點的書，他做事都拖拖拉拉的，老是在發呆",
    "想找適合高年級男生看的，他現在卡在橋樑書進不去少年小說，字太多的他就喊累",
    "有沒有那種可以跟小孩討論為什麼要誠實的書？他最近會為了玩平板說謊",
    "那種畫畫溫柔一點、適合睡前講，可以讓小孩情緒穩定下來不要一直吵鬧的",
    
    # --- 第 3 類：不完整快搜 (測試容錯與補書) ---
    "小一 恐龍 漫畫 有注音",
    "身體 秘密 兒童 百科",
    "武俠 小說 高中生",
    "注音 練習 趣味",
    "麵包小偷 新的",
    
    # --- 第 4 類：心理與情境 ---
    "小孩最近怕黑不敢一個人睡",
    "如何教小孩跟朋友吵架要怎麼和解",
    "被同學排擠心情不好",
    "換牙 恐懼 繪本",
    "小孩不愛刷牙怎麼辦",
    
    # --- 第 5 類：低年級與銜接 ---
    "找適合小一新生的，字不要太多，要有注音，最好是那種可以分兩三次講完的故事",
    "大班準備上小一，想要那種生活常規的繪本，或是讓他不排斥看字的小書",
    "小二男生，已經可以看一點橋樑書了，想找幽默一點、像麵包小偷那種感覺的",
    "低年級小孩適合看的科普，圖要大，文字簡單一點",
    
    # --- 第 6 類：中年級與自主閱讀 ---
    "小三小四的孩子，想找系列書讓他養成閱讀習慣，現在已經不看繪本了",
    "有沒有適合中年級女生，講朋友之間的人際關係、或是學校生活的小說",
    "我女兒小三，對科學實驗很有興趣，想找那種步驟清楚、在家可以自己玩的實驗書",
    "想找適合九歲小孩看的冒險故事，最好是這兩年出的新書",
    
    # --- 第 7 類：高年級與深度閱讀 ---
    "五六年級的孩子，想找那種有點社會議題、或是像少年文學那種比較有深度的作品",
    "高年級男生喜歡看漫畫，有沒有那種漫畫轉小說，可以引導他慢慢看長篇文字的橋樑",
    "適合11、12歲看的，那種關於自我認同或是成長心理的書",
    "推薦給升國一的，寒假想給他看一些經典改編，或是歷史類的讀物",
    
    # --- 第 8 類：模糊與多重年齡 ---
    "家裡有兩個小孩，小一和小三，有沒有那種可以共讀，兩個都聽得懂的科普讀物",
    "想找小學生會瘋掉的那種笑話書，寒假想讓他們在家裡開心一下",
    "適合國小中高年級看的，關於AI或是未來科技的介紹",

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

    【🚨 實體保護與負面屏蔽規則 (優先等級最高)】
    1. **實體不可拆解**：若提問包含專有名詞（如：麵包小偷、野貓軍團、神奇柑仔店），請將其視為不可分割的實體。**絕對禁止**拆解字面意義（例如：禁止將『麵包小偷』提取為「偷竊」或「犯罪」主題）。
    2. **類型詞保留**：如「武俠」、「橋樑書」、「漫畫」，這些是強烈的檢索信號，必須完整保留，不可轉化為抽象詞彙（如將武俠轉為陰謀）。
    3. **禁止負面詞彙**：`search_keywords` 絕對禁止出現「偷竊、小偷、陰謀、背叛、欺騙、死板、暴力」等具備負面引導的詞彙。
    4. **特定情境轉換**：只有當「吵鬧/不乖」與「睡前/睡覺」同時出現時，代表家長需求是安定情緒的睡前故事。

    【🚨 注音標籤特殊處理規則】
    1. **字面檢查原則**：只有當使用者提問中「確實出現了」『注音』這兩個字時，"注音標籤" 才能設為 "有注音"。
    2. **嚴禁常識推論**：絕對禁止因為出現「屁屁偵探、小學生、低年級」就自動推想要有注音。
    3. **預設寬鬆**：只要提問中沒出現「注音」二字，欄位必須保持為 null。

    【核心規則：保持寬鬆與關鍵字配比】
    1. **標籤極簡原則**：除非使用者「明確提出」要求（如：我要找XX歲、特定型式），否則標籤（filters）請一律保持 null。
    2. **適讀年齡提取限制**：只有出現明確「幾歲」或「幾年級」時才提取；暗示性字眼（如：適合小朋友）則保持 null。
    3. **關鍵字雙軌配比**：`search_keywords` 提供 3-5 個詞。
        - **50% 核心詞/主題**：關於知識、學習、能力發展（如：觀察力、科學、行為養成）。以及保留書名、作者或類型（如：麵包小偷、柴田啟子、武俠小說）。
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

【範例修正：落實寬鬆政策】
    輸入: "找適合小一的偵探書" (註：沒提到注音，所以標籤為 null)
    輸出: {{
        "search_keywords": "偵探 觀察力 幽默 冒險 屁屁偵探",
        "filters": {{ "適讀年齡": "6～9歲", "型式": null, "注音標籤": null }}
    }}    

    【範例 1：實體保護與關聯（麵包小偷）】
    輸入: "麵包小偷 新的"
    輸出: {{
        "search_keywords": "麵包小偷 柴田啟子 烘焙 幽默",
        "filters": {{ "適讀年齡": null, "型式": "繪本・圖畫書", "注音標籤": null }}
    }}

    【範例 2：類型保留（武俠）】
    輸入: "武俠 小說 高中生"
    輸出: {{
        "search_keywords": "武俠小說 少年文學 冒險 經典",
        "filters": {{ "適讀年齡": "13～18歲", "型式": "讀本・小說・文字書", "注音標籤": null }}
    }}

    【範例 3：情境組合轉換（睡前）】
    輸入: "找畫畫溫柔一點，適合睡前講，可以讓小孩不要一直吵鬧的"
    輸出: {{
        "search_keywords": "睡前故事 溫柔繪畫 安定情緒 療癒",
        "filters": {{ "適讀年齡": "3～6歲", "型式": "繪本・圖畫書", "注音標籤": null }}
    }}

    【範例 4：自然語意擴散】
    輸入: "小孩最近不想上學"
    輸出: {{
        "search_keywords": "校園適應 分離焦慮 拒學 同儕相處 情感支持",
        "filters": {{ "適讀年齡": null, "型式": null, "注音標籤": null }}
    }}

    
    
    【範例 5：型式誤判防範與標籤對應】
    輸入: "我想找橋樑書"
    輸出: {{
        "search_keywords": "閱讀素養 故事 自主閱讀",
        "filters": {{
            "適讀年齡": "6～9歲",
            "型式": "讀本・小說・文字書",
            "注音標籤": null
        }}
    }}

    【範例 6：複合查詢與標籤對應】
    輸入: "找一本適合小一的恐龍漫畫，要有注音"
    輸出: {{
        "search_keywords": "恐龍 古生物 冒險",
        "filters": {{
            "適讀年齡": "6～9歲",
            "型式": "漫畫",
            "注音標籤": "有注音"
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
    # 建立嚴格過濾器 (支援陣列 OR 搜尋)
    active_filters = {k: {"$in": v} if isinstance(v, list) else v for k, v in filters.items() if v}
    
    def fetch(f):
        temp_candidates = {}
        for db, label in [(db_shell, "殼"), (db_core, "核")]:
            res = db.similarity_search_with_score(query, k=10, filter=f)
            for doc, score in res:
                bid = doc.metadata.get('ISBN') or doc.metadata.get('Title')
                if not bid: continue
                if bid in temp_candidates:
                    if score > temp_candidates[bid]["score"]: 
                        temp_candidates[bid]["score"] = score
                else:
                    temp_candidates[bid] = {"doc": doc, "score": score, "is_strict_match": True}
        return temp_candidates

    # --- 1. 執行第一輪：嚴格搜尋 (Strict Search) ---
    strict_results = fetch(active_filters)
    
    # 判定嚴格命中的高品質書目數量
    # 將結果轉為 list 並按分數排序
    strict_list = sorted(list(strict_results.values()), key=lambda x: x["score"], reverse=True)
    high_quality_count = sum(1 for item in strict_list if item['score'] > 0.75)
    
    # --- 2. 判定是否啟動放寬搜尋 (Relaxation) ---
    is_relaxed = False
    broad_results = {}
    
    # 定義可放寬的標籤
    droppable_keys = ["型式", "注音", "注音標籤"]
    has_droppable = any(k in active_filters for k in droppable_keys)

    # 如果高品質書太少，且有可放寬的條件，就啟動第二輪
    if high_quality_count < 3 and has_droppable:
        is_relaxed = True
        # 放寬篩選：僅保留「適讀年齡」
        relaxed_filter = {"適讀年齡": active_filters["適讀年齡"]} if "適讀年齡" in active_filters else {}
        broad_results = fetch(relaxed_filter)

    # --- 3. [修正版] 全域排序邏輯 ---
    # 不再分層，將 [S] 與 [R] 混合後直接按分數高低排
    combined_results = []
    seen_isbns = set()

    # 合併 strict
    for item in strict_list:
        if item['score'] >= 0.68:
            item['is_strict_match'] = True
            combined_results.append(item)
            seen_isbns.add(item['doc'].metadata.get('ISBN'))

    # 合併 broad (補書)
    if is_relaxed:
        broad_list = sorted(list(broad_results.values()), key=lambda x: x["score"], reverse=True)
        for item in broad_list:
            isbn = item['doc'].metadata.get('ISBN')
            if isbn not in seen_isbns and item['score'] >= 0.68:
                item['is_strict_match'] = False
                combined_results.append(item)
                seen_isbns.add(isbn)
    
    # 全局重新排序：誰準誰排第一
    final_results = sorted(combined_results, key=lambda x: x["score"], reverse=True)

    return final_results[:10], is_relaxed

# ================= 核心邏輯函式 (Layer 5 & 自動化測試 v20260209-2) =================

def run_layer_5(user_query, ai_analysis, search_results, is_relaxed_triggered):
    """
    Layer 5: 領讀人模式 - 整合誠實機制、結構化分段與正向防火牆
    """
    if not search_results:
        return "目前書櫃暫時沒有完全適合的書籍，我們正在努力補足中！請您稍等幾天，"

    try:
        # --- [Step 1: 判定與分數獲取] ---
        safe_filters = ai_analysis.get("filters", {}) if isinstance(ai_analysis, dict) else {}
        active_filter_count = sum(1 for v in safe_filters.values() if v)
        is_vague = True if active_filter_count == 0 else False
        
        # 獲取第一名分數
        top_score = search_results[0].get('score', 0)

        # --- [Step 2: 判定開場白 (補書預防針)] ---
        if is_relaxed_triggered:
            intro_announcement = (
                "目前完全符合需求的書較少，所以我額外推薦類似主題的優選書單，推薦給你和孩子嘗試看看。同時，我也會努力擴充書櫃，再麻煩您稍待。"
           
            )
        else:
            if top_score > 0.75:
                intro_announcement = "這些是我為您與孩子挑選優質好書，希望能符合您的需求。"
            else:
                intro_announcement = "目前書櫃暫時沒有完全適合的書籍，但我先挑選一些類似的書單，請參考看看。我們也正在努力補足書櫃，請您稍等幾天。"

        # --- [Step 3: 素材分發] ---
        # 既然 Refine_Content 一定有，我們直接取值，不需預設文字
        # 為了滿足「前五名串聯短文」的需求，我們調整分類邏輯
        core_materials = ""
        surprise_materials = ""
        
        for i, item in enumerate(search_results[:8], 1): # 擴大取樣至前 8 名，分兩群
            meta = item['doc'].metadata
            title = meta.get('Title', '未知')
            expert_view = meta.get('Refine_Content', '') # 直接取值
            
            if i <= 5:
                # 前五名：提供詳細專家建議，供 AI 串聯短文
                core_materials += f"【書目{i}：{title}】專家導讀：{expert_view}\n"
            else:
                # 剩餘書目：歸類為驚喜發現，僅提供書名
                surprise_materials += f"【發現：{title}】\n"

        # --- [Step 4: 結構化 Prompt 封裝] ---
        core_instructions = f"""
        【🚨 導讀結構規範】
        1. 正向防火牆：禁止討論偷竊、犯罪等負面意義。請聚焦於書籍的「核」（內在教育或學習價值與心理意義）與「殼」（外在故事情節特色與視覺設計風格）**想要傳達的內容重點。
        2. 結構化分段（務必使用以下 ### 標題）：
           ### 🌟 優選書單
            - **核心任務**：以兒童閱讀專家角度，將素材中排名前五名的書，串聯成一段溫暖且節奏明快的導讀短文。
            - **寫作限制**：
            1. **精簡至上**：每本書僅提取「一個最核心特點」，用一兩句話精準呈現，嚴禁冗長贅述。
            2. **格式要求**：書名請加《》，書與書之間以句號分隔，不需分點編號。
            3. **嚴格邊界**：僅限使用 {core_materials}。有多少寫多少，若只有一本則深入介紹該書。
            4. **字數控制**：這段文字總產出務必控制在 **200-250 字** 以內。
       
           ### 🎨 驚喜發現
           - 簡單以一篇短文簡短說明所有剩餘書籍的「特點」與推薦原因，每本書名使用書名號《》包裝：{surprise_materials}
           - 短文格式：每本書名使用書名號《》包裝，每本書的特點用句號分隔
           - 短文約 100 字
           - 素材邊界：請「僅根據」提供的素材進行導讀。若 {surprise_materials} 內容不足或為空，請有多少寫多少，嚴禁自行編造或擴充資料庫以外的書目。
           - 誠實反應：若素材中只有一本書，就請專注於那一本的精彩導讀。
        """

        if is_vague:
            prompt_text = f"""
            你是一位在父母圈裡熱心、專業的「閱讀暖心夥伴」，除了幫助有選書困擾的家長解決問題，也用溫暖的文字安慰家長。
            使用者提問："{user_query}" (這是一個較廣的需求)
            
            【任務指令】
            1. **溫暖開場**：請先用一兩句話回應、肯定或安慰家長的擔心或提問，讓家長感受到被理解。
            2. **系統說明**：接續第一點後，請務必帶入這段話：『{intro_announcement}』
            3. **核心推薦**：{core_instructions}
            4. **結尾與互動**：提供一兩句話鼓勵家長，並溫柔追問篩選條件缺少的細節（如孩子年級、興趣、需不需要注音等）。
            5. **限制**：禁止表情符號（除標題外），繁體中文。
            """
        else:
            prompt_text = f"""
            你是一位在父母圈裡熱心、專業的「閱讀暖心夥伴」，除了幫助有選書困擾的家長解決問題，也用溫暖的文字安慰家長。
            使用者需求："{user_query}"
            
            【任務指令】
            1. **溫暖開場**：請先用一兩句話回應、肯定或安慰家長的擔心，展現同理心。
            2. **系統說明**：接續第一點後，請務必帶入這段話：『{intro_announcement}』
            3. **核心推薦**：{core_instructions}
            4. **溫暖總結**：最後提供一兩句話鼓勵與安慰家長。
            5. **限制**：禁止表情符號（除標題外），繁體中文。
            """

        # --- [Step 5: 執行呼叫] ---
        prompt = ChatPromptTemplate.from_messages([("human", "{text}")])
        chain = prompt | llm_brain 
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

            # Step 2: 檢索
            results, is_relaxed_triggered = run_layer_4(refined_q, filters)
            
            # Step 3: 導讀 (修正點：完整傳入 4 個參數)
            report = run_layer_5(
                user_query=query, 
                ai_analysis=ai_ana, 
                search_results=results, 
                is_relaxed_triggered=is_relaxed_triggered
            )
            
            # --- 後續數據收集 ---
            active_filter_count = sum(1 for v in filters.values() if v)
            is_vague_mode = "是" if active_filter_count == 0 else "否"

            book_list = []
            for item in results:
                t = item['doc'].metadata.get('Title', '未知')
                s = round(item['score'], 3)
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
                
            time.sleep(1.5) 
            
        except Exception as e:
            print(f"❌ 處理 '{query}' 時發生錯誤: {e}")

    # 輸出結果
    df = pd.DataFrame(all_data)
    file_path = "ibookle_test_report_layer5_final_v20260209-2.csv"
    file_exists = os.path.isfile(file_path)
    df.to_csv(file_path, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
    
    print(f"\n✅ 測試完成！數據已成功追加至 {file_path}")

if __name__ == "__main__":
    start_batch_test()