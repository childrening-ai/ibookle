import pandas as pd
import os

# 設定您的 CSV 檔名
CSV_FILE = "ibookle_final_upload_ready.csv"

def analyze_metadata():
    if not os.path.exists(CSV_FILE):
        print(f"❌ 找不到檔案: {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)
    
    print(f"📊 正在分析 {CSV_FILE} 的欄位資料...\n")

    # 1. 檢查 [適讀年齡]
    if '適讀年齡' in df.columns:
        ages = df['適讀年齡'].dropna().unique().tolist()
        print(f"👶 [適讀年齡] 實際有的值 ({len(ages)}種):")
        print(ages)
    else:
        print("❌ 找不到 [適讀年齡] 欄位")
    
    print("-" * 30)

    # 2. 檢查 [型式]
    # 注意：您的 CSV 欄位可能叫 '型式' 或 'Category'，請確認
    col_format = '型式' if '型式' in df.columns else 'Category'
    if col_format in df.columns:
        formats = df[col_format].dropna().unique().tolist()
        print(f"📚 [{col_format}] 實際有的值 ({len(formats)}種):")
        print(formats)
    else:
        print(f"❌ 找不到 [型式] 欄位")

    print("-" * 30)

    # 3. 檢查 [注音]
    # 注意：您的 CSV 欄位可能叫 '注音' 或 '注音標籤'
    col_pinyin = '注音標籤' if '注音標籤' in df.columns else '注音'
    if col_pinyin in df.columns:
        pinyins = df[col_pinyin].dropna().unique().tolist()
        print(f"🔠 [{col_pinyin}] 實際有的值 ({len(pinyins)}種):")
        print(pinyins)
    else:
        print(f"❌ 找不到 [注音] 欄位")

    print("\n✅ 請將上面印出的列表，複製貼上更新到 app_layer3_complete.py 的 VALID_METADATA 中！")

if __name__ == "__main__":
    analyze_metadata()