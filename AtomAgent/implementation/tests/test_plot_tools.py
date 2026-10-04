import os
# --- [關鍵修正] 解決 OpenMP Error #15 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import os
import json
import pandas as pd

# 將專案根目錄加入路徑，確保能 import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools.plot_tools import plot_and_save_data
from dotenv import load_dotenv

# 載入環境變數 (OpenAI API Key)
load_dotenv()

def create_dummy_csv(filename="test_data.csv"):
    """建立一個簡單的模擬 CSV 檔案"""
    data = {
        "Step": [0, 100, 200, 300, 400, 500],
        "Potential_Energy": [-3.50, -3.55, -3.62, -3.68, -3.70, -3.71],
        "Temperature": [300, 305, 310, 298, 295, 300]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"📄 Created dummy CSV: {filename}")
    return df

def test_generic_plot():
    print("\n🧪 Testing Generic Plotter Tool (plot_and_save_data)...")
    
    # 1. 準備數據 (模擬 Agent 讀取 CSV 的過程)
    csv_file = "test_data.csv"
    create_dummy_csv(csv_file)
    
    # 讀取 CSV 並轉為 JSON 字串 (这是工具要求的输入格式)
    df = pd.read_csv(csv_file)
    data_json = df.to_json() # 或 df.to_dict(orient='list') 後 json.dumps
    
    # 2. 設定繪圖參數
    # 我們故意用自然語言描述，測試 LLM 是否聽得懂
    description = "Plot 'Potential_Energy' vs 'Step' as a blue dashed line with circle markers. Add a title 'Energy Minimization Test' and label axes."
    output_filename = "test_plot_result.png"
    
    print(f"🤖 Invoking LLM to plot data to '{output_filename}'...")
    
    try:
        # 3. 執行工具
        # 注意: 這裡直接呼叫 invoke，就像 Agent 在呼叫它一樣
        result = plot_and_save_data.invoke({
            "data_json": data_json,
            "description": description,
            "filename": output_filename
        })
        
        print(f"📝 Tool Output: {result}")
        
        # 4. 驗證結果
        if os.path.exists(output_filename):
            print(f"✅ Success! Plot generated at: {os.path.abspath(output_filename)}")
        else:
            print("❌ Failed: Output file not found.")
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    test_generic_plot()