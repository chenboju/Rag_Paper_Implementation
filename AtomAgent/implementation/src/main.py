import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# --- [修正路徑問題] ---
# 取得目前檔案 (main.py) 的路徑 -> .../src/main.py
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../src
# 取得上一層目錄 (專案根目錄) -> .../AtomLangchainGraphAgent
project_root = os.path.dirname(current_dir)
# 將根目錄加入系統路徑
sys.path.append(project_root)
# ---------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 現在 Python 可以從根目錄找到 'src' 了
from src.core.graph import app

def main():
    # 載入環境變數
    load_dotenv()
    print("🚀 AtomAgents System Initialized (LangGraph)")
    print("------------------------------------------")
    
    # 模擬使用者輸入
    # 這是一個需要調用工具才能回答的問題
    user_input = "我想研究銅 (Cu) 的性質。請先幫我計算它的晶格常數，確認結構穩定後，告訴我結果。"
    
    print(f"👤 User: {user_input}\n")
    
    # 初始化狀態
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "current_step_index": 0,
        "simulation_results": {},
        # 如果 state.py 有定義 error_count 或 plan，這裡最好也初始化，避免 KeyError
        "plan": [],
        "error_count": 0
    }
    
    # 執行 Graph
    try:
        # app.stream 會一步一步回傳執行的過程
        for event in app.stream(initial_state):
            for node_name, value in event.items():
                print(f"\n--- Node: {node_name} ---")
                
                # 顯示該節點產生的最新訊息
                if "messages" in value and len(value["messages"]) > 0:
                    last_msg = value["messages"][-1]
                    
                    # 如果是 AI 的訊息
                    if last_msg.type == "ai":
                        # 檢查是否有呼叫工具
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            print(f"🛠️  Call Tool: {last_msg.tool_calls[0]['name']}")
                            print(f"    Args: {last_msg.tool_calls[0]['args']}")
                        else:
                            print(f"💬 Response: {last_msg.content}")
                    
                    # 如果是 Tool 的執行結果
                    elif last_msg.type == "tool":
                        print(f"📊 Tool Output: {last_msg.content[:200]}...") # 只印前200字避免洗版
    except Exception as e:
        print(f"❌ Execution Error: {e}")

if __name__ == "__main__":
    main()