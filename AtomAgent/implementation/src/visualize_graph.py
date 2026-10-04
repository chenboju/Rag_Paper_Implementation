import os
import sys

# 修正路徑，確保能 import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.graph import app

def main():
    print("🎨 Generating Graph Visualization...")
    
    try:
        # 1. 取得 Graph 物件
        graph = app.get_graph()
        
        # 2. 轉換為 Mermaid PNG 二進位資料
        # 這會呼叫外部 API (Mermaid Ink) 來渲染圖片
        png_data = graph.draw_mermaid_png()
        
        # 3. 存檔
        output_file = "atom_agents_structure.png"
        with open(output_file, "wb") as f:
            f.write(png_data)
            
        print(f"✅ Graph saved to: {os.path.abspath(output_file)}")
        print("請打開這張圖片查看你的 Agent 架構！")

    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        print("提示: 如果你是離線環境，請改用 'print_mermaid' 方法。")

if __name__ == "__main__":
    main()