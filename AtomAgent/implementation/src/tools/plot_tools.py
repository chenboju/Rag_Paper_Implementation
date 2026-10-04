import os
import json
import pandas as pd
from typing import List, Optional, Union, Any
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 引入物理視覺化引擎
try:
    from src.engine.visualizer import Visualizer
except ImportError:
    Visualizer = None
    pass

# 強制先載入環境變數
load_dotenv()

# --- 1. 通用繪圖工具 (LLM Coder) ---

# 初始化 Python 執行環境
python_repl = PythonREPL()
# 專門用來寫畫圖程式的 LLM
coder_llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"), temperature=0)

@tool("plot_and_save_data")
def plot_and_save_data(data_json: str, description: str, filename: str) -> str:
    """
    通用繪圖工具。將數據繪製成圖表並保存為圖片檔案。
    適用於簡單的 XY 曲線、散點圖或長條圖。
    """
    try:
        # 1. 預處理數據
        if isinstance(data_json, str):
            try:
                data_obj = json.loads(data_json)
            except json.JSONDecodeError:
                data_obj = data_json
        else:
            data_obj = data_json

        # 2. 準備 Prompt
        # [關鍵修改] 告訴 LLM 數據已經在 `df` (Pandas DataFrame) 變數裡了
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Python Matplotlib expert. Write a script to plot the given data.
            Rules:
            1. The data is ALREADY loaded into a Pandas DataFrame variable named `df`.
            2. Use `df` directly (e.g., `df.plot()`, `plt.plot(df['col'], ...)`). 
            3. Do NOT create sample data. Do NOT load data from files.
            4. `df` is guaranteed to exist. You can check `df.columns` if unsure, but assume column names match the user's description.
            5. Ensure labels, titles, and legends are added.
            6. SAVE the plot to a file named '{filename}'.
            7. Do NOT use plt.show().
            8. Output ONLY the python code.
            """),
            ("user", "Description: {desc}\n(Data is ready in DataFrame `df`)")
        ])
        
        chain = prompt | coder_llm
        code = chain.invoke({"desc": description, "filename": filename}).content
        
        # 清理 Code
        code = code.replace("```python", "").replace("```", "").strip()
        
        # 3. [關鍵修正] 注入數據並轉為 DataFrame
        # 我們將數據注入變數 `raw_data`，然後立即轉為 `df`，幫 LLM 做好前處理
        header = f"""
import json
import pandas as pd
import matplotlib.pyplot as plt

# Inject data
raw_data = {json.dumps(data_obj)}

# Standardize to DataFrame
try:
    if isinstance(raw_data, list):
        # List of dicts or values
        df = pd.DataFrame(raw_data)
    elif isinstance(raw_data, dict):
        # Dict of lists or dict of dicts
        df = pd.DataFrame(raw_data)
    else:
        # Single value or unknown
        df = pd.DataFrame([raw_data])
except Exception as e:
    print(f"Data conversion warning: {{e}}")
    df = pd.DataFrame() # Fallback

# --- Start of LLM Generated Code ---
"""
        full_code = header + code
        
        # 執行程式碼
        print(f"🎨 Generating generic plot: {filename}...")
        result = python_repl.run(full_code)
        
        # 檢查檔案是否生成
        if os.path.exists(filename):
            return f"Success: Plot saved to {filename}."
        else:
            return f"Error: Code executed but file {filename} was not found.\nCode output: {result}\nGenerated Code: {code[:200]}..."

    except Exception as e:
        return f"Plotting failed: {str(e)}"


# --- 2. 物理專業繪圖工具 (保持不變) ---

class DDMapInput(BaseModel):
    work_dir: str = Field(description="包含模擬檔案的工作目錄路徑")
    pristine_file: str = Field(description="完美晶格結構檔名 (例如 'initial.data')")
    dislocation_dump_file: str = Field(description="含有位錯的 Dump 檔名 (例如 'dump.relaxed.screw')")
    burgers_vector: float = Field(description="柏格斯向量大小 (Angstrom)")
    output_filename: str = Field(default="dd_map.png", description="輸出圖片檔名")

@tool("generate_dd_map", args_schema=DDMapInput)
def generate_dd_map(
    work_dir: str, 
    pristine_file: str, 
    dislocation_dump_file: str, 
    burgers_vector: float, 
    output_filename: str = "dd_map.png"
) -> str:
    """
    [物理專用] 繪製 Differential Displacement Map (DD Map) 以視覺化位錯核心結構。
    """
    if not Visualizer:
        return "Error: Visualizer module not loaded (atomman not found)."

    print(f"🎨 Generating DD Map for {dislocation_dump_file}...")
    try:
        result_path = Visualizer.plot_dd_map(
            work_dir=work_dir,
            pristine_file=pristine_file,
            dislocation_dump_file=dislocation_dump_file,
            burgers_vector=burgers_vector,
            output_filename=output_filename
        )
        if result_path and os.path.exists(result_path):
            return f"Success: DD Map saved to {result_path}"
        else:
            return "Error: Failed to generate DD Map (file not created)."
    except Exception as e:
        return f"Error running Visualizer: {e}"

class NEBPlotInput(BaseModel):
    work_dir: str = Field(description="工作目錄路徑")
    path_energies: List[float] = Field(description="NEB 路徑上的能量列表 (eV)")
    output_filename: str = Field(default="neb_mep.png", description="輸出圖片檔名")

@tool("generate_neb_plot", args_schema=NEBPlotInput)
def generate_neb_plot(
    work_dir: str, 
    path_energies: List[float], 
    output_filename: str = "neb_mep.png"
) -> str:
    """
    [物理專用] 繪製 NEB 最低能量路徑 (MEP) 曲線圖。
    """
    if not Visualizer:
        return "Error: Visualizer module not loaded."

    print(f"🎨 Generating NEB Plot...")
    try:
        result_path = Visualizer.plot_neb_mep(
            work_dir=work_dir,
            path_energies=path_energies,
            output_filename=output_filename
        )
        if result_path and os.path.exists(result_path):
            return f"Success: NEB Plot saved to {result_path}"
        else:
            return "Error: Failed to generate NEB Plot."
    except Exception as e:
        return f"Error running Visualizer: {e}"