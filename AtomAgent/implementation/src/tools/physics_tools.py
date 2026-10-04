import os
import json
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional

from src.engine.structure_gen import StructureGenerator
from src.engine.lammps_runner import LAMMPSRunner
from src.engine.parsers import SimulationParser
from src.engine.templates import LAMMPSTemplates

# --- 輸入模型定義 ---
class LatticeConstantInput(BaseModel):
    elements: List[str] = Field(description="化學元素列表，例如 ['Cu']")
    concentrations: List[float] = Field(description="濃度百分比，例如 [100.0]")
    lattice_type: str = Field(default="fcc")
    estimated_a: float = Field(default=3.61) # Cu 的預設值
    work_dir: str = Field(default="calc_lattice")

# --- 工具定義 ---
@tool("calculate_lattice_constant", args_schema=LatticeConstantInput)
def calculate_lattice_constant(
    elements: List[str],
    concentrations: List[float],
    lattice_type: str = "fcc",
    estimated_a: float = 3.61,
    work_dir: str = "calc_lattice"
) -> str:
    """
    計算合金的平衡晶格常數 (Lattice Constant)。
    """
    try:
        # 1. 準備目錄
        full_work_dir = os.path.join(os.getenv("DATA_DIR", "./data"), work_dir)
        os.makedirs(full_work_dir, exist_ok=True)
        
        # 2. 生成結構
        struct_file = "initial_structure.data"
        StructureGenerator.create_crystal(
            work_dir=full_work_dir,
            filename=struct_file,
            lattice_type=lattice_type,
            lattice_constant=estimated_a,
            elements=elements,
            concentrations=concentrations,
            directions=[[1,0,0], [0,1,0], [0,0,1]],
            size=[10, 10, 10]
        )
        
        # 3. [關鍵修改] 生成勢函數文件 (Auto-fallback to LJ)
        # 為了教學演示，我們直接寫入一個簡單的 Lennard-Jones 勢函數，
        # 這樣你就不需要去網上下載 Cu.eam.alloy 也能跑通。
        pot_file = "potential.inp"
        
        # 我們手動建立一個簡單的勢函數內容
        # 這是 Lennard-Jones 參數，模擬類似金屬的鍵結
        lj_content = """
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
mass * 63.546 
""" 
        # 將 LJ 內容寫入檔案 (覆蓋原本指向 ../../NbMo.eam.alloy 的邏輯)
        with open(os.path.join(full_work_dir, pot_file), "w") as f:
            f.write(lj_content)

        # 4. 生成 LAMMPS 腳本
        script_content = LAMMPSTemplates.get_lattice_constant_script(
            data_file=struct_file, 
            potential_file=pot_file
        )
        
        runner = LAMMPSRunner()
        runner.write_input_script(full_work_dir, script_content, "relax.in")
        
        # 5. 執行模擬
        print(f"   (Tool) Executing LAMMPS in {full_work_dir}...")
        success, msg = runner.run(full_work_dir, "relax.in", log_filename="log.relax")
        
        if not success:
            # 如果 LAMMPS 失敗，回傳錯誤給 Agent
            return f"Simulation Failed: {msg}"
            
        # 6. 解析結果
        result = SimulationParser.parse_lattice_constant(full_work_dir, "relaxed_outputs.csv")
        final_a = result['lx'] / 10.0
        
        output = {
            "status": "success",
            "lattice_constant": round(final_a, 4),
            "potential_energy_per_atom": round(result['pe'] / result['num_atoms'], 4),
            "message": "Simulation completed using Lennard-Jones potential (Demo mode)."
        }
        
        return json.dumps(output)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})