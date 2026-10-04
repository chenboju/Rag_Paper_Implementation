import os
import subprocess
import shutil
import time
from typing import Tuple, Optional

class LAMMPSRunner:
    """
    負責管理 LAMMPS 模擬的執行。
    將 '生成腳本' 與 '執行模擬' 的底層邏輯封裝在此。
    """
    
    def __init__(self, lammps_exec_path: str = "lmp", use_mpi: bool = True):
        """
        Args:
            lammps_exec_path: LAMMPS 執行檔路徑 (例如 'lmp' 或 'lmp_mpi')
            use_mpi: 是否使用 MPI 平行運算
        """
        self.exec_path = lammps_exec_path
        self.use_mpi = use_mpi

    def write_input_script(self, work_dir: str, script_content: str, filename: str = "simulation.in") -> str:
        """將 LAMMPS 輸入腳本寫入指定目錄"""
        if not os.path.exists(work_dir):
            os.makedirs(work_dir, exist_ok=True)
            
        file_path = os.path.join(work_dir, filename)
        with open(file_path, "w") as f:
            f.write(script_content)
        return file_path

    def run(self, work_dir: str, input_script_name: str, num_procs: int = 4, log_filename: str = "log.lammps") -> Tuple[bool, str]:
        """
        執行 LAMMPS 模擬。
        
        Args:
            work_dir: 工作目錄
            input_script_name: 輸入腳本檔名 (需位於 work_dir 內)
            num_procs: MPI 使用的核心數
            log_filename: Log 檔案名稱
            
        Returns:
            (success, output_message)
        """
        # 組合指令
        # 模式 1: 使用 MPI (推薦) -> mpiexec -n 4 lmp -in simulation.in -log log.lammps
        # 模式 2: 單核執行 -> lmp -in simulation.in -log log.lammps
        
        cmd = []
        if self.use_mpi and shutil.which("mpiexec"):
            cmd.extend(["mpiexec", "-n", str(num_procs)])
            
        cmd.append(self.exec_path)
        cmd.extend(["-in", input_script_name])
        cmd.extend(["-log", log_filename])
        
        print(f"🔨 Running LAMMPS in {work_dir}: {' '.join(cmd)}")
        
        try:
            # 使用 subprocess 執行，並捕獲輸出
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False # 不自動拋出異常，由我們判斷 returncode
            )
            
            if result.returncode == 0:
                return True, f"Simulation completed successfully. Log: {log_filename}"
            else:
                error_msg = f"LAMMPS Error (Code {result.returncode}):\n{result.stderr}"
                # 有時候錯誤訊息會在 stdout 中
                if "ERROR" in result.stdout:
                    error_msg += f"\nStdout Context:\n{result.stdout[-500:]}"
                return False, error_msg

        except FileNotFoundError:
            return False, f"Executable not found: {self.exec_path}. Please check .env or system path."
        except Exception as e:
            return False, f"System Error: {str(e)}"

# 簡單測試用
if __name__ == "__main__":
    runner = LAMMPSRunner()
    # 測試腳本：建立一個簡單的 Box
    test_script = """
    clear
    units metal
    dimension 3
    boundary p p p
    atom_style atomic
    lattice fcc 3.61
    region box block 0 10 0 10 0 10
    create_box 1 box
    create_atoms 1 box
    mass 1 63.55
    run 0
    """
    
    test_dir = "./tests/temp_lammps_test"
    runner.write_input_script(test_dir, test_script)
    success, msg = runner.run(test_dir, "simulation.in", num_procs=1)
    print(msg)