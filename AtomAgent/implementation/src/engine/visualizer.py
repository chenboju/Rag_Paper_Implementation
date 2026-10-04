import os
import numpy as np
import matplotlib.pyplot as plt
import atomman as am

class Visualizer:
    """
    負責將模擬數據轉換為圖像 (PNG)。
    """

    @staticmethod
    def plot_dd_map(
        work_dir: str, 
        pristine_file: str, 
        dislocation_dump_file: str, 
        burgers_vector: float,
        output_filename: str = "dd_map.png"
    ) -> str:
        """
        繪製 Differential Displacement Map (DD Map)。
        
        Args:
            pristine_file: 完美晶格的 Data file 路徑
            dislocation_dump_file: 含有位錯的 Dump file 路徑
            burgers_vector: 柏格斯向量大小 (Angstrom)
        """
        try:
            pristine_path = os.path.join(work_dir, pristine_file)
            dump_path = os.path.join(work_dir, dislocation_dump_file)
            
            # 使用 Atomman 載入系統
            base_system = am.load('atom_data', pristine_path)
            disl_system = am.load('atom_dump', dump_path)
            
            # 設定參數
            vburger = np.array([0.0, 0.0, burgers_vector])
            
            # 這裡簡化了原本代碼中複雜的 xlim/ylim 計算，
            # 實務上可能需要根據 box size 動態調整，這裡先做一個通用版
            # 若 Atomman 報錯，通常是因為邊界條件設定，需確保輸入檔案正確
            
            base_system.pbc = (False, False, True)
            
            # 產生圖表物件
            fig = plt.figure(figsize=(8, 6))
            
            # 呼叫 Atomman 的 DD map 繪圖功能
            # 注意: 這裡的參數需要根據實際模擬的 Box 大小微調
            result = am.defect.differential_displacement(
                base_system, 
                disl_system, 
                vburger, 
                cutoff=burgers_vector * 1.5, # 自動推算 cutoff
                plot_scale=3
            )
            
            # 儲存
            output_path = os.path.join(work_dir, output_filename)
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            return output_path

        except Exception as e:
            print(f"⚠️ Plotting failed: {e}")
            return ""

    @staticmethod
    def plot_neb_mep(
        work_dir: str,
        path_energies: list[float],
        output_filename: str = "neb_mep.png"
    ) -> str:
        """
        繪製 NEB 最低能量路徑 (Minimum Energy Path)。
        """
        try:
            plt.figure(figsize=(8, 5))
            plt.plot(path_energies, 'o-', linewidth=2, markersize=8)
            plt.title("NEB Minimum Energy Path")
            plt.xlabel("Replica ID")
            plt.ylabel("Energy (eV)")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            output_path = os.path.join(work_dir, output_filename)
            plt.savefig(output_path)
            plt.close()
            
            return output_path
        except Exception as e:
            print(f"⚠️ NEB Plotting failed: {e}")
            return ""