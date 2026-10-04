import os
import pandas as pd
from typing import Dict, Any

class SimulationParser:
    """
    負責解析 LAMMPS 輸出的 CSV 或 Log 檔案，提取關鍵物理數據。
    """

    @staticmethod
    def parse_lattice_constant(work_dir: str, filename: str = "relaxed_outputs.csv") -> Dict[str, float]:
        """
        解析晶格常數模擬結果。
        預期 CSV 格式: step numb_atoms a_x a_y a_z pe vol
        """
        filepath = os.path.join(work_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Result file not found: {filepath}")

        # 讀取最後一行 (收斂後的結果)
        try:
            # comment='#' 會忽略掉由 LAMMPS title 產生的標頭行，所以我們必須手動指定 names
            df = pd.read_csv(filepath, sep=r'\s+', comment='#', names=['step', 'num_atoms', 'lx', 'ly', 'lz', 'pe', 'vol'])
            
            if df.empty:
                raise ValueError("Parsed dataframe is empty. LAMMPS might have failed to output data.")
                
            last_row = df.iloc[-1]
            
            return {
                "lx": float(last_row['lx']),
                "ly": float(last_row['ly']),
                "lz": float(last_row['lz']),
                "pe": float(last_row['pe']), # Potential Energy
                "vol": float(last_row['vol']),
                "num_atoms": float(last_row['num_atoms']) # [修正] 補上這行
            }
        except Exception as e:
            raise ValueError(f"Error parsing lattice constant file: {e}")

    @staticmethod
    def parse_elastic_constants(work_dir: str, filename: str = "elastic_constants.csv") -> Dict[str, float]:
        """
        解析彈性模數 (C11, C12, C44...)。
        """
        filepath = os.path.join(work_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Elastic constants file not found: {filepath}")

        try:
            # 讀取原本腳本產出的單行 CSV
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # 確保有資料
            valid_lines = [line for line in lines if not line.startswith('#') and line.strip()]
            if not valid_lines:
                raise ValueError("Elastic constants file is empty or contains only comments.")

            # 抓取最後一行數據
            data = valid_lines[-1].strip().split()
            
            # 對應 displace.mod 的輸出順序 (參考 templates.py 中的 fix print 指令)
            # C11 C12 C13 ... C66 Bulk Shear Poisson
            return {
                "C11": float(data[0]),
                "C12": float(data[1]),
                "C44": float(data[15]), 
                "Bulk_Modulus": float(data[21]),
                "Shear_Modulus": float(data[22])
            }
        except Exception as e:
             raise ValueError(f"Error parsing elastic constants: {e}")

    @staticmethod
    def parse_neb_barrier(work_dir: str, filename: str) -> Dict[str, float]:
        """
        解析 NEB 模擬的能障 (Peierls Barrier)。
        """
        filepath = os.path.join(work_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"NEB result file not found: {filepath}")

        try:
            # 格式: replica_id energy
            df = pd.read_csv(filepath, sep=r'\s+', comment='#', names=['replica', 'energy'])
            
            if df.empty:
                raise ValueError("NEB energy file is empty.")

            # 能量歸零 (減去初始態能量)
            initial_energy = df['energy'].iloc[0]
            df['rel_energy'] = df['energy'] - initial_energy
            
            barrier = df['rel_energy'].max()
            energy_change = df['rel_energy'].iloc[-1] # 終態與初態的能量差
            
            return {
                "barrier": barrier,
                "energy_change": energy_change,
                "path_energies": df['rel_energy'].tolist() # 提供給繪圖用
            }
        except Exception as e:
            raise ValueError(f"Error parsing NEB data: {e}")