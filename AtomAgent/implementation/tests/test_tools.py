import unittest
import sys
import os
import json
import shutil
from unittest.mock import patch

# 確保可以 import src 模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools.physics_tools import calculate_lattice_constant

class TestPhysicsTools(unittest.TestCase):
    
    def setUp(self):
        """測試前的準備工作"""
        self.test_work_dir = "test_lattice_calc"
        # 確保資料夾路徑正確 (指向 data/test_lattice_calc)
        self.full_work_path = os.path.join(os.getenv("DATA_DIR", "./data"), self.test_work_dir)
        
        # 清理舊的測試資料
        if os.path.exists(self.full_work_path):
            shutil.rmtree(self.full_work_path)

    def test_lattice_constant_workflow(self):
        """
        測試從頭到尾的晶格常數計算流程。
        包含: ASE 建構 -> Template 生成 -> LAMMPS 執行 -> 解析器讀取
        """
        print(f"\n🧪 開始測試工具: calculate_lattice_constant...")

        # 定義輸入參數
        inputs = {
            "elements": ["Cu"], # 使用簡單單元素測試
            "concentrations": [100.0],
            "lattice_type": "fcc",
            "estimated_a": 3.61,
            "work_dir": self.test_work_dir
        }

        # --- 關鍵技巧: Mocking ---
        # 因為我們可能還沒下載真實的 EAM 勢函數檔案，
        # 這裡我們攔截 'create_potential_file'，改為寫入一個簡單的 LJ 勢函數。
        # 這樣可以確保 LAMMPS 能跑通，驗證軟體流程無誤。
        
        with patch('src.tools.physics_tools.StructureGenerator.create_potential_file') as mock_create_pot:
            
            # 定義 Mock 的行為: 寫入一個 Lennard-Jones 勢函數文件
            def side_effect_fake_potential(work_dir, pair_style, pair_coeff, filename):
                filepath = os.path.join(work_dir, filename)
                with open(filepath, 'w') as f:
                    f.write("# Mock Potential for Testing\n")
                    f.write("pair_style lj/cut 2.5\n")
                    f.write("pair_coeff * * 1.0 1.0\n") # epsilon=1.0, sigma=1.0
                    f.write("mass * 1.0\n") 
                print(f"   (Mocking) Created dummy LJ potential at {filepath}")
                return filepath

            mock_create_pot.side_effect = side_effect_fake_potential

            # 1. 執行工具
            result_json = calculate_lattice_constant.invoke(inputs)
            
            # 2. 驗證回傳格式
            self.assertIsInstance(result_json, str)
            print(f"   (Result) Tool output: {result_json}")
            
            # 3. 解析 JSON 並驗證內容
            result_dict = json.loads(result_json)
            
            self.assertEqual(result_dict['status'], 'success')
            self.assertIn('lattice_constant', result_dict)
            self.assertIn('potential_energy_per_atom', result_dict)
            
            # 驗證數值是否合理 (LJ 勢能的平衡位置大約在 1.12 sigma 左右，不過我們只求不報錯)
            self.assertIsInstance(result_dict['lattice_constant'], float)
            
            # 4. 驗證檔案是否真的產生
            self.assertTrue(os.path.exists(os.path.join(self.full_work_path, "relaxed_outputs.csv")))
            self.assertTrue(os.path.exists(os.path.join(self.full_work_path, "log.relax")))

        print("✅ 測試通過！LAMMPS 成功執行並回傳了結果。")

if __name__ == '__main__':
    unittest.main()