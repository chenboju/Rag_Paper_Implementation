import os
import re
import numpy as np
from ase.lattice.cubic import BodyCenteredCubic, FaceCenteredCubic
from ase.io import lammpsdata

class StructureGenerator:
    """
    負責生成原子結構 (Data File) 與 勢函數文件 (Potential File)。
    不涉及執行 LAMMPS，只負責產生靜態文件。
    """

    @staticmethod
    def create_crystal(
        work_dir: str,
        filename: str,
        lattice_type: str,
        lattice_constant: float,
        elements: list[str], # 例如 ['Nb', 'Mo']
        concentrations: list[float], # 例如 [50, 50]
        directions: list[list[int]], # 例如 [[1,0,0], [0,1,0], [0,0,1]]
        size: list[int] # 例如 [10, 10, 10]
    ) -> str:
        """
        使用 ASE 建立晶體結構並儲存為 LAMMPS Data File。
        
        Args:
            filename: 輸出的檔名 (e.g., 'structure.data')
        Returns:
            生成的檔案完整路徑
        """
        if not os.path.exists(work_dir):
            os.makedirs(work_dir, exist_ok=True)
            
        full_path = os.path.join(work_dir, filename)
        
        # 1. 建立基礎晶格
        if lattice_type.lower() == 'bcc':
            atoms = BodyCenteredCubic(
                directions=directions,
                size=size,
                symbol=elements[0], # 先全設為第一個元素
                latticeconstant=lattice_constant,
                pbc=True
            )
        elif lattice_type.lower() == 'fcc':
            atoms = FaceCenteredCubic(
                directions=directions,
                size=size,
                symbol=elements[0],
                latticeconstant=lattice_constant,
                pbc=True
            )
        else:
            raise ValueError(f"Unsupported lattice type: {lattice_type}")

        # 2. 處理合金 (隨機置換)
        # 如果是二元合金且濃度不是 0/100
        if len(elements) > 1 and 0 < concentrations[0] < 100:
            num_atoms = len(atoms)
            num_solute = int(num_atoms * concentrations[1] / 100)
            
            # 隨機選取索引進行置換
            indices = np.random.permutation(num_atoms)
            # 前面部分保持元素1 (ASE預設)，後面部分改成元素2
            # 注意：ASE 的 symbols 設定比較 tricky，我們直接在寫檔時處理 Type ID 會更穩
            # 這裡我們先設定 ASE 對象的 symbols 以便檢查
            atoms.symbols[indices[:num_solute]] = elements[1]

        # 3. 寫入 LAMMPS Data File
        lammpsdata.write_lammps_data(full_path, atoms)
        
        # 4. [修正] 手動修復 ASE 寫出的 Data File 格式問題
        # ASE 預設寫出的 Masses 有時不符合我們後續 potential 的需求，
        # 或者我們需要確保 Type ID 正確 (Type 1 = Element 1, Type 2 = Element 2)
        StructureGenerator._patch_lammps_data_file(full_path, len(elements))
        
        return full_path

    @staticmethod
    def _patch_lammps_data_file(filepath: str, num_atom_types: int):
        """
        修正 ASE 輸出的 Data File：
        1. 確保 Masses 區塊存在 (雖然通常用 potential 覆蓋，但格式需要)
        2. 確保有多種原子 Type
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        headers_done = False
        
        for line in lines:
            # 插入 Masses 區塊 (如果 ASE 沒寫或者需要強制覆蓋)
            if "Atoms" in line and not headers_done:
                new_lines.append("\nMasses\n\n")
                for i in range(1, num_atom_types + 1):
                    new_lines.append(f"{i} 1.0\n") # 質量通常在 Potential 檔或 Input script 設定，這裡給佔位符
                new_lines.append("\n")
                headers_done = True
            
            new_lines.append(line)
            
        with open(filepath, 'w') as f:
            f.writelines(new_lines)

    @staticmethod
    def create_potential_file(
        work_dir: str,
        pair_style: str,
        pair_coeff: str,
        filename: str = "potential.inp"
    ) -> str:
        """建立 potential.inp 文件"""
        content = f"pair_style {pair_style}\npair_coeff {pair_coeff}\n"
        
        full_path = os.path.join(work_dir, filename)
        with open(full_path, "w") as f:
            f.write(content)
            
        return full_path