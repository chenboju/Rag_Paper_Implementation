class LAMMPSTemplates:
    """
    存放標準化的 LAMMPS 輸入腳本模板。
    負責生成各種物理模擬所需的 .in 文件內容。
    """

    @staticmethod
    def get_lattice_constant_script(
        data_file: str, 
        potential_file: str = "potential.inp",
        thermo_time: int = 100,
        dump_time: int = 1000
    ) -> str:
        """
        生成晶格常數計算腳本 (Energy Minimization & Box Relaxation)。
        對應原代碼: lattice_constant_simulation
        """
        return f"""
clear
units metal
dimension 3
boundary p p p
atom_style atomic

# 讀取數據與勢函數
read_data {data_file}
include {potential_file}

compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol lx ly lz press pxx pyy pzz fnorm
thermo_modify format float %5.5g
# print thermo every N times
thermo {thermo_time}

shell mkdir dump_lattice_constant

reset_timestep 0
# dump first_dump all custom 1 dump.lattice.initial.0 id type x y z c_peratom
# run 0
# undump first_dump

variable dump_id equal 1
variable dump_out_name string "./dump_lattice_constant/dump.out"
# dump ${{dump_id}} all custom {dump_time} ${{dump_out_name}}.* id type x y z c_peratom

variable p1 equal "step"
variable p2 equal "count(all)"
variable p3 equal "lx"
variable p4 equal "ly"
variable p5 equal "lz"
variable p6 equal "pe"
variable p7 equal "vol"

# --- Relaxation Loop ---
fix 1 all box/relax aniso 0 
min_style cg
minimize 0 1e-5 2000 20000
unfix 1

min_style cg
minimize 0 1e-5 20000 20000

fix 1 all box/relax aniso 0 
min_style cg
minimize 0 1e-5 2000 20000
unfix 1

min_style cg
minimize 0 1e-5 20000 20000

fix 1 all box/relax aniso 0 
min_style cg
minimize 0 1e-5 2000 20000
unfix 1

min_style cg
minimize 0 1e-5 20000 20000
# -----------------------

# Output final results to CSV
fix extra all print 100 "${{p1}} ${{p2}} ${{p3}} ${{p4}} ${{p5}} ${{p6}} ${{p7}}" file "relaxed_outputs.csv" title "#step numb_atoms a_x a_y a_z pe vol" screen no

reset_timestep 0
# dump last_dump all custom 1 dump.lattice.final.0 id type x y z c_peratom
run 0

unfix extra
# undump last_dump
# quit (由 Runner 控制)
"""

    @staticmethod
    def get_elastic_constant_script(
        data_file: str,
        potential_file: str = "potential.inp",
        thermo_time: int = 100,
        dump_time: int = 1000
    ) -> str:
        """
        生成彈性模數計算腳本。
        對應原代碼: elastic_constant_simulation
        注意：此腳本依賴 'displace.mod' 存在於同一目錄下。
        """
        return f"""
clear
variable data_file string "{data_file}"
variable potential_file string "{potential_file}"
variable thermo_time equal {thermo_time}
variable dump_time equal {dump_time}

# 注意: displace.mod 必須存在於工作目錄
variable displace_file string "displace.mod" 

# Define the finite deformation size. Try several values of this
# variable to verify that results do not depend on it.
variable up equal 1.0e-6
 
# Define the amount of random jiggle for atoms
# This prevents atoms from staying on saddle points
variable atomjiggle equal 1.0e-5

# metal units, elastic constants in GPa
units		metal
variable cfac equal 1.0e-4
variable cunits string GPa

# Define minimization parameters
variable etol equal 0.0 
variable ftol equal 1.0e-5
variable maxiter equal 5000
variable maxeval equal 5000
variable dmax equal 1.0e-2

# Setup neighbor style
neighbor 1.0 nsq
neigh_modify once no every 1 delay 0 check yes

boundary	p p p

read_data ${{data_file}}
include ${{potential_file}}

change_box all triclinic

# Setup output
thermo		50
thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol
thermo_modify norm no

# Setup minimization style
min_style	     cg
min_modify	     dmax ${{dmax}} line quadratic

# Compute initial state
fix 3 all box/relax  aniso 0.0
minimize ${{etol}} ${{ftol}} ${{maxiter}} ${{maxeval}}

variable tmp equal pxx
variable pxx0 equal ${{tmp}}
variable tmp equal pyy
variable pyy0 equal ${{tmp}}
variable tmp equal pzz
variable pzz0 equal ${{tmp}}
variable tmp equal pyz
variable pyz0 equal ${{tmp}}
variable tmp equal pxz
variable pxz0 equal ${{tmp}}
variable tmp equal pxy
variable pxy0 equal ${{tmp}}

variable tmp equal lx
variable lx0 equal ${{tmp}}
variable tmp equal ly
variable ly0 equal ${{tmp}}
variable tmp equal lz
variable lz0 equal ${{tmp}}

# These formulas define the derivatives w.r.t. strain components
# Constants uses $, variables use v_ 
variable d1 equal -(v_pxx1-${{pxx0}})/(v_delta/v_len0)*${{cfac}}
variable d2 equal -(v_pyy1-${{pyy0}})/(v_delta/v_len0)*${{cfac}}
variable d3 equal -(v_pzz1-${{pzz0}})/(v_delta/v_len0)*${{cfac}}
variable d4 equal -(v_pyz1-${{pyz0}})/(v_delta/v_len0)*${{cfac}}
variable d5 equal -(v_pxz1-${{pxz0}})/(v_delta/v_len0)*${{cfac}}
variable d6 equal -(v_pxy1-${{pxy0}})/(v_delta/v_len0)*${{cfac}}

displace_atoms all random ${{atomjiggle}} ${{atomjiggle}} ${{atomjiggle}} 87287 units box

# Write restart
unfix 3
# write_restart restart.equil

# uxx Perturbation
variable dir equal 1
include ${{displace_file}}

# uyy Perturbation
variable dir equal 2
include ${{displace_file}}

# uzz Perturbation
variable dir equal 3
include ${{displace_file}}

# uyz Perturbation
variable dir equal 4
include ${{displace_file}}

# uxz Perturbation
variable dir equal 5
include ${{displace_file}}

# uxy Perturbation
variable dir equal 6
include ${{displace_file}}

# Output final values

variable C11all equal ${{C11}}
variable C22all equal ${{C22}}
variable C33all equal ${{C33}}

variable C12all equal 0.5*(${{C12}}+${{C21}})
variable C13all equal 0.5*(${{C13}}+${{C31}})
variable C23all equal 0.5*(${{C23}}+${{C32}})

variable C44all equal ${{C44}}
variable C55all equal ${{C55}}
variable C66all equal ${{C66}}

variable C14all equal 0.5*(${{C14}}+${{C41}})
variable C15all equal 0.5*(${{C15}}+${{C51}})
variable C16all equal 0.5*(${{C16}}+${{C61}})

variable C24all equal 0.5*(${{C24}}+${{C42}})
variable C25all equal 0.5*(${{C25}}+${{C52}})
variable C26all equal 0.5*(${{C26}}+${{C62}})

variable C34all equal 0.5*(${{C34}}+${{C43}})
variable C35all equal 0.5*(${{C35}}+${{C53}})
variable C36all equal 0.5*(${{C36}}+${{C63}})

variable C45all equal 0.5*(${{C45}}+${{C54}})
variable C46all equal 0.5*(${{C46}}+${{C64}})
variable C56all equal 0.5*(${{C56}}+${{C65}})

# Average moduli for cubic crystals

variable C11cubic equal (${{C11all}}+${{C22all}}+${{C33all}})/3.0
variable C12cubic equal (${{C12all}}+${{C13all}}+${{C23all}})/3.0
variable C44cubic equal (${{C44all}}+${{C55all}}+${{C66all}})/3.0

variable bulkmodulus equal (${{C11cubic}}+2*${{C12cubic}})/3.0
variable shearmodulus1 equal ${{C44cubic}}
variable shearmodulus2 equal (${{C11cubic}}-${{C12cubic}})/2.0
variable poissonratio equal 1.0/(1.0+${{C11cubic}}/${{C12cubic}})
      
# Output to Log
print "========================================="
print "Components of the Elastic Constant Tensor"
print "========================================="
print "Elastic Constant C11all = ${{C11all}} ${{cunits}}"
print "Elastic Constant C12all = ${{C12all}} ${{cunits}}"
print "Elastic Constant C44all = ${{C44all}} ${{cunits}}"
print "Bulk Modulus = ${{bulkmodulus}} ${{cunits}}"
print "Shear Modulus 1 = ${{shearmodulus1}} ${{cunits}}"
print "Poisson Ratio = ${{poissonratio}}"

# Output to CSV for Parser
fix extra all print 1 "${{C11all}} ${{C12all}} ${{C13all}} ${{C14all}} ${{C15all}} ${{C16all}} ${{C22all}} ${{C23all}} ${{C24all}} ${{C25all}} ${{C26all}} ${{C33all}} ${{C34all}} ${{C35all}} ${{C36all}} ${{C44all}} ${{C45all}} ${{C46all}} ${{C55all}} ${{C56all}} ${{C66all}} ${{bulkmodulus}} ${{shearmodulus1}} ${{poissonratio}}" file "elastic_constants.csv" title "#C11 C12 C13 C14 C15 C16 C22 C23 C24 C25 C26 C33 C34 C35 C36 C44 C45 C46 C55 C56 C66 bulkmodulus shearmodulus poissonratio" screen no

run 0
"""

    @staticmethod
    def get_displace_mod_content() -> str:
        """
        回傳 displace.mod 的內容。
        這個檔案被 elastic constant 腳本用來進行微擾計算。
        """
        return """
# NOTE: This file is called by the elastic constant script
# It performs a single deformation in the direction defined by variable 'dir'

# Reset box tilt
if "${dir} == 1" then "variable len0 equal lx"
if "${dir} == 2" then "variable len0 equal ly"
if "${dir} == 3" then "variable len0 equal lz"
if "${dir} == 4" then "variable len0 equal lz"
if "${dir} == 5" then "variable len0 equal lz"
if "${dir} == 6" then "variable len0 equal ly"

# Deformation magnitude
variable delta equal ${up}*${len0}
variable deltaxy equal ${up}*xy
variable deltaxz equal ${up}*xz
variable deltayz equal ${up}*yz

# Deform the box
if "${dir} == 1" then "change_box all x delta 0 ${delta} remap units box"
if "${dir} == 2" then "change_box all y delta 0 ${delta} remap units box"
if "${dir} == 3" then "change_box all z delta 0 ${delta} remap units box"
if "${dir} == 4" then "change_box all yz delta ${delta} remap units box"
if "${dir} == 5" then "change_box all xz delta ${delta} remap units box"
if "${dir} == 6" then "change_box all xy delta ${delta} remap units box"

# Relax atoms
minimize ${etol} ${ftol} ${maxiter} ${maxeval}

# Calculate stress tensor variables
variable pxx1 equal pxx
variable pyy1 equal pyy
variable pzz1 equal pzz
variable pxy1 equal pxy
variable pxz1 equal pxz
variable pyz1 equal pyz

# Compute elastic constants from stress-strain
if "${dir} == 1" then "variable C11 equal ${d1}"
if "${dir} == 1" then "variable C21 equal ${d2}"
if "${dir} == 1" then "variable C31 equal ${d3}"
if "${dir} == 1" then "variable C41 equal ${d4}"
if "${dir} == 1" then "variable C51 equal ${d5}"
if "${dir} == 1" then "variable C61 equal ${d6}"

if "${dir} == 2" then "variable C12 equal ${d1}"
if "${dir} == 2" then "variable C22 equal ${d2}"
if "${dir} == 2" then "variable C32 equal ${d3}"
if "${dir} == 2" then "variable C42 equal ${d4}"
if "${dir} == 2" then "variable C52 equal ${d5}"
if "${dir} == 2" then "variable C62 equal ${d6}"

if "${dir} == 3" then "variable C13 equal ${d1}"
if "${dir} == 3" then "variable C23 equal ${d2}"
if "${dir} == 3" then "variable C33 equal ${d3}"
if "${dir} == 3" then "variable C43 equal ${d4}"
if "${dir} == 3" then "variable C53 equal ${d5}"
if "${dir} == 3" then "variable C63 equal ${d6}"

if "${dir} == 4" then "variable C14 equal ${d1}"
if "${dir} == 4" then "variable C24 equal ${d2}"
if "${dir} == 4" then "variable C34 equal ${d3}"
if "${dir} == 4" then "variable C44 equal ${d4}"
if "${dir} == 4" then "variable C54 equal ${d5}"
if "${dir} == 4" then "variable C64 equal ${d6}"

if "${dir} == 5" then "variable C15 equal ${d1}"
if "${dir} == 5" then "variable C25 equal ${d2}"
if "${dir} == 5" then "variable C35 equal ${d3}"
if "${dir} == 5" then "variable C45 equal ${d4}"
if "${dir} == 5" then "variable C55 equal ${d5}"
if "${dir} == 5" then "variable C65 equal ${d6}"

if "${dir} == 6" then "variable C16 equal ${d1}"
if "${dir} == 6" then "variable C26 equal ${d2}"
if "${dir} == 6" then "variable C36 equal ${d3}"
if "${dir} == 6" then "variable C46 equal ${d4}"
if "${dir} == 6" then "variable C56 equal ${d5}"
if "${dir} == 6" then "variable C66 equal ${d6}"

# Restore original box
if "${dir} == 1" then "change_box all x delta 0 -${delta} remap units box"
if "${dir} == 2" then "change_box all y delta 0 -${delta} remap units box"
if "${dir} == 3" then "change_box all z delta 0 -${delta} remap units box"
if "${dir} == 4" then "change_box all yz delta -${delta} remap units box"
if "${dir} == 5" then "change_box all xz delta -${delta} remap units box"
if "${dir} == 6" then "change_box all xy delta -${delta} remap units box"

# Relax atoms back to initial state
minimize ${etol} ${ftol} ${maxiter} ${maxeval}
"""

    @staticmethod
    def get_surface_energy_periodic_script(
        data_file: str,
        potential_file: str = "potential.inp",
        thermo_time: int = 100
    ) -> str:
        """
        生成表面能計算 (週期性部分) 腳本。
        對應原代碼: surface_energy_periodic
        """
        return f"""
clear
units metal
dimension 3
boundary p p p
atom_style atomic

read_data {data_file}
include {potential_file}

compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol press pxx pyy pzz fnorm
thermo_modify format float %5.5g
thermo {thermo_time}

shell mkdir surface_energy

variable p1 equal "step"
variable p2 equal "lx"
variable p3 equal "ly"
variable p4 equal "lz"
variable p5 equal "pe"
variable p6 equal "vol"

fix extra all print 100 "${{p1}} ${{p2}} ${{p3}} ${{p4}} ${{p5}} ${{p6}}" file "surface_energy_periodic.csv" title "#step a_x a_y a_z pe vol" screen no

min_style cg
minimize 0 1e-5 20000 20000

fix 1 all box/relax aniso 0 
min_style cg
minimize 0 1e-5 2000 20000
unfix 1

reset_timestep 0
run 0

unfix extra

write_data data.surface.energy.periodic.lmp
"""

    @staticmethod
    def get_surface_energy_free_script(
        data_file: str,
        surface_dir: str, # "x", "y", or "z"
        potential_file: str = "potential.inp",
        thermo_time: int = 100
    ) -> str:
        """
        生成表面能計算 (自由表面部分) 腳本。
        對應原代碼: surface_energy_free
        """
        return f"""
clear
units metal
dimension 3
boundary p p p
atom_style atomic

variable surface_dir string {surface_dir}
variable xxx string "x"
variable yyy string "y"
variable zzz string "z"

read_data {data_file}
include {potential_file}

compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol press pxx pyy pzz fnorm
thermo_modify format float %5.5g
thermo {thermo_time}

shell mkdir surface_energy

variable p1 equal "step"
variable p2 equal "lx"
variable p3 equal "ly"
variable p4 equal "lz"
variable p5 equal "pe"
variable p6 equal "vol"

fix extra all print 100 "${{p1}} ${{p2}} ${{p3}} ${{p4}} ${{p5}} ${{p6}}" file "surface_energy_free.csv" title "#step a_x a_y a_z pe vol" screen no

# Change boundary conditions based on surface direction
if "${{surface_dir}}==${{xxx}} " then "change_box all boundary s p p" 
if "${{surface_dir}}==${{yyy}} " then "change_box all boundary p s p" 
if "${{surface_dir}}==${{zzz}} " then "change_box all boundary p p s"

min_style cg
minimize 0 1e-5 20000 20000

reset_timestep 0
run 0

unfix extra
"""

    @staticmethod
    def get_stacking_fault_script(
        data_file: str,
        tau_x: float, # shift amount
        xlat: float,  # lattice parameter in x
        Lz: float,
        potential_file: str = "potential.inp",
        thermo_time: int = 100,
        dump_time: int = 1000
    ) -> str:
        """
        生成堆垛層錯 (Stacking Fault) 計算腳本。
        對應原代碼: stacking_fault_simulation
        """
        return f"""
clear
units metal
dimension 3
boundary p p p

read_data {data_file}
include {potential_file}

change_box all boundary p p s

variable zmid equal {Lz}/2-0.1
region top block INF INF INF INF ${{zmid}} INF units box
group top region top

variable tau_xx equal {tau_x}*{xlat}

displace_atoms top move ${{tau_xx}} 0 0 units box

compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol press pxx pyy pzz fnorm
thermo_modify format float %5.5g
thermo {thermo_time}

shell mkdir stacking_fault_energy

variable dump_id equal 1
variable dump_out_name string "./stacking_fault_energy/dump.out"
# dump ${{dump_id}} all custom {dump_time} ${{dump_out_name}}.* id type x y z c_peratom

variable p1 equal "step"
variable p2 equal "lx"
variable p3 equal "ly"
variable p4 equal "lz"
variable p5 equal "pe"
variable p6 equal "vol"

fix extra all print 100 "${{p1}} ${{p2}} ${{p3}} ${{p4}} ${{p5}} ${{p6}}" file "stacking_fault.csv" title "#step a_x a_y a_z pe vol" screen no

fix xz_fixed all setforce 0.0 NULL 0.0

run 0

min_style fire
minimize 0 1e-5 20000 20000

# undump ${{dump_id}}

variable dump_name string "dump.stacking_fault.x_{tau_x}"
reset_timestep 0
dump last_dump all custom 1 ${{dump_name}} id type x y z c_peratom
run 0
undump last_dump

unfix extra
"""

    @staticmethod
    def get_create_screw_initial_script(
        data_file: str,
        output_name: str,
        burgers_vector: float,
        disl_y: float,
        potential_file: str = "potential.inp"
    ) -> str:
        """
        建立初始態螺旋位錯 (Screw Dislocation)。
        對應原代碼: create_screw_dislocation_initial
        """
        return f"""
clear
units metal
dimension 3
boundary p s p

read_data {data_file}
include {potential_file}

change_box all triclinic

variable ymid equal {disl_y}
region bot_region block INF INF INF ${{ymid}} INF INF units box
region top_region block INF INF ${{ymid}} INF INF INF units box

group bot region bot_region
group top region top_region

variable Xmin_t equal bound(top,xmin)+4.01
variable Xmax_t equal bound(top,xmax)-4.01

variable bz equal {burgers_vector}

displace_atoms top ramp z -${{bz}} 0 x ${{Xmin_t}} ${{Xmax_t}} units box

reset_timestep 0
dump last_dump all custom 1 dump.unrelaxed.{output_name} id type x y z
run 0
undump last_dump

write_data data.unrelaxed.{output_name}
"""

    @staticmethod
    def get_relax_screw_script(
        data_file: str,
        output_name: str,
        potential_file: str = "potential.inp",
        thermo_time: int = 100,
        dump_time: int = 1000
    ) -> str:
        """
        鬆弛螺旋位錯結構。
        對應原代碼: relax_screw_dislocation
        """
        return f"""
clear
units metal
dimension 3
boundary p s p

read_data {data_file}
include {potential_file}

compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol press pxx pyy pzz fnorm
thermo_modify format float %5.5g
thermo {thermo_time}

shell mkdir relax_screw_dislocation

variable dump_id equal 1
variable dump_out_name string "./relax_screw_dislocation/dump.out"
# dump ${{dump_id}} all custom {dump_time} ${{dump_out_name}}.* id type x y z c_peratom

variable p1 equal "step"
variable p2 equal "lx"
variable p3 equal "ly"
variable p4 equal "lz"
variable p5 equal "pe"
variable p6 equal "vol"

fix extra all print 100 "${{p1}} ${{p2}} ${{p3}} ${{p4}} ${{p5}} ${{p6}}" file "dislocation.csv" title "#step a_x a_y a_z pe vol" screen no

variable iii loop 5
variable tol equal 0.1

reset_timestep 0
label min

fix min1 all box/relax x 0.0 z 0.0 xz 0.0 vmax 0.001
min_style cg
minimize 0 1e-4 2000 2000
unfix min1
min_style cg
minimize 0 1e-4 20000 20000

variable press_x equal abs(pxx)
variable press_z equal abs(pzz)
variable press_xz equal abs(pxz)
variable Fn equal "fnorm"

if "${{Fn}} < 1e-4 && ${{press_x}} < ${{tol}} && ${{press_z}} < ${{tol}} && ${{press_xz}} < ${{tol}}" then "jump SELF exit_label"

next iii
jump SELF min

label exit_label

# undump ${{dump_id}} 

reset_timestep 0
dump last_dump all custom 1 dump.relaxed.{output_name} id type x y z c_peratom
run 0

write_data data.relaxed.{output_name}

undump last_dump
unfix extra
quit
"""

    @staticmethod
    def get_neb_script(
        initial_data_file: str,
        final_dump_file: str,
        energy_log_file: str,
        num_replicas: int,
        potential_file: str = "potential.inp",
        thermo_time: int = 100,
        dump_time: int = 1000
    ) -> str:
        """
        生成 NEB 計算腳本。
        對應原代碼: NEB_screw_simulation
        """
        return f"""
variable energy_file string "{energy_log_file}"
print "#potential energy (eV)" file ${{energy_file}}
clear
units metal
dimension 3
boundary p s p
atom_style atomic 
atom_modify map array sort 0 0.0

variable u uloop {num_replicas} pad

read_data {initial_data_file}
include {potential_file}

compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol press pxx pyy pzz fnorm
thermo_modify format float %5.5g
thermo {thermo_time}

shell mkdir NEB_simulation

variable dump_id equal 1
variable dump_out_name string "./NEB_simulation/dump.out"
# dump ${{dump_id}} all custom {dump_time} ${{dump_out_name}}.* id type x y z c_peratom

variable p5 equal "pe"

fix extra all print 100 "$u ${{p5}}" append ${{energy_file}} title "#replica pe" screen no

timestep 0.001
min_style fire
thermo 10
thermo_style custom step temp pe ke press pxx pyy pzz pxy pxz pyz fnorm

fix 3 all neb 0.001
neb 0.0 0.001 1000 1000 100 final {final_dump_file}

shell mkdir images

reset_timestep 0
dump 3 all custom 1 images/dump.initial.$u id type x y z c_peratom
dump_modify 3 format line "%d %d %.8e %.8e %.8e %.8e"
run 0
undump 3

print "finish"

unfix extra
"""