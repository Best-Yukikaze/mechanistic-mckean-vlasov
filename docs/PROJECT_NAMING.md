# 项目命名与目录体系

## 目的

本仓库只使用一套可读、可追溯的名称。名称必须把**研究轨道**、
**样品/文献谱系**、**科学里程碑**和**发布编号**分开；它们不能再共用
模糊的 `V1`、`V2`、`physical` 或 `mv` 前缀。

本规范不改变已经得到的科学结论，也不把历史 Hydrogel、MV-GEN-0、RL-0
或 D8 结果改写成当前的磁性颗粒证据。

## 五层身份

| 层级 | 写法 | 含义 | 示例 |
| --- | --- | --- | --- |
| 仓库 | `mechanistic_mckean_vlasov` | 唯一代码库，不带版本 | 本仓库 |
| 研究轨道 | 小写名词 | 研究对象，而非算法版本 | `magnetic_particle`、`hydrogel`、`mv_gen0_legacy` |
| 谱系 | 小写、可读的来源标识 | 样品、论文或“无特定样品”的泛化来源 | `generic`、`lyons2020_peg1000_8p9nm_agarose0p3` |
| 里程碑 | 小写动宾/结论目标 | 此文件回答的科学问题 | `physical_gate`、`constitutive_precheck` |
| 发布号 | `r<N>` | 该精确里程碑的不可变 Git/证据发布序号 | `r1` |

因此，正式 artifact 使用：

```text
<track>__<lineage>__<milestone>__r<N>.<extension>
```

例如：

```text
magnetic_particle__lyons2020_peg1000_8p9nm_agarose0p3__constitutive_precheck__r1.json
```

双下划线只分隔身份层；单下划线只连接同一层内的单词。

## `r`、`p`、`d`、protocol 与 schema 的区别

- `r<N>`：正式、可追溯的 release/evidence revision。Git 标签和已保存的
  canonical artifact 使用它。
- `p<N>`：计划文档的修订号，例如 `physical_next_p2`。计划更新不是新的
  科学结论，也不是新的 Git release。
- `d<N>`：父里程碑内的诊断子步骤，仅能出现在历史诊断说明中；不能代替
  release 号。
- `protocol_id` / `protocol_version`：冻结实验合同身份；不能由文件移动或
  Git 打标签改变。
- `schema`：JSON 的数据结构合同。已发布 JSON 内的 schema 保持原字节内容，
  即使文件路径迁移，也不得为“好看”而改写历史证据。

旧的 `V1` / `V2` 只在迁移表中作为**旧名称**出现；以后不再单独用它们描述
模型、数据、计划和 Git 发布。

## 正式目录分类

```text
mechanistic_mckean_vlasov/
├── src/mechanistic_mv/
│   ├── mechanics/                 # 稳定的 Physics/Controller 接口；不按 release 改名
│   │   └── magnetic_particle/      # 当前磁性颗粒 Physics 的 canonical 子命名空间
│   ├── continuum/                 # 连续体方程与数值方法
│   ├── control/                   # 控制器执行器接口
│   ├── envs/                      # Gym 包装；仅在物理门槛通过后使用
│   ├── evaluation/                # 评估逻辑
│   ├── rl/                        # 历史/未来 RL 实现，不能冒充物理证据
│   └── tasks/                     # 任务合同
├── data/external/
│   └── magnetic_particle/         # 已发布的磁性来源和谱系资料
├── datasets/hydrogel/             # Hydrogel 文献数据集，独立于磁性颗粒谱系
├── outputs/validation/
│   ├── magnetic_particle/         # 当前磁性物理证据
│   ├── hydrogel/                  # Hydrogel 工程/文献验证
│   └── mv_gen0/                   # 历史非磁性 MV-GEN 诊断，不是当前主线
├── outputs/
│   ├── mv_gen0*/                  # 未发布的历史训练输出；保留原路径，禁止作为当前结论
│   └── rl0/                        # 未发布的历史 RL 输出；保留原路径，禁止作为当前结论
├── scripts/                        # 可执行入口，名称写研究轨道和里程碑，不写 r 号
├── tests/                          # 与 scripts/模块同名的直接回归测试
├── references/                     # 只读文献与背景资料
└── tmp/                            # 临时文件，绝不作为 canonical evidence
```

目录层级保持浅：每个研究轨道只占一个稳定目录，不为单次试验反复创建新的
文件夹。文件名承担谱系、里程碑和发布号。

## 当前目录审计与归类

| 现有位置 | 归类 | 处理规则 |
| --- | --- | --- |
| `data/external/mv_physical_validation/` | 已发布磁性资料 | 迁移至 `data/external/magnetic_particle/` |
| `outputs/validation/mv_physical/` | 已发布磁性物理证据 | 迁移至 `outputs/validation/magnetic_particle/` |
| `src/mechanistic_mv/mechanics/magnetic_particle/` | 当前磁性 Physics canonical 模块 | `potential.py`、`dipolar_pair.py`、`continuum_admission.py` 是新代码入口；父目录中的旧名称是只转发的兼容层 |
| `datasets/hydrogel/` 与 `outputs/validation/hydrogel/` | Hydrogel 独立轨道 | 不并入 magnetic_particle，也不重算 |
| `outputs/validation/mv_gen0/`、`outputs/mv_gen0*`、`outputs/rl0/` | 未发布历史 RL/MV-GEN 输出 | 保留路径，登记为 `mv_gen0_legacy` / `rl0_legacy`，不纳入当前主线 |
| `scripts/run_mv_gen0*`、`src/.../mv_gen0*`、`tests/test_mv_gen0*` | 历史 MV-GEN 源码与测试 | 保留现名以避免破坏未发布实验合同；在目录分类中明确为 legacy |
| `tmp/` | 临时产物 | 不迁移、不提交、不引用为证据 |

## 已发布磁性轨道的迁移表

| 旧名称 | 新 canonical 名称 |
| --- | --- |
| `data/external/mv_physical_validation/source_provenance_v1.*` | `data/external/magnetic_particle/generic__source_provenance__r1.*` |
| `outputs/validation/mv_physical/mv_physical_validation_v1.*` | `outputs/validation/magnetic_particle/generic__physical_gate__r1.*` |
| `phase_a_transport_closure_v1.*` | `generic__transport_closure__r1.*` |
| `phase_b_magnetic_drift_validation_v1.*` | `generic__magnetic_drift__r1.*` |
| `phase_c_dipolar_w_validation_v1.*` | `generic__dipolar_interaction__r1.*` |
| `phase_d_joint_mv_validation_v1.*` | `generic__joint_mv__r1.*` |
| `lyons_lineage_provenance_v2.*` | `lyons2020_peg1000_8p9nm_agarose0p3__lineage_provenance__r1.*` |
| `lyons_constitutive_precheck_v2.*` | `lyons2020_peg1000_8p9nm_agarose0p3__constitutive_precheck__r1.*` |
| `lyons_front_digitized_manuscript_v2.csv` | `lyons2020_peg1000_8p9nm_agarose0p3__front_digitization__r1.csv` |
| `run_mv_physical_validation.py` | `run_magnetic_particle_generic_physical_gate.py` |
| `run_lyons_constitutive_precheck_v2.py` | `run_magnetic_particle_lyons_constitutive_precheck.py` |
| `test_mv_physical_validation.py` | `test_magnetic_particle_generic_physical_gate.py` |
| `test_lyons_constitutive_precheck_v2.py` | `test_magnetic_particle_lyons_constitutive_precheck.py` |
| `mechanics/magnetic_validation.py` | `mechanics/magnetic_particle/continuum_admission.py`；旧路径保留为 compatibility shim |
| `mechanics/magnetic_particle_potential.py` | `mechanics/magnetic_particle/potential.py`；旧路径保留为 compatibility shim |
| `mechanics/magnetic_dipole_interaction.py` | `mechanics/magnetic_particle/dipolar_pair.py`；旧路径保留为 compatibility shim |

Historic payload contents are preserved byte-for-byte whenever an artifact is
moved. The old path remains available in the historical Git release commit;
the current branch records the explicit mapping above. A legacy path recorded
inside an already released JSON is historical provenance, not an instruction
to recreate the old directory. Old Python/script names remain only as thin
compatibility shims; all new imports and commands use the canonical path.

## Git release naming

Git tag follows the same identity order, but uses hyphens:

```text
magnetic-particle-<lineage>-<milestone>-r<N>
```

The two existing releases receive canonical tags as follows. The old public
tags remain only as **deprecated compatibility aliases** so that an existing
GitHub link cannot silently break; all future reports and links must use the
canonical tag.

| 旧 Git 标签（废弃别名） | canonical Git 标签 | 指向的历史提交 |
| --- | --- | --- |
| `mv-physical-validation-v1` | `magnetic-particle-generic-physical-gate-r1` | `106aec0` |
| `mv-physical-validation-v2` | `magnetic-particle-lyons2020-constitutive-precheck-r1` | `9054020` |

本次目录迁移本身使用独立标签 `repository-naming-system-r1`。它只改变
命名、路径、入口和文档；不形成新的物理结论。

## 迁移不变量

1. 旧 V1 的 `CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID` 保持不变。
2. Lyons 狭义结论 `LYONS_CONSTITUTIVE_DATA_INSUFFICIENT` 保持不变。
3. 不重跑 PDE、Gym、RL 或 DQN；本次只运行路径、导入和直接命名回归测试。
4. 历史 Hydrogel、MV-GEN-0、RL-0、D8 内容不被移动到磁性目录，也不会以
   新名称伪装为当前磁性验证。
5. 任何新磁性 artifact 必须先进入上述 `magnetic_particle` 两个正式目录，
   再以 `track__lineage__milestone__rN` 命名。
