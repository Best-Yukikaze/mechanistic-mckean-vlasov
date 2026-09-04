# Lyons / Brougham 2020：谱系与公开数据审计 v2

窄结论：**LYONS_CONSTITUTIVE_DATA_INSUFFICIENT**。
项目级既有结论仍为 **CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID**；本审计没有重新运行 PDE，不能解除二维物理约化阻断。

本文件对应同目录 `lyons_lineage_provenance_v2.json`。只登记 PEG1000、8.9 nm 磁核、约 24 nm 水动力直径、0.3% w/v low-EEO agarose / DI water 这一研究谱系。论文与学位论文重叠并不能证明每张图来自同一合成批次；2026 年 PEG2000 的 8.1/10.5 nm 数据明确排除，不能转移其参数或 CSV。

## 已核查来源与访问限制

| 来源 | 定位 | 本次证据级别 |
|---|---|---|
| [2020 Nanoscale 论文](https://pubs.rsc.org/en/content/articlelanding/2020/nr/d0nr01602k) | DOI 10.1039/D0NR01602K，12，10550–10558 | 出版元数据；出版终稿全文未取得 |
| [作者公开稿](https://doras.dcu.ie/25059/1/DORAS%20submission.pdf) | PDF pp.3–5，Eq.1、Fig.1、Table 1 | 原作者存档；pp.3–4 图文已核查，不声称与终稿完全相同 |
| [官方 ESI](https://www.rsc.org/suppdata/d0/nr/d0nr01602k/d0nr01602k1.pdf) | p.2 S1/S2；pp.7–10 S7–S14 | 11 页索引文本可读；直接下载返回 404 |
| [Lyons 学位论文](https://doras.dcu.ie/25019/2/Stephen%20Lyons%20Thesis%20Sept%2015th.pdf) | PDF p.17，pp.71、78–79、95–98 | 原始学位论文文本；PDF 页与印刷页不同 |
| [2021 更正](https://pubs.rsc.org/en/content/articlehtml/2021/nr/d0nr90262d) | DOI 10.1039/D0NR90262D；Fig.3/4 和 p.10555 | 指标为 `v_exp * d_hyd`、单位 mm²/h；不是 `v_exp / d_hyd`，数值未改 |

## 必须保留的区别

- χ：学位论文 **0.281**；作者稿/ESI **0.289**。分开计算条件性算术，不平均、不择优，也不视为两个独立磁场实验。
- **0.23 T** 是 ESI 中磁化率取值的参考场；**0.55 T** 是磁体标称值；**45 T/m** 是引用的梯度。它们不能拼成沿同一轨迹实测并带误差的 `B(x), grad B(x)`。
- `0.37 ± 0.02 mm/h` 是光学前沿速度，误差是 16 个凝胶速度的 SD，不是 SEM，更不是跨磁场条件的迁移率 CV。
- 298 K 对应表征；运输实验温度未确认，不能补填。报告仅条件性计算 `D_Einstein/T` 与同温 `D_Einstein/D0`，不输出已准入 D。
- 8.9 ± 0.8 nm 是粒径分布信息；ESI 与论文 TEM 计数有 205/208 差别，不能直接换成磁力的完整标准不确定度。
- 6 mm 是凝胶高度，不是已验证的二维出平面厚度、垂向分布或近接触单元闭合。

## 数据取得结果

没有找到同批次、带逐凝胶 ID/误差/局部场图的原始前沿 CSV。此结论仅限已核查公开路径，不表示作者或未索引库中一定没有数据。

已经取得的作者稿 Fig.1 共 16 个蓝色标记坐标保留在 JSON 中；不继续数字化。复跑脚本只从这些存档坐标重建前沿点，不读取图像、不访问网络。坐标轴标定、像素中心、1.5 px 分析者定位界限都有记录；这个界限不是实验 SD，尚未计入坐标轴系统误差。输出 CSV 必须标为 **DIGITIZED_AUTHOR_MANUSCRIPT_NOT_RAW_DATA**，不能重建或冒充 16 个独立凝胶的原始重复数据。

## 条件算术与阻断

仅调用 Physics 的公开磁力接口，以论文各自 χ 加标称场的配方检查 `F=χ V B gradB/μ0`。`M_front=v_front/F` 只是前沿表观值；缺少前沿到平均漂移的观测闭合，不能登记为 M_eff。`D/T=k_B M_front` 同样依赖未验证的 Einstein 假设。完整不确定度缺失，只传播已报速度 SD。无独立多场实验则 `CV(M)` 为 null；不拟合 tortuosity φ 来消除差异。

下一步需要：批次对应、磁化数据及单位/不确定度、沿路径场图、前沿阈值与原始浓度/质心数据、独立力条件、运输温度和被动扩散测量。

## 复跑与请求草案

在仓库根目录、`PYTHONPATH=src` 下执行：

```powershell
& 'D:\conda environment\envs\dl\python.exe' scripts/run_lyons_constitutive_precheck_v2.py
& 'D:\conda environment\envs\dl\python.exe' -m unittest tests.test_lyons_constitutive_precheck_v2 -v
```

脚本退出 **2** 明确表示科学数据不足，不是计算已通过物理准入。JSON/Markdown 报告均保存于 `outputs/validation/mv_physical/`；Markdown 和 JSON 均含精准作者数据请求草案，标明 **NOT SENT**。脚本不发送消息、不联网，不运行 PDE/Gym/RL/DQN、拟合或全库回归。`source_sha256` 可缺省，仅作元数据而非门槛。旧 v1 产物不改写。
