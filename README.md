# PCB Delta-L Workbench

Python + PyQt6 PCB 传输线损耗研究工具。当前为 **2L 特征值算法原型**，尚未与 Intel Delta-L 4.0 规范或 Keysight PLTS 标杆数据比对，不应标称符合 Delta-L 4.0。

## 使用 uv 管理与启动

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后在项目根目录执行：

```bash
uv sync --locked
uv run --locked pcb-delta-l
```

`uv sync` 创建项目专用的 `.venv` 并根据 `uv.lock` 安装依赖；默认也安装开发依赖。也可使用 `uv run --locked python -m pcb_delta_l.gui`。Python 版本要求为 3.10 或更新版本。

运行测试：

```bash
uv run --locked pytest -q
```

增加依赖使用 `uv add 包名`，开发依赖使用 `uv add --dev 包名`；更新依赖后将 `pyproject.toml` 与 `uv.lock` 一同提交。仅安装运行依赖可使用 `uv sync --locked --no-dev`。

## 功能

- **仿真测试件**：设置扫频、不同走线长度、差模阻抗、有效介电常数、集肤/介质损耗以及两端相同的引入结构；生成 50 Ω 单端参考的 Touchstone 1.0 S4P 和 `truth.csv`。导出的 S4P 使用 ASCII、CRLF 和每频点四行矩阵数据，频点数不超过 16000；该格式已进行本地读回验证，尚需在 PLTS 验证。真值为走线传播衰减（dB/in）。模型不包含模式转换、噪声或不一致的引入结构，且色散模型经过简化；请勿将其视为真实 PCB 的完整模型。
- **Delta-L 计算**：导入长短线 S2P/S4P，指定走线长度及差分端口。默认 1、2、3、4 分别为输入正、输出正、输入负、输出负，即差分对 (1,3)/(2,4)。同图显示特征值损耗、拟合损耗和 Uncertainty，导出逐频点结果。长短线要求相同频率网格和参考阻抗。
- 矩阵方法：转换差模二端口 ABCD，计算 `M = T_long @ inv(T_short)` 的特征值，取小模值根计算 `-20 log10(|λ|)/ΔL`。假设长短线的左右引入结构分别相同、差分线均匀且可近似为解耦二端口。对于数值病态或传输几乎为零的点标记无效；噪声、模式转换与分支歧义可能影响结果。

仿真和分析结果通过导出和导入文件衔接。当前实现 2L，支持导入 PLTS 曲线对比。

## PLTS 导入排查

在 PLTS 中使用普通文件导入分别打开短线和长线，再从 Delta-L 选择；请先使用纯英文目录和文件名。Delta-L 的端口映射应与实际连接一致，仿真文件输入差分对为 (1,3)、输出差分对为 (2,4)，正负极分别为 1/3 和 2/4。频点数使用 16000 以下，初次测试建议采用默认 1601 点、2 in 和 5 in。若仍失败，请保存 S4P、PLTS 版本、实际长度设置、报错时机和截图以便定位。文件格式检查不能代替在 PLTS 中验证。

## Curve Fitted 与 Uncertainty

拟合采用 [IPC-TM-650 2.5.5.14](https://www.ipc.org/sites/default/files/test_methods_docs/TM%202.5.5.14.pdf) 式 (8) 的形式：`IL_fit = IL0 + a*x^b + c*x^2 + d*x`。其中 `x=(f-f0)/(fmax-f0)`，`IL0` 固定为首个有效损耗点。程序对候选指数 0.20～0.95（步长 0.05）分别求解线性最小二乘，取误差最小的一组。默认均匀权重，可选低频权重 `W=(1-x)^3`。至少需要 8 个有效频点，可设置拟合截止频率。

Uncertainty 使用中心频点左右约 1 GHz 窗口内的有符号残差，计算 `100*(abs(mean(r)) + 3*std(r))/abs(IL_fit)`，其中 `std` 为总体标准差。等间距数据以整点数确定窗口半宽。与两份 PLTS 样本一致，频带边缘无法取完整窗口时输出 0，有缺失数据的窗口留空。

两组默认仿真数据（10 in + 5 in、5 in + 2 in）验证了首点锚定、指数 0.55、均匀权重和窗口统计。导出和绘图将第二个等间距频点的 Eigen Value 与 Curve Fitted 各替换为各自前三点的平均值，复现样本中的边界处理。拟合和 Uncertainty 使用平均前的曲线。指数搜索网格为本实现的选择，其他样本和 PLTS 设置尚待对照。

### Delta-L 结果比较

“Delta-L 结果比较”页面支持载入两个 PLTS 格式或旧版程序结果 CSV，也可将参考 CSV 与当前计算结果比较。选择每个 PLTS 格式文件第一列实际的频率单位；旧版程序 CSV 的 `frequency_hz` 固定为 Hz。页面在 A 的频点上对 B 作线性插值，仅在有效数据段内计算 `B − A`，不外推，并显示最大绝对差及 RMS。

计算结果现在以 PLTS 四列 CSV 格式导出：`Time(ns), Eigen Value(Delta-L), Curve Fitted(Delta-L), Uncertainty(%)(Delta-L)`。为匹配所提供的 PLTS 示例而保留原始表头；**第一列数值为 MHz，第四列为局部残差窗口计算的 Uncertainty**。拟合区间外或无效值以空字段表示。

### 混模 S 参数查看与导出

在“混模 S 参数”页加载一个或两个四端口 Touchstone 文件，每个文件显示 SDD11/12/21/22 和 SCC11/12/21/22。端口输入依次为输入正、输出正、输入负、输出负，默认 `1,2,3,4`，即差分对 (1,3)/(2,4)。可逐条保存，或批量导出为 A_/B_ 前缀的 8 或 16 个 CSV 文件；每个文件的列名与 PLTS 单曲线导出一致，例如 `Frequency(MHz),SDD11(Log Mag)`。零幅值以 `-inf` 表示。
