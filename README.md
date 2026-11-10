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

- **仿真测试件**：设置扫频、不同走线长度、差模阻抗、有效介电常数、集肤/介质损耗以及两端相同的引入结构；生成 50 Ω 单端参考的 S4P 和 `truth.csv`。真值为走线传播衰减（dB/in）。模型不包含模式转换、噪声或不一致的引入结构，且色散模型经过简化；请勿将其视为真实 PCB 的完整模型。
- **Delta-L 计算**：导入长短线 S2P/S4P，指定走线长度及差分端口。默认 1、2、3、4 分别为输入正、输出正、输入负、输出负，即差分对 (1,3)/(2,4)。显示原始特征值损耗，导出频率、损耗、复数特征值、矩阵条件数和有效标志。长短线要求相同频率网格和参考阻抗。
- 矩阵方法：转换差模二端口 ABCD，计算 `M = T_long @ inv(T_short)` 的特征值，取小模值根计算 `-20 log10(|λ|)/ΔL`。假设长短线的左右引入结构分别相同、差分线均匀且可近似为解耦二端口。对于数值病态或传输几乎为零的点标记无效；噪声、模式转换与分支歧义可能影响结果。

仿真和分析结果需要通过导出/导入文件衔接。拟合、PLTS 曲线直接导入比较、不确定度、1L、3L 和 Dk/Df 未实现。下一阶段需要 PLTS 同一组 S4P 对应的特征值原始曲线、拟合曲线及设置，以确认差异及实现范围。仿真真值并非对真实测量准确度的验证。
