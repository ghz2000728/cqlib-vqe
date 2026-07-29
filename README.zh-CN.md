# cqlib-vqe

`cqlib-vqe` 是面向 cqlib 2.x 的分子电子结构 VQE 软件包。它提供分子哈密顿量预处理、规范单重态 UCCSD 线路、Jordan--Wigner（JW）、Bravyi--Kitaev（BK）和 parity 映射、全算符池自适应选择、本地态矢量执行，以及天衍云端/硬件测量支持。

建议统一从 `cqlib_vqe` 导入公开接口。

## 功能概览

- 基于 PySCF 和 OpenFermion 的分子预处理，可生成活性空间报告与 Pauli 哈密顿量。
- 具有稳定压缩参数顺序的规范单重态 UCCSD 激发生成元。
- JW、BK 与 parity 映射在哈密顿量、激发生成元和 Hartree--Fock 参考态之间保持一致。
- `DirectStatevectorEstimator` 用于高频本地 VQE 目标函数，`NativeStatevectorEstimator` 用于线路或态矢量期望值计算。
- 全池自适应 selected-UCCSD：初始 CCSD 振幅为零的算符仍保留在候选池中。
- 天衍 QWC 分组测量规划，以及对任务顺序、结果量子位头和计数载荷的严格校验。

## 安装

请使用 Python 3.10 或更高版本，并建议在独立环境中安装：

```bash
git clone https://github.com/cq-lib/cqlib-vqe.git
cd cqlib-vqe

python -m venv .venv
source .venv/bin/activate              # Windows PowerShell：.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

按实际工作流选择安装项：

```bash
# 仅运行已编译的变分线路。
python -m pip install -e .

# 进行分子哈密顿量和 UCCSD 预处理。
python -m pip install -e ".[chemistry]"

# 在分子预处理基础上启用天衍执行。
python -m pip install -e ".[chemistry,tianyan]"

# 开发与测试环境。
python -m pip install -e ".[chemistry,tianyan,test]"
```

安装后运行完整测试：

```bash
pytest -q
```

## 快速开始：预编译单参数 VQE

当变分线路已经以 Pauli 生成元给出、且运行时不需要分子预处理时，可使用 `UCCSDFactory.from_compiled`。每个生成元由若干 `(pauli_string, coefficient)` 项组成；Pauli 字符串采用本项目的 `q0-left` 约定，即最左侧字符作用在量子位 0。

```python
from cqlib_vqe import DirectStatevectorEstimator, UCCSDFactory, VQESolver

# G = i * 0.5 * Y，哈密顿量为 H = Z。
factory = UCCSDFactory.from_compiled(
    n_qubits=1,
    n_electrons=0,
    generators=[[('Y', 0.5)]],
    initial_values=[0.1],
    construction_mode='jit',
)

estimator = DirectStatevectorEstimator(n_qubits=1)
solver = VQESolver(
    factory,
    estimator,
    optimizer_method='COBYLA',
    max_iter=30,
    execution_mode='auto',
)
result = solver.run([('Z', 1.0)])

print(result['optimal_value'])
print(result['optimal_params'])
```

`solver.run(...)` 返回字典，其中包括 `optimal_value`、`optimal_params`、`n_evals`、`n_iters`、`success`、`message`、`execution_mode` 和完整的能量 `history`。

该最小示例有意将 COBYLA 限制为 30 次评估。因此即使已经达到数值最小值，也可能因评估预算耗尽而返回 `success=False`。用于实际研究时，应提高 `max_iter`，并采用与研究目标相符的能量和参数收敛判据。

可直接运行对应示例：

```bash
python examples/minimal_compiled_smoke.py
```

## 分子 VQE 工作流

分子计算从 `MolecularDataEngine` 开始。调用 `run()` 后，可获得 `n_qubits`、`n_electrons`、`hamiltonian_data`、CCSD 初始振幅与参考能量。随后构建 UCCSD 线路工厂、选择估计器，并通过 `VQESolver` 执行优化。

```python
from cqlib_vqe import (
    DirectStatevectorEstimator,
    MolecularDataEngine,
    UCCSDFactory,
    VQESolver,
)

molecule = MolecularDataEngine(
    geometry=[
        ('H', (0.0, 0.0, 0.0)),
        ('H', (0.0, 0.0, 0.735)),
    ],
    basis='sto-3g',
    multiplicity=1,
    charge=0,
    mapper_type='jw',
    excitation_threshold=1e-3,
).run()

factory = UCCSDFactory(
    molecule,
    construction_mode='jit',
    trotter_steps=1,
    trotter_order=2,
)
estimator = DirectStatevectorEstimator(n_qubits=molecule.n_qubits)
solver = VQESolver(
    factory,
    estimator,
    optimizer_method='COBYLA',
    max_iter=100,
    tol=1e-7,
    execution_mode='auto',
)

def progress(_parameters, energy, evaluation):
    if evaluation == 1 or evaluation % 10 == 0:
        print(f'eval={evaluation:4d}  energy={energy:.12f} Ha')

result = solver.run(molecule.hamiltonian_data, callback=progress)
print('VQE energy:', result['optimal_value'])
print('reference energies:', molecule.reference_energies)
```

维护中的 H2 示例可直接运行：

```bash
python examples/h2_vqe_cqlib2.py
```

### 执行模式的选择

本地模拟建议使用 `execution_mode='auto'`。若线路工厂与估计器支持融合态矢量路径，它会自动选择高效的直接态矢量执行；否则退回线路执行。对于云端估计器等必须以线路提交的场景，应使用 `execution_mode='circuit'`。只有同时具备 `factory.build_statevector(...)` 和 `estimator.evaluate_parameters(...)` 时，才应显式使用 `execution_mode='direct_statevector'`。

`construction_mode='jit'` 为重复求值生成数值线路；`construction_mode='bind'` 保留符号模板，适用于需要参数绑定的执行路径。

## 映射与活性空间

创建 `MolecularDataEngine` 时，可将 `mapper_type` 设为 `"jw"`、`"bk"` 或 `"parity"`：

```python
molecule = MolecularDataEngine(
    geometry=[('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.735))],
    basis='sto-3g',
    mapper_type='bk',
).run()
```

所选映射会一致地作用于分子哈密顿量、UCCSD 激发生成元与 Hartree--Fock 参考态。对 BK 和 parity 映射，参考态的计算基比特模式通常不能仅由其汉明重量判断，因此不应以 JW 占据数规则替代线路工厂生成的参考态。

对于较大体系，请在运行分子引擎前显式配置活性空间；量子位预算必须为正偶数：

```python
from cqlib_vqe import ActiveSpaceConfig, MolecularDataEngine

active_space = ActiveSpaceConfig.automatic(
    max_active_qubits=10,
    freeze_core=True,
    min_virtual_orbitals=1,
)
molecule = MolecularDataEngine(
    geometry=[('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.596))],
    basis='sto-3g',
    mapper_type='jw',
    active_space=active_space,
).run()

print(molecule.active_space_report)
```

使用 `ActiveSpaceConfig.full()` 可保留完整轨道空间。每个数值结果都应记录活性空间报告，因为它决定了实际求解的哈密顿量与参数池。

## 自适应 selected-UCCSD

自适应求解器以规范 CCSD 振幅初始化变分线路，随后扫描完整的剩余规范算符池。即使某个算符的初始 CCSD 振幅为零，它仍是候选算符，并按照能量梯度参与筛选。

五分子运行器提供了可审计的端到端流程。建议先以 H2 和较少的算符池扫描轮数开始：

```bash
python examples/run_five_adaptive_full_pool.py h2 \\
  --output-dir run_outputs/h2_adaptive \\
  --pool-rounds 2 \\
  --maxiter 80
```

输出目录会包含每个分子的日志和 JSON 记录，以及汇总 CSV 与 JSON。该脚本支持 `h2`、`h4`、`lih`、`beh2` 和 `h2o`；省略位置参数时将依次运行全部五个分子。仅当确实需要且计算资源充足时，才使用 `--active-space-policy full`。

## 天衍执行

天衍执行会提交真实测量任务，并可能消耗平台额度。安装 `tianyan` 可选依赖后，请在版本控制之外配置凭据，并先查询当前账号可用的后端名称：

```bash
export TIANYAN_API_KEY='YOUR_API_KEY'
python tools/check_tianyan_api.py --online
```

然后指定后端和物理量子位映射。映射按“逻辑量子位到物理量子位”的顺序给出，长度必须严格等于 `n_qubits`：

```bash
export TIANYAN_DEVICE='BACKEND_NAME'
export TIANYAN_PHYSICAL_QUBITS='0,1,2,3'
export TIANYAN_SHOTS=2000
export TIANYAN_MAXITER=10
export TIANYAN_CALIBRATION=disabled
python -u examples/h2_vqe_tianyan.py
```

`TianyanEnergyEstimator` 会先对 Pauli 项进行量子位逐项对易分组，再提交测量任务；返回后会校验任务顺序、测量量子位头与计数载荷。估计器不会自行推断路由，也不会静默修复后端元数据。请勿提交 API 密钥、凭据文件或含有账号信息的原始云端输出。

## 常用入口

| 任务 | 命令 |
| --- | --- |
| 最小预编译 VQE | `python examples/minimal_compiled_smoke.py` |
| 本地 H2 UCCSD-VQE | `python examples/h2_vqe_cqlib2.py` |
| 自适应 H2 工作流 | `python examples/run_five_adaptive_full_pool.py h2` |
| 天衍接口检查 | `python tools/check_tianyan_api.py --online` |
| 完整测试集 | `pytest -q` |

## 许可证

Apache License 2.0，详见 [LICENSE](LICENSE)。
