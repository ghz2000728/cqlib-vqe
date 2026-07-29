# cqlib-vqe

`cqlib-vqe` 面向 cqlib 2.x，提供活性空间分子哈密顿量、canonical singlet-UCCSD、完整算符池自适应筛选、高性能态矢目标函数，以及严格的天衍云仿真/真机能量估计器。

## 主要能力

- canonical 闭壳层 singlet-UCCSD 与可审计的参数编号；
- Jordan-Wigner、Bravyi-Kitaev 与 parity 映射端到端一致：分子哈密顿量、canonical 激发生成元和 Hartree-Fock 参考态使用同一编码；
- 完整算符池自适应筛选，可重新发现初始 CCSD 振幅为零但变分上重要的生成元；
- cqlib 支持时使用原生 `Statevector.apply_pauli_rotation`；
- 哈密顿量内容缓存和可复用 Pauli 执行计划；
- 天衍 QWC 分组、严格任务顺序/测量比特/计数校验；
- 提交前编译为天衍原生 QCIS 门集；
- 推荐使用 `cqlib_vqe` 导入命名空间，同时在 1.x 中保留 `vqe`、`chemistry` 旧导入兼容。

## 安装

```bash
pip install -e ".[chemistry,tianyan,test]"
pytest -q
```

## 本地 H2

```bash
python examples/h2_vqe_cqlib2.py
```

## 天衍云仿真/真机

先查询账户实际可用的后端名称：

```bash
export TIANYAN_API_KEY='YOUR_API_KEY'
python tools/check_tianyan_api.py --online
```

再运行：

```bash
export TIANYAN_DEVICE='查询结果中的后端名称'
export TIANYAN_PHYSICAL_QUBITS='0,1,2,3'
export TIANYAN_SHOTS=2000
export TIANYAN_MAXITER=10
export TIANYAN_CALIBRATION=disabled
python -u examples/h2_vqe_tianyan.py
```

设备名称区分大小写，本包不硬编码具体设备。物理比特映射必须满足所选设备的拓扑约束；估计器不会静默路由，也不会修补任务顺序、测量比特头或空计数结果。

完整说明见 [docs/TIANYAN.md](docs/TIANYAN.md)。

## 验证结果

云上全振幅 H2 端到端验证得到 `-1.136986799379 Ha`，FCI 参考值为 `-1.137306035753 Ha`，绝对误差 `0.319 mHa`。这验证了原生 QCIS、QWC 分组、任务回收、bitstring 解析和能量重建链路。

## 许可证

Apache License 2.0。
