# This code is part of cqlib.
#
# Copyright (C) 2025-2026 China Telecom Quantum Group.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Small numerical cqlib2 test double used when the native extension is absent."""
from __future__ import annotations

import copy
import sys
import types

import numpy as np

try:
    import cqlib  # noqa: F401
except ImportError:
    class Parameter:
        def __init__(self, value):
            self.value = value

        def evaluate(self, bindings=None):
            if isinstance(self.value, str):
                if bindings is None or self.value not in bindings:
                    raise ValueError(f"unbound {self.value}")
                return float(bindings[self.value])
            if isinstance(self.value, tuple):
                op, a, b = self.value
                av = a.evaluate(bindings) if isinstance(a, Parameter) else float(a)
                bv = b.evaluate(bindings) if isinstance(b, Parameter) else float(b)
                return av * bv if op == "mul" else av + bv
            return float(self.value)

        def __mul__(self, other):
            return Parameter(("mul", self, other))

        def __rmul__(self, other):
            return Parameter(("mul", other, self))

        def __add__(self, other):
            return Parameter(("add", self, other))

        __radd__ = __add__

    class Circuit:
        def __init__(self, num_qubits):
            self.num_qubits = int(num_qubits)
            self.operations = []

        def _add(self, name, *args):
            self.operations.append((name, *args))

        def x(self, q): self._add("x", q)
        def h(self, q): self._add("h", q)
        def rx(self, q, a): self._add("rx", q, a)
        def rz(self, q, a): self._add("rz", q, a)
        def cx(self, c, t): self._add("cx", c, t)
        def x2p(self, q): self._add("x2p", q)
        def x2m(self, q): self._add("x2m", q)
        def y2p(self, q): self._add("y2p", q)
        def y2m(self, q): self._add("y2m", q)
        def xy2p(self, q, a): self._add("xy2p", q, a)
        def xy2m(self, q, a): self._add("xy2m", q, a)
        def cz(self, c, t): self._add("cz", c, t)
        def m(self, *qubits): self._add("m", *qubits)

        def assign_parameters(self, bindings=None):
            out = Circuit(self.num_qubits)
            for op in self.operations:
                converted = []
                for item in op:
                    converted.append(item.evaluate(bindings) if isinstance(item, Parameter) else item)
                out.operations.append(tuple(converted))
            return out

        def __deepcopy__(self, memo):
            out = Circuit(self.num_qubits)
            out.operations = copy.deepcopy(self.operations, memo)
            return out

    def _single_apply(data, q, matrix):
        out = data.copy()
        stride = 1 << q
        for base in range(0, len(data), stride * 2):
            for offset in range(stride):
                i0 = base + offset
                i1 = i0 + stride
                a, b = data[i0], data[i1]
                out[i0] = matrix[0, 0] * a + matrix[0, 1] * b
                out[i1] = matrix[1, 0] * a + matrix[1, 1] * b
        return out

    class PauliString:
        def __init__(self, display):
            self.display = display
            self.num_qubits = len(display)

        @staticmethod
        def from_str(value):
            return PauliString(value.lstrip("+-i"))

        def expectation_statevector(self, state):
            return state.expectation(self)

    class Hamiltonian:
        def __init__(self, num_qubits):
            self.num_qubits = int(num_qubits)
            self.terms = []

        def add_term(self, pauli, coeff): self.terms.append((pauli, complex(coeff)))
        def simplify(self): pass
        def expectation_statevector(self, state):
            return state.expectation(self)

    class Statevector:
        def __init__(self, num_qubits):
            self.num_qubits = int(num_qubits)
            self.data = np.zeros(1 << self.num_qubits, dtype=complex)
            self.data[0] = 1

        @staticmethod
        def from_circuit(circuit):
            state = Statevector(circuit.num_qubits)
            for op in circuit.operations:
                name, *args = op
                getattr(state, f"apply_{name}")(*args)
            return state

        def apply_x(self, q):
            self.data = _single_apply(self.data, q, np.array([[0, 1], [1, 0]], complex))

        def apply_h(self, q):
            self.data = _single_apply(
                self.data, q, np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
            )

        def apply_rx(self, q, theta):
            c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
            self.data = _single_apply(self.data, q, np.array([[c, s], [s, c]], complex))

        def apply_rz(self, q, theta):
            self.data = _single_apply(
                self.data,
                q,
                np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]),
            )

        def apply_cx(self, control, target):
            out = np.zeros_like(self.data)
            for index, amp in enumerate(self.data):
                target_index = index ^ (1 << target) if ((index >> control) & 1) else index
                out[target_index] += amp
            self.data = out

        def apply_pauli_rotation(self, pauli, theta):
            mats = {
                "I": np.eye(2, dtype=complex),
                "X": np.array([[0, 1], [1, 0]], complex),
                "Y": np.array([[0, -1j], [1j, 0]], complex),
                "Z": np.array([[1, 0], [0, -1]], complex),
            }
            matrix = np.array([[1]], dtype=complex)
            for char in pauli.display:
                matrix = np.kron(matrix, mats[char])
            c = np.cos(theta / 2)
            s = -1j * np.sin(theta / 2)
            self.data = c * self.data + s * (matrix @ self.data)

        def expectation(self, observable):
            mats = {
                "I": np.eye(2, dtype=complex),
                "X": np.array([[0, 1], [1, 0]], complex),
                "Y": np.array([[0, -1j], [1j, 0]], complex),
                "Z": np.array([[1, 0], [0, -1]], complex),
            }
            terms = (
                [(observable, 1.0)]
                if isinstance(observable, PauliString)
                else observable.terms
            )
            total = 0j
            for pauli, coeff in terms:
                matrix = np.array([[1]], dtype=complex)
                for char in pauli.display:
                    matrix = np.kron(matrix, mats[char])
                total += coeff * np.vdot(self.data, matrix @ self.data)
            return float(total.real)

    cqlib = types.ModuleType("cqlib")
    cqlib.Circuit = Circuit
    cqlib.Parameter = Parameter
    cqlib.Qubit = int

    qis = types.ModuleType("cqlib.qis")
    qis.Statevector = Statevector
    qis.Hamiltonian = Hamiltonian
    qis.PauliString = PauliString

    def _qcis_dumps(circuit):
        lines = []
        for operation in circuit.operations:
            name, *args = operation
            upper = name.upper()
            if name in {"x", "h"}:
                lines.append(f"{upper} Q{args[0]}")
            elif name in {"rx", "rz"}:
                lines.append(f"{upper} Q{args[0]} {float(args[1]):.17g}")
            elif name in {"cx", "cz"}:
                lines.append(f"{upper} Q{args[0]} Q{args[1]}")
            elif name in {"x2p", "x2m", "y2p", "y2m"}:
                lines.append(f"{upper} Q{args[0]}")
            elif name in {"xy2p", "xy2m"}:
                lines.append(f"{upper} Q{args[0]} {float(args[1]):.17g}")
            elif name == "m":
                lines.append("M " + " ".join(f"Q{q}" for q in args))
            else:
                raise ValueError(f"unsupported fake QCIS operation {operation!r}")
        return "\n".join(lines) + ("\n" if lines else "")


    class _CompileResult:
        def __init__(self, circuit):
            self.circuit = circuit

    def _append_native_h(circuit, qubit):
        circuit.rz(qubit, np.pi)
        circuit.y2p(qubit)

    def _compile_circuit(circuit, target_basis=None):
        out = Circuit(circuit.num_qubits)
        for operation in circuit.operations:
            name, *args = operation
            if name == "x":
                out.x2p(args[0])
                out.x2p(args[0])
            elif name == "h":
                _append_native_h(out, args[0])
            elif name == "rx":
                out.y2m(args[0])
                out.rz(args[0], args[1])
                out.y2p(args[0])
            elif name == "rz":
                out.rz(args[0], args[1])
            elif name == "cx":
                control, target = args
                _append_native_h(out, target)
                out.cz(control, target)
                _append_native_h(out, target)
            elif name in {"x2p", "x2m", "y2p", "y2m", "xy2p", "xy2m", "cz", "m"}:
                out.operations.append(operation)
            else:
                raise ValueError(f"unsupported fake compiler operation {operation!r}")
        return _CompileResult(out)

    compile_module = types.ModuleType("cqlib.compile")
    compile_module.compile = _compile_circuit

    ir = types.ModuleType("cqlib.ir")
    qcis = types.ModuleType("cqlib.ir.qcis")
    qcis.dumps = _qcis_dumps
    ir.qcis = qcis

    sys.modules.update(
        {
            "cqlib": cqlib,
            "cqlib.qis": qis,
            "cqlib.compile": compile_module,
            "cqlib.ir": ir,
            "cqlib.ir.qcis": qcis,
        }
    )
