"""Runtime compatibility check against the installed cqlib2 Python binding."""

from __future__ import annotations

import math
import sys


def require(obj, name: str) -> None:
    if not hasattr(obj, name):
        raise RuntimeError(f"Missing required cqlib2 API: {obj!r}.{name}")


def main() -> int:
    try:
        import cqlib
        from cqlib import Circuit, Parameter
        from cqlib.ir import qcis
        from cqlib.qis import Hamiltonian, PauliString, Statevector
    except Exception as exc:
        print(f"[FAIL] Cannot import cqlib2 API: {exc}")
        return 1

    for method in ("x", "h", "rx", "rz", "cx", "assign_parameters"):
        require(Circuit, method)
    for method in ("from_circuit", "apply_x", "apply_h", "apply_rx", "apply_rz", "apply_cx", "expectation"):
        require(Statevector, method)
    for method in ("from_str", "expectation_statevector"):
        require(PauliString, method)
    for method in ("add_term", "simplify", "expectation_statevector"):
        require(Hamiltonian, method)
    require(qcis, "dumps")

    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(1, theta)
    bound = circuit.assign_parameters({"theta": 0.0})
    state = Statevector.from_circuit(bound)

    zz = PauliString.from_str("ZZ")
    hamiltonian = Hamiltonian(2)
    hamiltonian.add_term(zz, 1.0)
    hamiltonian.simplify()
    value = state.expectation(hamiltonian)
    if not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Unexpected Bell-state <ZZ>: {value}")

    qcis_text = qcis.dumps(bound)
    if not isinstance(qcis_text, str) or not qcis_text.strip():
        raise RuntimeError("qcis.dumps returned an empty/non-string result")

    print("[PASS] cqlib2 imports and required APIs are available")
    print("[PASS] symbolic parameter binding works")
    print("[PASS] Statevector.from_circuit and native Hamiltonian expectation work")
    print("[PASS] qcis.dumps works")

    if callable(getattr(Statevector, "apply_pauli_rotation", None)):
        native_state = Statevector(1)
        native_state.apply_x(0)
        native_state.apply_pauli_rotation(PauliString.from_str("Z"), math.pi)
        native_value = native_state.expectation(PauliString.from_str("Z"))
        if not math.isclose(native_value, -1.0, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Unexpected state after native Pauli rotation: {native_value}")
        print("[PASS] optional native Statevector.apply_pauli_rotation is available")
    else:
        print("[INFO] native Pauli rotation is unavailable; decomposed execution remains available")
    print(f"cqlib module: {getattr(cqlib, '__file__', '<native>')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
