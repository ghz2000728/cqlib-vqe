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

from cqlib_vqe.vqe.estimator import EnergyEstimator


class DummyBackend:
    def run(self, circuit):
        return {"00": 1.0}


def test_qwc_group_master_is_updated():
    estimator = EnergyEstimator(DummyBackend())
    groups = estimator.group_hamiltonian([("XI", 1.0), ("IX", 2.0), ("ZZ", 3.0)])
    assert any(master == "XX" and len(terms) == 2 for master, terms in groups)
