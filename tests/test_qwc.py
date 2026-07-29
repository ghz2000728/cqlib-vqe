from cqlib_vqe.vqe.estimator import EnergyEstimator


class DummyBackend:
    def run(self, circuit):
        return {"00": 1.0}


def test_qwc_group_master_is_updated():
    estimator = EnergyEstimator(DummyBackend())
    groups = estimator.group_hamiltonian([("XI", 1.0), ("IX", 2.0), ("ZZ", 3.0)])
    assert any(master == "XX" and len(terms) == 2 for master, terms in groups)
