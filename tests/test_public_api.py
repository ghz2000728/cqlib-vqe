from __future__ import annotations

import cqlib_vqe
import chemistry
import vqe


def test_canonical_package_metadata_and_exports():
    assert cqlib_vqe.__version__ == "1.4.3"
    assert cqlib_vqe.UCCSDFactory is vqe.UCCSDFactory
    assert cqlib_vqe.MolecularDataEngine is chemistry.MolecularDataEngine
    assert cqlib_vqe.TianyanEnergyEstimator is vqe.TianyanEnergyEstimator


def test_canonical_submodule_wrappers_share_implementations():
    from cqlib_vqe.chemistry.active_space import ActiveSpaceConfig
    from cqlib_vqe.vqe.factory import UCCSDFactory
    from cqlib_vqe.vqe.tianyan import TianyanEnergyEstimator

    assert ActiveSpaceConfig is chemistry.ActiveSpaceConfig
    assert UCCSDFactory is vqe.UCCSDFactory
    assert TianyanEnergyEstimator is vqe.TianyanEnergyEstimator
