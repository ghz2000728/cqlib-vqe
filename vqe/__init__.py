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

"""VQE circuit factories, estimators, and solver."""
from .estimator import (
    NativeStatevectorEstimator,
    DirectStatevectorEstimator,
    FastStatevectorEstimator,
    EnergyEstimator,
    ParallelEnergyEstimator,
    CloudEnergyEstimator,
)
from .factory import UCCSD_Factory, UCCSDFactory
from .tianyan import (
    TianyanMeasurementGroup,
    TianyanMeasurementPlan,
    TianyanTermResult,
    TianyanGroupResult,
    TianyanEnergyResult,
    TianyanSubmittedEvaluation,
    TianyanEnergyEstimator,
    TianyanCloudEnergyEstimator,
)
from .solver import VQESolver
from .adaptive import (
    AdaptiveSelectionConfig,
    CandidateGradient,
    AdaptiveRound,
    AdaptiveSelectedUCCSDResult,
    AdaptiveSelectedUCCSDSolver,
)

__all__ = [
    "NativeStatevectorEstimator",
    "DirectStatevectorEstimator",
    "FastStatevectorEstimator",
    "EnergyEstimator",
    "ParallelEnergyEstimator",
    "CloudEnergyEstimator",
    "TianyanMeasurementGroup",
    "TianyanMeasurementPlan",
    "TianyanTermResult",
    "TianyanGroupResult",
    "TianyanEnergyResult",
    "TianyanSubmittedEvaluation",
    "TianyanEnergyEstimator",
    "TianyanCloudEnergyEstimator",
    "UCCSD_Factory",
    "UCCSDFactory",
    "VQESolver",
    "AdaptiveSelectionConfig",
    "CandidateGradient",
    "AdaptiveRound",
    "AdaptiveSelectedUCCSDResult",
    "AdaptiveSelectedUCCSDSolver",
]
