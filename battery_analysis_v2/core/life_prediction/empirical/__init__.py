"""
Life Prediction - Empirical Module
===================================

경험적 용량 열화 모델을 위한 모듈입니다.

핵심 모델:
    capacity(t, T) = 1 - exp(a*T + b) * (cycle*fd)^b1 - exp(c*T + d) * (cycle*fd)^(e*T + f)
    
여기서:
    - T: 온도 (K)
    - cycle: 사이클 수
    - fd: 가속 계수
    - a,b,b1,c,d,e,f: 피팅 파라미터

원본: BatteryDataTool.py eu_fitting_confirm_button() 내부 capacityfit() 함수

Author: Battery Analysis Team
Date: 2026-01-14
"""

from .capacity_fit import (
    capacityfit,
    capacityfit_numba,
    swellingfit,
    CapacityDegradationModel,
    fit_capacity_model,
)

from .eu_prediction import (
    EULifePredictor,
    predict_eu_cycle_life,
)

from .approval_prediction import (
    ApprovalLifePredictor,
    predict_approval_cycle_life,
)

__all__ = [
    # Capacity fitting
    'capacityfit',
    'capacityfit_numba',
    'swellingfit',
    'CapacityDegradationModel',
    'fit_capacity_model',
    
    # EU prediction
    'EULifePredictor',
    'predict_eu_cycle_life',
    
    # Approval prediction
    'ApprovalLifePredictor',
    'predict_approval_cycle_life',
]

__version__ = '0.1.0'
