"""
Cycle Analysis Module
=====================

배터리 사이클 데이터 분석 모듈입니다.

주요 기능:
- 용량 분석 (열화율, EOL 예측)
- 효율 분석 (쿨롱 효율, 에너지 효율)
- DCIR 분석 (저항 변화)
- 사이클 분류 (RPT, 노화 패턴 등)

Author: Battery Analysis Team
Date: 2026-01-14
"""

from .capacity_analyzer import (
    CapacityAnalyzer,
    calculate_capacity_retention,
    calculate_capacity_fade_rate,
    predict_eol_cycle,
)

from .efficiency_analyzer import (
    EfficiencyAnalyzer,
    calculate_coulombic_efficiency,
    calculate_energy_efficiency,
)

from .dcir_analyzer import (
    DCIRAnalyzer,
    calculate_dcir_growth,
    analyze_resistance_trend,
)

from .cycle_classifier import (
    CycleClassifier,
    CycleType,
    classify_cycles,
)

__all__ = [
    # Capacity
    'CapacityAnalyzer',
    'calculate_capacity_retention',
    'calculate_capacity_fade_rate',
    'predict_eol_cycle',
    
    # Efficiency
    'EfficiencyAnalyzer',
    'calculate_coulombic_efficiency',
    'calculate_energy_efficiency',
    
    # DCIR
    'DCIRAnalyzer',
    'calculate_dcir_growth',
    'analyze_resistance_trend',
    
    # Classifier
    'CycleClassifier',
    'CycleType',
    'classify_cycles',
]

__version__ = '0.1.0'
