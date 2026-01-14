"""
dV/dQ Analysis Module
=====================

Battery dV/dQ 분석을 위한 핵심 모듈입니다.

주요 기능:
- Half-cell OCV 프로파일 정의
- Full-cell 전압 시뮬레이션
- LAM/LLI 열화 정량화

Author: Battery Analysis Team
Date: 2026-01-14
"""

from .halfcell_profiles import (
    HalfCellProfile,
    create_cathode_profile,
    create_anode_profile,
)

from .fullcell_fitting import (
    generate_simulation_full,
    FullCellSimulator,
    calculate_dvdq,
)

from .degradation_quantifier import (
    calculate_lam,
    calculate_lli,
    DegradationQuantifier,
)

__all__ = [
    # Half-cell profiles
    'HalfCellProfile',
    'create_cathode_profile', 
    'create_anode_profile',
    
    # Full-cell fitting
    'generate_simulation_full',
    'FullCellSimulator',
    'calculate_dvdq',
    
    # Degradation quantification
    'calculate_lam',
    'calculate_lli',
    'DegradationQuantifier',
]

__version__ = '0.1.0'
