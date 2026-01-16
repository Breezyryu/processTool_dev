"""
Data Loader Module
==================

배터리 충방전기 데이터 로드 모듈입니다.

지원 충방전기:
- PNE (패턴 폴더 있음)
- Toyo (패턴 폴더 없음)

Author: Battery Analysis Team
Date: 2026-01-14
"""

from .base_loader import (
    BaseCyclerLoader,
    CycleData,
    ProfileData,
    CyclerType,
)

from .pne_loader import (
    PNELoader,
    pne_cycle_data,
    pne_profile_data,
)

from .toyo_loader import (
    ToyoLoader,
    toyo_cycle_data,
    toyo_profile_data,
)

from .factory import (
    detect_cycler_type,
    create_loader,
    load_cycle_data,
    load_profile_data,
)

__all__ = [
    # Base
    'BaseCyclerLoader',
    'CycleData',
    'ProfileData',
    'CyclerType',
    
    # PNE
    'PNELoader',
    'pne_cycle_data',
    'pne_profile_data',
    
    # Toyo
    'ToyoLoader',
    'toyo_cycle_data',
    'toyo_profile_data',
    
    # Factory
    'detect_cycler_type',
    'create_loader',
    'load_cycle_data',
    'load_profile_data',
]

__version__ = '0.1.0'
