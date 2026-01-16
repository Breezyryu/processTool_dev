"""
Loader Factory Module
=====================

충방전기 타입 자동 감지 및 로더 생성 팩토리입니다.

Author: Battery Analysis Team
Date: 2026-01-14
"""

from pathlib import Path
from typing import Optional, Union

from .base_loader import (
    BaseCyclerLoader,
    CycleData,
    ProfileData,
    CyclerType,
    detect_cycler_type,
)
from .pne_loader import PNELoader
from .toyo_loader import ToyoLoader


def create_loader(raw_path: Union[str, Path],
                  cycler_type: Optional[CyclerType] = None,
                  rated_capacity: Optional[float] = None,
                  initial_crate: float = 0.2) -> BaseCyclerLoader:
    """
    충방전기 로더 생성
    
    Args:
        raw_path: 데이터 폴더 경로
        cycler_type: 충방전기 타입 (None이면 자동 감지)
        rated_capacity: 정격 용량 (mAh)
        initial_crate: 초기 C-rate
        
    Returns:
        적절한 로더 인스턴스
        
    Raises:
        ValueError: 알 수 없는 충방전기 타입
    """
    if cycler_type is None:
        cycler_type = detect_cycler_type(raw_path)
    
    if cycler_type == CyclerType.PNE:
        return PNELoader(raw_path, rated_capacity, initial_crate)
    elif cycler_type == CyclerType.TOYO:
        return ToyoLoader(raw_path, rated_capacity, initial_crate)
    else:
        raise ValueError(f"Unknown cycler type: {cycler_type}. Path: {raw_path}")


def load_cycle_data(raw_path: Union[str, Path],
                    rated_capacity: Optional[float] = None,
                    initial_crate: float = 0.2,
                    **kwargs) -> CycleData:
    """
    사이클 데이터 자동 로드
    
    충방전기 타입을 자동 감지하고 적절한 로더로 데이터를 로드합니다.
    
    Args:
        raw_path: 데이터 폴더 경로
        rated_capacity: 정격 용량 (mAh)
        initial_crate: 초기 C-rate
        **kwargs: 추가 로더 옵션
        
    Returns:
        CycleData 객체
        
    Examples:
        >>> data = load_cycle_data("/path/to/battery/data")
        >>> print(f"Cycles: {len(data.cycle)}")
        >>> print(f"Capacity retention: {data.capacity_retention[-1]:.1%}")
    """
    loader = create_loader(raw_path, rated_capacity=rated_capacity, 
                          initial_crate=initial_crate)
    return loader.load_cycle_data(**kwargs)


def load_profile_data(raw_path: Union[str, Path],
                      cycle: int,
                      rated_capacity: Optional[float] = None,
                      initial_crate: float = 0.2,
                      **kwargs) -> ProfileData:
    """
    프로파일 데이터 자동 로드
    
    Args:
        raw_path: 데이터 폴더 경로
        cycle: 사이클 번호
        rated_capacity: 정격 용량 (mAh)
        initial_crate: 초기 C-rate
        **kwargs: 추가 로더 옵션
        
    Returns:
        ProfileData 객체
        
    Examples:
        >>> profile = load_profile_data("/path/to/data", cycle=100)
        >>> print(f"Duration: {profile.time[-1]:.0f}s")
        >>> df = profile.to_dataframe()
    """
    loader = create_loader(raw_path, rated_capacity=rated_capacity,
                          initial_crate=initial_crate)
    return loader.load_profile_data(cycle, **kwargs)


def load_multiple_channels(raw_paths: list,
                          rated_capacity: Optional[float] = None,
                          initial_crate: float = 0.2) -> dict:
    """
    다중 채널 데이터 로드
    
    Args:
        raw_paths: 데이터 폴더 경로 리스트
        rated_capacity: 정격 용량
        initial_crate: 초기 C-rate
        
    Returns:
        {path: CycleData} 딕셔너리
    """
    results = {}
    for path in raw_paths:
        try:
            data = load_cycle_data(path, rated_capacity, initial_crate)
            results[str(path)] = data
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            results[str(path)] = None
    return results
