"""
Half-Cell OCV Profiles
======================

양극(Cathode) 및 음극(Anode) OCV 프로파일을 정의하고 관리하는 모듈입니다.

전기화학적 배경:
- 양극: Li_x CoO2, NMC, LFP 등의 OCV-SOC 관계
- 음극: Graphite, Silicon 등의 OCV-SOC 관계
- Full-cell OCV = V_cathode(SOC) - V_anode(SOC)

Author: Battery Analysis Team
Date: 2026-01-14
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple
from pathlib import Path


@dataclass
class HalfCellProfile:
    """
    Half-cell OCV 프로파일 데이터 구조
    
    Attributes:
        name: 프로파일 이름 (예: "NMC811", "Graphite")
        electrode_type: 'cathode' 또는 'anode'
        capacity: 용량 데이터 (mAh 또는 normalized)
        voltage: OCV 전압 데이터 (V)
        soc: SOC 데이터 (0-1 또는 0-100%)
        temperature: 측정 온도 (K)
    """
    name: str
    electrode_type: str  # 'cathode' or 'anode'
    capacity: np.ndarray
    voltage: np.ndarray
    soc: Optional[np.ndarray] = None
    temperature: float = 298.15  # 25°C in Kelvin
    
    def __post_init__(self):
        """데이터 유효성 검증"""
        if self.electrode_type not in ('cathode', 'anode'):
            raise ValueError(f"electrode_type must be 'cathode' or 'anode', got '{self.electrode_type}'")
        
        if len(self.capacity) != len(self.voltage):
            raise ValueError(f"capacity and voltage must have same length: {len(self.capacity)} vs {len(self.voltage)}")
        
        # SOC가 제공되지 않으면 용량에서 계산
        if self.soc is None:
            max_cap = np.max(self.capacity)
            self.soc = self.capacity / max_cap if max_cap > 0 else self.capacity
    
    @classmethod
    def from_csv(cls, filepath: Union[str, Path], 
                 name: str, 
                 electrode_type: str,
                 capacity_col: str = 'capacity',
                 voltage_col: str = 'voltage',
                 **kwargs) -> 'HalfCellProfile':
        """
        CSV 파일에서 프로파일 로드
        
        Args:
            filepath: CSV 파일 경로
            name: 프로파일 이름
            electrode_type: 'cathode' 또는 'anode'
            capacity_col: 용량 컬럼명
            voltage_col: 전압 컬럼명
            **kwargs: 추가 파라미터
            
        Returns:
            HalfCellProfile 인스턴스
        """
        df = pd.read_csv(filepath)
        return cls(
            name=name,
            electrode_type=electrode_type,
            capacity=df[capacity_col].values,
            voltage=df[voltage_col].values,
            **kwargs
        )
    
    def interpolate_voltage(self, target_capacity: np.ndarray) -> np.ndarray:
        """
        주어진 용량에서 전압 보간
        
        Args:
            target_capacity: 목표 용량 배열
            
        Returns:
            보간된 전압 배열
        """
        return np.interp(target_capacity, self.capacity, self.voltage)
    
    def apply_degradation(self, mass: float, slip: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        열화 파라미터 적용
        
        열화 모델:
            capacity_new = capacity * mass - slip
            
        Args:
            mass: 활물질 비율 (1.0 = 100% 잔존)
            slip: SOC 슬립 (리튬 손실에 의한)
            
        Returns:
            (수정된 용량, 원본 전압) 튜플
        """
        capacity_new = self.capacity * mass - slip
        return capacity_new, self.voltage


def create_cathode_profile(capacity: np.ndarray, 
                          voltage: np.ndarray,
                          name: str = "Cathode") -> HalfCellProfile:
    """
    양극 프로파일 생성 헬퍼 함수
    
    Args:
        capacity: 용량 데이터
        voltage: OCV 전압 데이터
        name: 프로파일 이름
        
    Returns:
        HalfCellProfile 인스턴스
    """
    return HalfCellProfile(
        name=name,
        electrode_type='cathode',
        capacity=np.asarray(capacity),
        voltage=np.asarray(voltage)
    )


def create_anode_profile(capacity: np.ndarray,
                        voltage: np.ndarray,
                        name: str = "Anode") -> HalfCellProfile:
    """
    음극 프로파일 생성 헬퍼 함수
    
    Args:
        capacity: 용량 데이터  
        voltage: OCV 전압 데이터
        name: 프로파일 이름
        
    Returns:
        HalfCellProfile 인스턴스
    """
    return HalfCellProfile(
        name=name,
        electrode_type='anode',
        capacity=np.asarray(capacity),
        voltage=np.asarray(voltage)
    )


# ============================================================
# 표준 전극 프로파일 (예시)
# ============================================================

def get_standard_nmc_profile() -> HalfCellProfile:
    """
    NMC 양극의 표준 OCV 프로파일 (예시)
    
    실제 사용 시 실험 데이터로 대체해야 합니다.
    """
    # 예시 데이터 - 실제 데이터로 대체 필요
    soc = np.linspace(0, 1, 100)
    # NMC OCV 모델 (단순화된 형태)
    voltage = 4.2 - 0.5 * (1 - soc) + 0.1 * np.sin(np.pi * soc)
    capacity = soc * 100  # normalized to 100 mAh
    
    return HalfCellProfile(
        name="NMC_Standard",
        electrode_type='cathode',
        capacity=capacity,
        voltage=voltage,
        soc=soc
    )


def get_standard_graphite_profile() -> HalfCellProfile:
    """
    Graphite 음극의 표준 OCV 프로파일 (예시)
    
    실제 사용 시 실험 데이터로 대체해야 합니다.
    """
    # 예시 데이터 - 실제 데이터로 대체 필요
    soc = np.linspace(0, 1, 100)
    # Graphite OCV 모델 (단순화된 형태 - 단계 구조)
    voltage = 0.1 + 0.2 * (1 - soc) - 0.05 * np.sin(3 * np.pi * soc)
    capacity = soc * 100  # normalized to 100 mAh
    
    return HalfCellProfile(
        name="Graphite_Standard",
        electrode_type='anode',
        capacity=capacity,
        voltage=voltage,
        soc=soc
    )
