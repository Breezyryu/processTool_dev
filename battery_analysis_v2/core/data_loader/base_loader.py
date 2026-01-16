"""
Base Loader Module
==================

배터리 충방전기 데이터 로더의 추상 기반 클래스입니다.

Author: Battery Analysis Team
Date: 2026-01-14
"""

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
import pandas as pd
import numpy as np


class CyclerType(Enum):
    """충방전기 타입"""
    PNE = auto()
    TOYO = auto()
    UNKNOWN = auto()


@dataclass
class CycleData:
    """
    사이클 데이터 구조
    
    Attributes:
        cycle: 사이클 번호
        charge_capacity: 충전 용량 (mAh)
        discharge_capacity: 방전 용량 (mAh)
        charge_energy: 충전 에너지 (mWh)
        discharge_energy: 방전 에너지 (mWh)
        efficiency: 쿨롱 효율 (방전/충전)
        dcir: DC 내부저항 (mΩ)
        temperature: 온도 (°C)
        rest_voltage: 휴지 전압 (V)
        average_voltage: 평균 전압 (V)
        rated_capacity: 정격 용량 (mAh)
    """
    cycle: np.ndarray
    charge_capacity: np.ndarray
    discharge_capacity: np.ndarray
    efficiency: np.ndarray
    rated_capacity: float
    
    charge_energy: Optional[np.ndarray] = None
    discharge_energy: Optional[np.ndarray] = None
    dcir: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    rest_voltage: Optional[np.ndarray] = None
    average_voltage: Optional[np.ndarray] = None
    original_cycle: Optional[np.ndarray] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame으로 변환"""
        data = {
            'cycle': self.cycle,
            'charge_capacity': self.charge_capacity,
            'discharge_capacity': self.discharge_capacity,
            'efficiency': self.efficiency,
        }
        
        if self.charge_energy is not None:
            data['charge_energy'] = self.charge_energy
        if self.discharge_energy is not None:
            data['discharge_energy'] = self.discharge_energy
        if self.dcir is not None:
            data['dcir'] = self.dcir
        if self.temperature is not None:
            data['temperature'] = self.temperature
        if self.rest_voltage is not None:
            data['rest_voltage'] = self.rest_voltage
        if self.average_voltage is not None:
            data['average_voltage'] = self.average_voltage
        
        return pd.DataFrame(data)
    
    @property
    def capacity_retention(self) -> np.ndarray:
        """용량 유지율 (첫 사이클 대비)"""
        if len(self.discharge_capacity) > 0:
            initial = self.discharge_capacity[0]
            return self.discharge_capacity / initial if initial > 0 else self.discharge_capacity
        return np.array([])
    
    @property
    def soh(self) -> np.ndarray:
        """SOH (정격 용량 대비)"""
        return self.discharge_capacity / self.rated_capacity


@dataclass
class ProfileData:
    """
    프로파일 데이터 구조
    
    Attributes:
        time: 시간 (초)
        voltage: 전압 (V)
        current: 전류 (mA)
        capacity: 용량 (mAh)
        temperature: 온도 (°C)
        condition: 상태 (1=충전, 2=방전, 3=휴지)
        step: 스텝 번호
    """
    time: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    capacity: np.ndarray
    
    temperature: Optional[np.ndarray] = None
    condition: Optional[np.ndarray] = None
    step: Optional[np.ndarray] = None
    cycle: Optional[int] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame으로 변환"""
        data = {
            'time': self.time,
            'voltage': self.voltage,
            'current': self.current,
            'capacity': self.capacity,
        }
        
        if self.temperature is not None:
            data['temperature'] = self.temperature
        if self.condition is not None:
            data['condition'] = self.condition
        if self.step is not None:
            data['step'] = self.step
        
        return pd.DataFrame(data)
    
    def filter_by_condition(self, condition: int) -> 'ProfileData':
        """조건별 필터링 (1=충전, 2=방전, 3=휴지)"""
        if self.condition is None:
            return self
        
        mask = self.condition == condition
        return ProfileData(
            time=self.time[mask],
            voltage=self.voltage[mask],
            current=self.current[mask],
            capacity=self.capacity[mask],
            temperature=self.temperature[mask] if self.temperature is not None else None,
            condition=self.condition[mask],
            step=self.step[mask] if self.step is not None else None,
            cycle=self.cycle
        )


class BaseCyclerLoader(ABC):
    """
    충방전기 데이터 로더 추상 기반 클래스
    
    서브클래스는 다음 메서드를 구현해야 합니다:
    - load_cycle_data(): 사이클 데이터 로드
    - load_profile_data(): 프로파일 데이터 로드
    """
    
    def __init__(self, raw_path: Union[str, Path], 
                 rated_capacity: Optional[float] = None,
                 initial_crate: float = 0.2):
        """
        Args:
            raw_path: 원시 데이터 폴더 경로
            rated_capacity: 정격 용량 (mAh), None이면 자동 추출
            initial_crate: 초기 C-rate (용량 자동 계산용)
        """
        self.raw_path = Path(raw_path)
        self.initial_crate = initial_crate
        
        if rated_capacity is not None:
            self._rated_capacity = rated_capacity
        else:
            self._rated_capacity = self._extract_capacity_from_path()
    
    @property
    def rated_capacity(self) -> float:
        """정격 용량"""
        return self._rated_capacity
    
    @property
    @abstractmethod
    def cycler_type(self) -> CyclerType:
        """충방전기 타입"""
        pass
    
    @abstractmethod
    def load_cycle_data(self, **kwargs) -> CycleData:
        """
        사이클 데이터 로드
        
        Returns:
            CycleData 객체
        """
        pass
    
    @abstractmethod
    def load_profile_data(self, cycle: int, **kwargs) -> ProfileData:
        """
        특정 사이클의 프로파일 데이터 로드
        
        Args:
            cycle: 사이클 번호
            
        Returns:
            ProfileData 객체
        """
        pass
    
    def _extract_capacity_from_path(self) -> float:
        """
        경로 이름에서 용량 추출
        
        예: "4500mAh_cell01" -> 4500.0
        """
        path_str = str(self.raw_path)
        
        # 정규식으로 용량 추출 (예: 4500mAh, 4.5Ah)
        match = re.search(r'(\d+(?:[\-\.]\d+)?)\s*mAh', path_str, re.IGNORECASE)
        if match:
            capacity_str = match.group(1).replace('-', '.')
            return float(capacity_str)
        
        # Ah 단위 시도
        match = re.search(r'(\d+(?:[\-\.]\d+)?)\s*Ah', path_str, re.IGNORECASE)
        if match:
            capacity_str = match.group(1).replace('-', '.')
            return float(capacity_str) * 1000
        
        return 0.0
    
    def estimate_capacity_from_data(self, df: pd.DataFrame, 
                                    current_col: str = 'current') -> float:
        """
        데이터에서 용량 추정 (첫 사이클 전류 기반)
        """
        if current_col in df.columns and len(df) > 0:
            max_current = df[current_col].abs().max()
            if max_current > 0:
                return max_current / self.initial_crate
        return 0.0


def detect_cycler_type(raw_path: Union[str, Path]) -> CyclerType:
    """
    충방전기 타입 자동 감지
    
    원본 함수: BatteryDataTool.py check_cycler()
    
    판별 기준:
    - Pattern 폴더가 있으면 PNE
    - Pattern 폴더가 없으면 Toyo
    
    Args:
        raw_path: 원시 데이터 폴더 경로
        
    Returns:
        CyclerType 열거형
    """
    path = Path(raw_path)
    pattern_path = path / "Pattern"
    
    if pattern_path.is_dir():
        return CyclerType.PNE
    else:
        # Toyo 확인: capacity.log 또는 숫자 폴더 존재
        capacity_log = path / "capacity.log"
        if capacity_log.exists():
            return CyclerType.TOYO
        
        # 숫자 폴더 확인 (000001 등)
        for item in path.iterdir():
            if item.is_file() and item.name.isdigit():
                return CyclerType.TOYO
    
    return CyclerType.UNKNOWN
