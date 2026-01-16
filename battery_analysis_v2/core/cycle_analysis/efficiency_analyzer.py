"""
Efficiency Analyzer Module
==========================

배터리 효율 분석 모듈입니다.

주요 기능:
- 쿨롱 효율 (Coulombic Efficiency)
- 에너지 효율 (Energy Efficiency)
- 효율 트렌드 분석

Author: Battery Analysis Team
Date: 2026-01-14
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from scipy.stats import linregress


@dataclass
class EfficiencyStats:
    """효율 통계"""
    mean: float
    std: float
    min: float
    max: float
    median: float
    trend: float  # 사이클당 변화율


def calculate_coulombic_efficiency(charge_capacity: np.ndarray,
                                    discharge_capacity: np.ndarray) -> np.ndarray:
    """
    쿨롱 효율 계산
    
    CE = Discharge Capacity / Charge Capacity
    
    Args:
        charge_capacity: 충전 용량 배열
        discharge_capacity: 방전 용량 배열
        
    Returns:
        쿨롱 효율 배열 (0-1)
    """
    charge = np.asarray(charge_capacity)
    discharge = np.asarray(discharge_capacity)
    
    return np.divide(
        discharge, charge,
        out=np.ones_like(discharge, dtype=float),
        where=charge > 0
    )


def calculate_energy_efficiency(charge_energy: np.ndarray,
                                 discharge_energy: np.ndarray) -> np.ndarray:
    """
    에너지 효율 계산
    
    EE = Discharge Energy / Charge Energy
    
    Args:
        charge_energy: 충전 에너지 배열
        discharge_energy: 방전 에너지 배열
        
    Returns:
        에너지 효율 배열 (0-1)
    """
    charge = np.asarray(charge_energy)
    discharge = np.asarray(discharge_energy)
    
    return np.divide(
        discharge, charge,
        out=np.ones_like(discharge, dtype=float),
        where=charge > 0
    )


def calculate_round_trip_efficiency(charge_capacity: np.ndarray,
                                     discharge_capacity: np.ndarray,
                                     charge_voltage: np.ndarray,
                                     discharge_voltage: np.ndarray) -> np.ndarray:
    """
    Round-trip 효율 계산 (전압 포함)
    
    RTE = (Discharge Capacity * Discharge Voltage) / (Charge Capacity * Charge Voltage)
    
    Args:
        charge_capacity: 충전 용량
        discharge_capacity: 방전 용량
        charge_voltage: 평균 충전 전압
        discharge_voltage: 평균 방전 전압
        
    Returns:
        Round-trip 효율 배열
    """
    charge_energy = np.asarray(charge_capacity) * np.asarray(charge_voltage)
    discharge_energy = np.asarray(discharge_capacity) * np.asarray(discharge_voltage)
    
    return calculate_energy_efficiency(charge_energy, discharge_energy)


class EfficiencyAnalyzer:
    """
    효율 분석기 클래스
    
    사용 예시:
        >>> analyzer = EfficiencyAnalyzer(charge_cap, discharge_cap)
        >>> ce_stats = analyzer.coulombic_efficiency_stats()
        >>> trend = analyzer.analyze_trend()
    """
    
    def __init__(self,
                 charge_capacity: np.ndarray,
                 discharge_capacity: np.ndarray,
                 charge_energy: Optional[np.ndarray] = None,
                 discharge_energy: Optional[np.ndarray] = None,
                 cycles: Optional[np.ndarray] = None):
        """
        Args:
            charge_capacity: 충전 용량 배열
            discharge_capacity: 방전 용량 배열
            charge_energy: 충전 에너지 (옵션)
            discharge_energy: 방전 에너지 (옵션)
            cycles: 사이클 번호 (옵션)
        """
        self.charge_capacity = np.asarray(charge_capacity)
        self.discharge_capacity = np.asarray(discharge_capacity)
        self.charge_energy = charge_energy
        self.discharge_energy = discharge_energy
        self.cycles = cycles if cycles is not None else np.arange(len(charge_capacity))
    
    @property
    def coulombic_efficiency(self) -> np.ndarray:
        """쿨롱 효율"""
        return calculate_coulombic_efficiency(
            self.charge_capacity, self.discharge_capacity
        )
    
    @property
    def energy_efficiency(self) -> Optional[np.ndarray]:
        """에너지 효율"""
        if self.charge_energy is not None and self.discharge_energy is not None:
            return calculate_energy_efficiency(
                self.charge_energy, self.discharge_energy
            )
        return None
    
    def coulombic_efficiency_stats(self, 
                                    start_cycle: int = 0,
                                    end_cycle: Optional[int] = None) -> EfficiencyStats:
        """
        쿨롱 효율 통계
        
        Args:
            start_cycle: 시작 사이클 인덱스
            end_cycle: 종료 사이클 인덱스
            
        Returns:
            EfficiencyStats 객체
        """
        ce = self.coulombic_efficiency[start_cycle:end_cycle]
        cycles = self.cycles[start_cycle:end_cycle]
        
        # 트렌드 계산
        if len(cycles) > 2:
            slope, _, _, _, _ = linregress(cycles, ce)
        else:
            slope = 0.0
        
        return EfficiencyStats(
            mean=np.mean(ce),
            std=np.std(ce),
            min=np.min(ce),
            max=np.max(ce),
            median=np.median(ce),
            trend=slope
        )
    
    def energy_efficiency_stats(self,
                                 start_cycle: int = 0,
                                 end_cycle: Optional[int] = None) -> Optional[EfficiencyStats]:
        """에너지 효율 통계"""
        ee = self.energy_efficiency
        if ee is None:
            return None
        
        ee = ee[start_cycle:end_cycle]
        cycles = self.cycles[start_cycle:end_cycle]
        
        if len(cycles) > 2:
            slope, _, _, _, _ = linregress(cycles, ee)
        else:
            slope = 0.0
        
        return EfficiencyStats(
            mean=np.mean(ee),
            std=np.std(ee),
            min=np.min(ee),
            max=np.max(ee),
            median=np.median(ee),
            trend=slope
        )
    
    def analyze_trend(self, window: int = 50) -> pd.DataFrame:
        """
        효율 트렌드 분석 (이동 평균)
        
        Args:
            window: 이동 평균 윈도우 크기
            
        Returns:
            트렌드 DataFrame
        """
        ce = self.coulombic_efficiency
        
        result = pd.DataFrame({
            'cycle': self.cycles,
            'coulombic_efficiency': ce
        })
        
        # 이동 평균
        result['ce_rolling_mean'] = pd.Series(ce).rolling(window, min_periods=1).mean()
        result['ce_rolling_std'] = pd.Series(ce).rolling(window, min_periods=1).std()
        
        if self.energy_efficiency is not None:
            ee = self.energy_efficiency
            result['energy_efficiency'] = ee
            result['ee_rolling_mean'] = pd.Series(ee).rolling(window, min_periods=1).mean()
        
        return result
    
    def detect_anomalies(self, threshold: float = 3.0) -> np.ndarray:
        """
        효율 이상치 탐지 (Z-score 기반)
        
        Args:
            threshold: Z-score 임계값
            
        Returns:
            이상치 사이클 인덱스 배열
        """
        ce = self.coulombic_efficiency
        mean = np.mean(ce)
        std = np.std(ce)
        
        if std == 0:
            return np.array([])
        
        z_scores = np.abs((ce - mean) / std)
        anomalies = np.where(z_scores > threshold)[0]
        
        return self.cycles[anomalies]
    
    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame으로 변환"""
        data = {
            'cycle': self.cycles,
            'charge_capacity': self.charge_capacity,
            'discharge_capacity': self.discharge_capacity,
            'coulombic_efficiency': self.coulombic_efficiency
        }
        
        if self.energy_efficiency is not None:
            data['energy_efficiency'] = self.energy_efficiency
        
        return pd.DataFrame(data)
