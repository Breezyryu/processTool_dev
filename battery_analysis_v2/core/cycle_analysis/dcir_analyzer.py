"""
DCIR Analyzer Module
====================

DC 내부저항 (DCIR) 분석 모듈입니다.

주요 기능:
- DCIR 증가율 분석
- 저항 트렌드 분석
- SOH-DCIR 상관관계

Author: Battery Analysis Team
Date: 2026-01-14
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from scipy.stats import linregress


@dataclass
class DCIRGrowthResult:
    """DCIR 성장 분석 결과"""
    growth_rate_per_cycle: float   # 사이클당 증가율 (mΩ/cycle)
    growth_rate_percent: float     # 초기 대비 증가율 (%/cycle)
    initial_dcir: float            # 초기 DCIR (mΩ)
    current_dcir: float            # 현재 DCIR (mΩ)
    total_growth: float            # 총 증가량 (mΩ)
    total_growth_percent: float    # 총 증가율 (%)
    r_squared: float               # 회귀 결정 계수


def calculate_dcir_growth(cycles: np.ndarray,
                          dcir: np.ndarray) -> DCIRGrowthResult:
    """
    DCIR 증가율 계산
    
    Args:
        cycles: 사이클 배열
        dcir: DCIR 배열 (mΩ)
        
    Returns:
        DCIRGrowthResult 객체
    """
    cycles = np.asarray(cycles)
    dcir = np.asarray(dcir)
    
    # NaN 제거
    mask = ~np.isnan(dcir)
    cycles = cycles[mask]
    dcir = dcir[mask]
    
    if len(cycles) < 2:
        return DCIRGrowthResult(
            growth_rate_per_cycle=0,
            growth_rate_percent=0,
            initial_dcir=dcir[0] if len(dcir) > 0 else 0,
            current_dcir=dcir[-1] if len(dcir) > 0 else 0,
            total_growth=0,
            total_growth_percent=0,
            r_squared=0
        )
    
    # 선형 회귀
    slope, intercept, r_value, _, _ = linregress(cycles, dcir)
    
    initial = dcir[0]
    current = dcir[-1]
    
    return DCIRGrowthResult(
        growth_rate_per_cycle=slope,
        growth_rate_percent=(slope / initial * 100) if initial > 0 else 0,
        initial_dcir=initial,
        current_dcir=current,
        total_growth=current - initial,
        total_growth_percent=((current - initial) / initial * 100) if initial > 0 else 0,
        r_squared=r_value ** 2
    )


def analyze_resistance_trend(cycles: np.ndarray,
                              dcir: np.ndarray,
                              capacity: np.ndarray) -> Dict:
    """
    저항 트렌드 분석 (용량과의 상관관계 포함)
    
    Args:
        cycles: 사이클 배열
        dcir: DCIR 배열
        capacity: 용량 배열
        
    Returns:
        분석 결과 딕셔너리
    """
    cycles = np.asarray(cycles)
    dcir = np.asarray(dcir)
    capacity = np.asarray(capacity)
    
    # NaN 제거
    mask = ~(np.isnan(dcir) | np.isnan(capacity))
    dcir_clean = dcir[mask]
    capacity_clean = capacity[mask]
    cycles_clean = cycles[mask]
    
    results = {}
    
    # DCIR 성장
    dcir_growth = calculate_dcir_growth(cycles_clean, dcir_clean)
    results['dcir_growth'] = dcir_growth
    
    # DCIR-용량 상관관계
    if len(dcir_clean) > 2:
        corr = np.corrcoef(dcir_clean, capacity_clean)[0, 1]
        results['dcir_capacity_correlation'] = corr
        
        # 선형 관계: DCIR = a * capacity + b
        slope, intercept, r_value, _, _ = linregress(capacity_clean, dcir_clean)
        results['dcir_per_capacity_loss'] = slope
        results['correlation_r_squared'] = r_value ** 2
    else:
        results['dcir_capacity_correlation'] = 0
        results['dcir_per_capacity_loss'] = 0
        results['correlation_r_squared'] = 0
    
    return results


class DCIRAnalyzer:
    """
    DCIR 분석기 클래스
    
    사용 예시:
        >>> analyzer = DCIRAnalyzer(cycles, dcir)
        >>> growth = analyzer.analyze_growth()
        >>> trend = analyzer.get_trend()
    """
    
    def __init__(self,
                 cycles: np.ndarray,
                 dcir: np.ndarray,
                 capacity: Optional[np.ndarray] = None,
                 temperature: Optional[np.ndarray] = None):
        """
        Args:
            cycles: 사이클 배열
            dcir: DCIR 배열 (mΩ)
            capacity: 용량 배열 (옵션)
            temperature: 온도 배열 (옵션)
        """
        self.cycles = np.asarray(cycles)
        self.dcir = np.asarray(dcir)
        self.capacity = capacity
        self.temperature = temperature
        
        # NaN 마스크
        self._valid_mask = ~np.isnan(self.dcir)
    
    @property
    def valid_dcir(self) -> np.ndarray:
        """유효한 DCIR 값만"""
        return self.dcir[self._valid_mask]
    
    @property
    def valid_cycles(self) -> np.ndarray:
        """유효한 DCIR에 대응하는 사이클"""
        return self.cycles[self._valid_mask]
    
    @property
    def dcir_ratio(self) -> np.ndarray:
        """DCIR 비율 (초기 대비)"""
        valid = self.valid_dcir
        if len(valid) > 0 and valid[0] > 0:
            return self.dcir / valid[0]
        return np.ones_like(self.dcir)
    
    def analyze_growth(self) -> DCIRGrowthResult:
        """DCIR 성장 분석"""
        return calculate_dcir_growth(self.valid_cycles, self.valid_dcir)
    
    def analyze_with_capacity(self) -> Optional[Dict]:
        """용량과의 상관관계 분석"""
        if self.capacity is None:
            return None
        return analyze_resistance_trend(self.cycles, self.dcir, self.capacity)
    
    def get_trend(self, window: int = 20) -> pd.DataFrame:
        """
        DCIR 트렌드 (이동 평균)
        
        Args:
            window: 이동 평균 윈도우
            
        Returns:
            트렌드 DataFrame
        """
        result = pd.DataFrame({
            'cycle': self.cycles,
            'dcir': self.dcir,
            'dcir_ratio': self.dcir_ratio
        })
        
        # 이동 평균 (NaN 처리)
        dcir_series = pd.Series(self.dcir)
        result['dcir_rolling_mean'] = dcir_series.rolling(window, min_periods=1).mean()
        result['dcir_rolling_std'] = dcir_series.rolling(window, min_periods=1).std()
        
        return result
    
    def predict_dcir_at_cycle(self, target_cycle: int) -> float:
        """
        특정 사이클에서의 DCIR 예측
        
        Args:
            target_cycle: 목표 사이클
            
        Returns:
            예측 DCIR (mΩ)
        """
        valid_cycles = self.valid_cycles
        valid_dcir = self.valid_dcir
        
        if len(valid_cycles) < 2:
            return valid_dcir[-1] if len(valid_dcir) > 0 else 0
        
        slope, intercept, _, _, _ = linregress(valid_cycles, valid_dcir)
        return slope * target_cycle + intercept
    
    def temperature_correction(self, 
                               reference_temp: float = 25.0,
                               activation_energy: float = 30000) -> np.ndarray:
        """
        온도 보정된 DCIR
        
        Arrhenius 기반: DCIR_corrected = DCIR * exp(Ea/R * (1/T - 1/T_ref))
        
        Args:
            reference_temp: 기준 온도 (°C)
            activation_energy: 활성화 에너지 (J/mol)
            
        Returns:
            온도 보정된 DCIR 배열
        """
        if self.temperature is None:
            return self.dcir
        
        R = 8.314  # 기체 상수
        T_ref = reference_temp + 273.15
        T = self.temperature + 273.15
        
        correction_factor = np.exp(
            activation_energy / R * (1/T - 1/T_ref)
        )
        
        return self.dcir * correction_factor
    
    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame으로 변환"""
        data = {
            'cycle': self.cycles,
            'dcir': self.dcir,
            'dcir_ratio': self.dcir_ratio
        }
        
        if self.temperature is not None:
            data['temperature'] = self.temperature
        
        return pd.DataFrame(data)
