"""
Capacity Analyzer Module
========================

배터리 용량 분석 모듈입니다.

주요 기능:
- 용량 유지율 계산
- 열화율 (Fade Rate) 분석
- EOL (End of Life) 예측

Author: Battery Analysis Team
Date: 2026-01-14
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, Dict
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.stats import linregress




@dataclass
class FadeAnalysisResult:
    """열화 분석 결과"""
    fade_rate_per_cycle: float    # 사이클당 열화율 (%/cycle)
    fade_rate_per_100_cycles: float  # 100사이클당 열화율 (%)
    r_squared: float              # 회귀 결정 계수
    initial_capacity: float       # 초기 용량 (%)
    current_capacity: float       # 현재 용량 (%)
    total_fade: float            # 총 열화량 (%)
    cycles_analyzed: int          # 분석 사이클 수


@dataclass
class EOLPrediction:
    """EOL 예측 결과"""
    predicted_cycle: int          # EOL 예상 사이클
    confidence_low: int           # 95% 신뢰구간 하한
    confidence_high: int          # 95% 신뢰구간 상한
    eol_threshold: float         # EOL 기준 (%)
    current_soh: float           # 현재 SOH (%)
    remaining_cycles: int         # 잔여 사이클
    method: str                   # 예측 방법


def calculate_capacity_retention(capacity: np.ndarray,
                                  reference: Optional[float] = None) -> np.ndarray:
    """
    용량 유지율 계산
    
    Args:
        capacity: 용량 배열 (mAh)
        reference: 기준 용량 (None이면 첫 값 사용)
        
    Returns:
        용량 유지율 배열 (0-1)
    """
    capacity = np.asarray(capacity)
    
    if reference is None:
        reference = capacity[0] if len(capacity) > 0 else 1.0
    
    if reference == 0:
        return np.ones_like(capacity)
    
    return capacity / reference


def calculate_capacity_fade_rate(cycles: np.ndarray,
                                  capacity: np.ndarray,
                                  method: str = 'linear') -> FadeAnalysisResult:
    """
    용량 열화율 계산
    
    Args:
        cycles: 사이클 배열
        capacity: 용량 배열 (mAh 또는 %)
        method: 계산 방법 ('linear', 'sqrt')
        
    Returns:
        FadeAnalysisResult 객체
    """
    cycles = np.asarray(cycles)
    capacity = np.asarray(capacity)
    
    # 용량을 백분율로 정규화
    if capacity[0] > 10:  # mAh로 판단
        capacity_pct = capacity / capacity[0] * 100
    else:
        capacity_pct = capacity * 100 if capacity.max() <= 1 else capacity
    
    if method == 'linear':
        # 선형 회귀
        slope, intercept, r_value, p_value, std_err = linregress(cycles, capacity_pct)
        fade_rate = -slope  # 양수로 변환 (열화율)
        r_squared = r_value ** 2
    elif method == 'sqrt':
        # 제곱근 모델: C = C0 - a * sqrt(n)
        def sqrt_model(x, c0, a):
            return c0 - a * np.sqrt(x)
        
        try:
            popt, pcov = curve_fit(sqrt_model, cycles, capacity_pct, 
                                   p0=[100, 0.1], maxfev=5000)
            fade_rate = popt[1]  # sqrt 계수
            
            # R² 계산
            predicted = sqrt_model(cycles, *popt)
            ss_res = np.sum((capacity_pct - predicted) ** 2)
            ss_tot = np.sum((capacity_pct - np.mean(capacity_pct)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
        except Exception:
            # 폴백: 선형 회귀
            slope, intercept, r_value, _, _ = linregress(cycles, capacity_pct)
            fade_rate = -slope
            r_squared = r_value ** 2
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return FadeAnalysisResult(
        fade_rate_per_cycle=fade_rate,
        fade_rate_per_100_cycles=fade_rate * 100,
        r_squared=r_squared,
        initial_capacity=capacity_pct[0],
        current_capacity=capacity_pct[-1],
        total_fade=capacity_pct[0] - capacity_pct[-1],
        cycles_analyzed=len(cycles)
    )


def predict_eol_cycle(cycles: np.ndarray,
                       capacity: np.ndarray,
                       eol_threshold: float = 80.0,
                       method: str = 'linear',
                       max_cycle: int = 10000) -> EOLPrediction:
    """
    EOL 사이클 예측
    
    Args:
        cycles: 사이클 배열
        capacity: 용량 배열 (mAh 또는 %)
        eol_threshold: EOL 기준 (% of initial)
        method: 예측 방법 ('linear', 'sqrt', 'polynomial')
        max_cycle: 최대 예측 사이클
        
    Returns:
        EOLPrediction 객체
    """
    cycles = np.asarray(cycles)
    capacity = np.asarray(capacity)
    
    # 용량 정규화
    if capacity[0] > 10:
        capacity_pct = capacity / capacity[0] * 100
    else:
        capacity_pct = capacity * 100 if capacity.max() <= 1 else capacity
    
    current_soh = capacity_pct[-1]
    
    if method == 'linear':
        slope, intercept, r_value, _, std_err = linregress(cycles, capacity_pct)
        
        if slope >= 0:  # 열화가 없거나 용량 증가
            return EOLPrediction(
                predicted_cycle=max_cycle,
                confidence_low=max_cycle,
                confidence_high=max_cycle,
                eol_threshold=eol_threshold,
                current_soh=current_soh,
                remaining_cycles=max_cycle - cycles[-1],
                method=method
            )
        
        # EOL 사이클: slope * x + intercept = eol_threshold
        eol_cycle = (eol_threshold - intercept) / slope
        
        # 95% 신뢰구간 계산 (간략화)
        n = len(cycles)
        if n > 2 and std_err > 0:
            t_value = 1.96  # 95% 신뢰구간
            cycle_std = std_err * np.sqrt(n)
            confidence_low = int((eol_threshold - intercept) / (slope - t_value * std_err))
            confidence_high = int((eol_threshold - intercept) / (slope + t_value * std_err))
        else:
            confidence_low = int(eol_cycle * 0.9)
            confidence_high = int(eol_cycle * 1.1)
        
    elif method == 'sqrt':
        def sqrt_model(x, c0, a):
            return c0 - a * np.sqrt(x)
        
        try:
            popt, pcov = curve_fit(sqrt_model, cycles, capacity_pct)
            c0, a = popt
            
            # EOL: c0 - a * sqrt(x) = eol_threshold
            eol_cycle = ((c0 - eol_threshold) / a) ** 2
            
            # 간단한 신뢰구간
            confidence_low = int(eol_cycle * 0.85)
            confidence_high = int(eol_cycle * 1.15)
            
        except Exception:
            return predict_eol_cycle(cycles, capacity, eol_threshold, 'linear', max_cycle)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    eol_cycle = min(int(eol_cycle), max_cycle)
    eol_cycle = max(eol_cycle, cycles[-1])  # 현재 사이클보다는 크게
    
    confidence_low = max(confidence_low, cycles[-1])
    confidence_high = min(confidence_high, max_cycle)
    
    return EOLPrediction(
        predicted_cycle=eol_cycle,
        confidence_low=confidence_low,
        confidence_high=confidence_high,
        eol_threshold=eol_threshold,
        current_soh=current_soh,
        remaining_cycles=eol_cycle - cycles[-1],
        method=method
    )


class CapacityAnalyzer:
    """
    용량 분석기 클래스
    
    사용 예시:
        >>> analyzer = CapacityAnalyzer(cycles, discharge_capacity)
        >>> fade = analyzer.analyze_fade()
        >>> eol = analyzer.predict_eol(threshold=80)
    """
    
    def __init__(self, cycles: np.ndarray, capacity: np.ndarray,
                 rated_capacity: Optional[float] = None):
        """
        Args:
            cycles: 사이클 배열
            capacity: 방전 용량 배열
            rated_capacity: 정격 용량
        """
        self.cycles = np.asarray(cycles)
        self.capacity = np.asarray(capacity)
        self.rated_capacity = rated_capacity or self.capacity[0]
    
    @property
    def retention(self) -> np.ndarray:
        """용량 유지율"""
        return calculate_capacity_retention(self.capacity, self.capacity[0])
    
    @property
    def soh(self) -> np.ndarray:
        """SOH (정격 대비)"""
        return self.capacity / self.rated_capacity
    
    def analyze_fade(self, method: str = 'linear') -> FadeAnalysisResult:
        """열화율 분석"""
        return calculate_capacity_fade_rate(self.cycles, self.capacity, method)
    
    def predict_eol(self, threshold: float = 80, 
                    method: str = 'linear') -> EOLPrediction:
        """EOL 예측"""
        return predict_eol_cycle(self.cycles, self.capacity, threshold, method)
    
    def get_cycle_at_soh(self, target_soh: float) -> Optional[int]:
        """특정 SOH에 도달한 사이클 찾기"""
        soh_pct = self.retention * 100
        below = np.where(soh_pct <= target_soh)[0]
        
        if len(below) > 0:
            return int(self.cycles[below[0]])
        return None
    
    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame으로 변환"""
        return pd.DataFrame({
            'cycle': self.cycles,
            'capacity': self.capacity,
            'retention': self.retention,
            'soh': self.soh
        })
