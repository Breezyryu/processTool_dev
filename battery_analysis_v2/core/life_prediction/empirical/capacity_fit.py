"""
Capacity Degradation Fitting Module
====================================

경험적 용량 열화 모델 피팅 모듈입니다.

핵심 수식:
    capacity(x, T) = 1 - exp(a*T + b) * (x*fd)^b1 - exp(c*T + d) * (x*fd)^(e*T + f)
    
물리적 의미:
    - 첫 번째 항: Calendar aging (저장 열화)
    - 두 번째 항: Cycle aging (사이클 열화)
    - Arrhenius 온도 의존성 반영

원본: BatteryDataTool.py eu_fitting_confirm_button() 내부 함수

Author: Battery Analysis Team
Date: 2026-01-14
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Callable, Dict, List
from dataclasses import dataclass, field
from scipy.optimize import curve_fit
import warnings




# ============================================================
# 순수 Python 구현
# ============================================================

def capacityfit(x: Tuple[np.ndarray, np.ndarray],
                a_par: float, b_par: float, b1_par: float,
                c_par: float, d_par: float, e_par: float, 
                f_par: float, f_d: float) -> np.ndarray:
    """
    경험적 용량 열화 모델 (순수 Python)
    
    원본: BatteryDataTool.py capacityfit 함수
    
    수식:
        capacity = 1 - exp(a*T + b) * (cycle*fd)^b1 - exp(c*T + d) * (cycle*fd)^(e*T + f)
    
    Args:
        x: (cycle, temperature) 튜플
           - x[0]: 사이클 수 또는 시간
           - x[1]: 온도 (K)
        a_par: a 파라미터 (Calendar aging 온도 계수)
        b_par: b 파라미터 (Calendar aging 상수)
        b1_par: b1 파라미터 (Calendar aging 지수)
        c_par: c 파라미터 (Cycle aging 온도 계수)
        d_par: d 파라미터 (Cycle aging 상수)
        e_par: e 파라미터 (Cycle aging 온도-지수 계수)
        f_par: f 파라미터 (Cycle aging 지수 상수)
        f_d: 가속 계수
        
    Returns:
        잔존 용량 비율 (0-1)
    """
    cycle = x[0]
    temperature = x[1]
    
    # Calendar aging 항
    term1 = np.exp(a_par * temperature + b_par) * (cycle * f_d) ** b1_par
    
    # Cycle aging 항
    term2 = np.exp(c_par * temperature + d_par) * (cycle * f_d) ** (e_par * temperature + f_par)
    
    # 잔존 용량
    capacity = 1 - term1 - term2
    
    return capacity


def swellingfit(x: Tuple[np.ndarray, np.ndarray],
                a_par: float, b_par: float, b1_par: float,
                c_par: float, d_par: float, e_par: float,
                f_par: float, f_d: float) -> np.ndarray:
    """
    스웰링 (부풀음) 열화 모델
    
    원본: BatteryDataTool.py swellingfit 함수
    
    수식:
        swelling = exp(a*T + b) * (cycle*fd)^b1 + exp(c*T + d) * (cycle*fd)^(e*T + f)
    
    Args:
        x: (cycle, temperature) 튜플
        a_par ~ f_d: 피팅 파라미터
        
    Returns:
        스웰링 비율
    """
    cycle = x[0]
    temperature = x[1]
    
    term1 = np.exp(a_par * temperature + b_par) * (cycle * f_d) ** b1_par
    term2 = np.exp(c_par * temperature + d_par) * (cycle * f_d) ** (e_par * temperature + f_par)
    
    return term1 + term2





# ============================================================
# 모델 클래스
# ============================================================

@dataclass
class ModelParameters:
    """용량 열화 모델 파라미터"""
    a: float = 0.03
    b: float = -18.0
    b1: float = 0.7
    c: float = 2.3
    d: float = -782.0
    e: float = -0.28
    f: float = 96.0
    fd: float = 1.0
    
    def to_array(self) -> np.ndarray:
        """배열로 변환 (curve_fit용)"""
        return np.array([self.a, self.b, self.b1, self.c, self.d, self.e, self.f, self.fd])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ModelParameters':
        """배열에서 생성"""
        return cls(
            a=arr[0], b=arr[1], b1=arr[2],
            c=arr[3], d=arr[4], e=arr[5],
            f=arr[6], fd=arr[7]
        )
    
    @classmethod
    def default(cls) -> 'ModelParameters':
        """기본 파라미터 (EU 기준 초기값)"""
        return cls(
            a=0.03, b=-18, b1=0.7,
            c=2.3, d=-782, e=-0.28,
            f=96, fd=1
        )


@dataclass
class FittingResult:
    """피팅 결과"""
    parameters: ModelParameters
    rmse: float
    r_squared: float
    covariance: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'a': self.parameters.a,
            'b': self.parameters.b,
            'b1': self.parameters.b1,
            'c': self.parameters.c,
            'd': self.parameters.d,
            'e': self.parameters.e,
            'f': self.parameters.f,
            'fd': self.parameters.fd,
            'rmse': self.rmse,
            'r_squared': self.r_squared
        }


class CapacityDegradationModel:
    """
    용량 열화 모델 클래스
    
    사용 예시:
        >>> model = CapacityDegradationModel()
        >>> result = model.fit(cycles, temperatures, capacities)
        >>> predicted = model.predict(new_cycles, new_temperatures)
    """
    
    def __init__(self, 
                 initial_params: Optional[ModelParameters] = None):
        """
        Args:
            initial_params: 초기 파라미터
        """
        self.params = initial_params or ModelParameters.default()
        self._is_fitted = False
        self.fitting_result: Optional[FittingResult] = None
    
    def _capacity_func(self, x, a, b, b1, c, d, e, f, fd):
        """피팅용 래퍼 함수"""
        return capacityfit(x, a, b, b1, c, d, e, f, fd)
    
    def fit(self,
            cycles: np.ndarray,
            temperatures: np.ndarray,
            capacities: np.ndarray,
            maxfev: int = 100000) -> FittingResult:
        """
        실험 데이터에 모델 피팅
        
        Args:
            cycles: 사이클 수 배열
            temperatures: 온도 배열 (K)
            capacities: 용량 비율 배열 (0-1)
            maxfev: 최대 함수 호출 횟수
            
        Returns:
            FittingResult 객체
        """
        cycles = np.asarray(cycles)
        temperatures = np.asarray(temperatures)
        capacities = np.asarray(capacities)
        
        # 초기값
        p0 = self.params.to_array()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    self._capacity_func,
                    (cycles, temperatures),
                    capacities,
                    p0=p0,
                    maxfev=maxfev
                )
            
            # 파라미터 업데이트
            self.params = ModelParameters.from_array(popt)
            
            # 예측 및 성능 지표 계산
            predicted = self._capacity_func((cycles, temperatures), *popt)
            residuals = capacities - predicted
            
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((capacities - np.mean(capacities)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean(residuals ** 2))
            
            self.fitting_result = FittingResult(
                parameters=self.params,
                rmse=rmse,
                r_squared=r_squared,
                covariance=pcov,
                residuals=residuals
            )
            
            self._is_fitted = True
            
            return self.fitting_result
            
        except Exception as e:
            raise RuntimeError(f"Fitting failed: {str(e)}")
    
    def predict(self,
                cycles: np.ndarray,
                temperatures: np.ndarray) -> np.ndarray:
        """
        용량 예측
        
        Args:
            cycles: 예측할 사이클 수
            temperatures: 예측할 온도 (K)
            
        Returns:
            예측 용량 비율
        """
        cycles = np.asarray(cycles)
        temperatures = np.asarray(temperatures)
        
        return capacityfit(
            (cycles, temperatures),
            self.params.a, self.params.b, self.params.b1,
            self.params.c, self.params.d, self.params.e,
            self.params.f, self.params.fd
        )
    
    def predict_cycle_to_eol(self,
                             temperature: float,
                             eol_threshold: float = 0.8,
                             max_cycles: int = 10000) -> int:
        """
        End-of-Life까지의 사이클 수 예측
        
        Args:
            temperature: 동작 온도 (K)
            eol_threshold: EOL 기준 (예: 0.8 = 80% 잔존)
            max_cycles: 최대 검색 사이클
            
        Returns:
            EOL까지의 사이클 수
        """
        cycles = np.arange(1, max_cycles + 1)
        temperatures = np.full_like(cycles, temperature, dtype=float)
        
        capacities = self.predict(cycles, temperatures)
        
        # EOL 도달 사이클 찾기
        below_eol = np.where(capacities < eol_threshold)[0]
        
        if len(below_eol) > 0:
            return int(cycles[below_eol[0]])
        else:
            return max_cycles  # EOL 미도달


def fit_capacity_model(cycles: np.ndarray,
                       temperatures: np.ndarray,
                       capacities: np.ndarray,
                       initial_params: Optional[ModelParameters] = None) -> FittingResult:
    """
    용량 열화 모델 피팅 헬퍼 함수
    
    Args:
        cycles: 사이클 수 배열
        temperatures: 온도 배열 (K)
        capacities: 용량 비율 배열
        initial_params: 초기 파라미터
        
    Returns:
        FittingResult 객체
    """
    model = CapacityDegradationModel(
        initial_params=initial_params
    )
    return model.fit(cycles, temperatures, capacities)



