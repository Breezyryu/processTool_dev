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

try:
    from numba import njit, prange, float64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


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
# Numba 최적화 버전 (Julia 비교용 벤치마크)
# ============================================================

if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def capacityfit_numba(x_cycle: np.ndarray, 
                         x_temp: np.ndarray,
                         a: float, b: float, b1: float,
                         c: float, d: float, e: float, 
                         f: float, fd: float) -> np.ndarray:
        """
        Numba 최적화된 용량 열화 모델
        
        병렬 처리 및 fastmath 옵션으로 최대 성능 달성
        
        Args:
            x_cycle: 사이클 수 배열
            x_temp: 온도 배열 (K)
            a,b,b1,c,d,e,f,fd: 피팅 파라미터
            
        Returns:
            잔존 용량 비율 배열
        """
        n = len(x_cycle)
        result = np.empty(n)
        
        for i in prange(n):
            cycle = x_cycle[i]
            temp = x_temp[i]
            
            # Calendar aging term
            term1 = np.exp(a * temp + b) * (cycle * fd) ** b1
            
            # Cycle aging term
            term2 = np.exp(c * temp + d) * (cycle * fd) ** (e * temp + f)
            
            result[i] = 1.0 - term1 - term2
        
        return result
    
    @njit(parallel=True, fastmath=True)
    def swellingfit_numba(x_cycle: np.ndarray,
                         x_temp: np.ndarray,
                         a: float, b: float, b1: float,
                         c: float, d: float, e: float,
                         f: float, fd: float) -> np.ndarray:
        """
        Numba 최적화된 스웰링 모델
        """
        n = len(x_cycle)
        result = np.empty(n)
        
        for i in prange(n):
            cycle = x_cycle[i]
            temp = x_temp[i]
            
            term1 = np.exp(a * temp + b) * (cycle * fd) ** b1
            term2 = np.exp(c * temp + d) * (cycle * fd) ** (e * temp + f)
            
            result[i] = term1 + term2
        
        return result
else:
    # Numba 미설치 시 순수 Python 폴백
    def capacityfit_numba(x_cycle, x_temp, a, b, b1, c, d, e, f, fd):
        """Numba 미설치 시 순수 Python 폴백"""
        return capacityfit((x_cycle, x_temp), a, b, b1, c, d, e, f, fd)
    
    def swellingfit_numba(x_cycle, x_temp, a, b, b1, c, d, e, f, fd):
        """Numba 미설치 시 순수 Python 폴백"""
        return swellingfit((x_cycle, x_temp), a, b, b1, c, d, e, f, fd)


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
                 initial_params: Optional[ModelParameters] = None,
                 use_numba: bool = True):
        """
        Args:
            initial_params: 초기 파라미터
            use_numba: Numba 최적화 사용 여부
        """
        self.params = initial_params or ModelParameters.default()
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self._is_fitted = False
        self.fitting_result: Optional[FittingResult] = None
    
    def _capacity_func(self, x, a, b, b1, c, d, e, f, fd):
        """피팅용 래퍼 함수"""
        if self.use_numba:
            return capacityfit_numba(x[0], x[1], a, b, b1, c, d, e, f, fd)
        else:
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
        
        if self.use_numba:
            return capacityfit_numba(
                cycles, temperatures,
                self.params.a, self.params.b, self.params.b1,
                self.params.c, self.params.d, self.params.e,
                self.params.f, self.params.fd
            )
        else:
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
                       initial_params: Optional[ModelParameters] = None,
                       use_numba: bool = True) -> FittingResult:
    """
    용량 열화 모델 피팅 헬퍼 함수
    
    Args:
        cycles: 사이클 수 배열
        temperatures: 온도 배열 (K)
        capacities: 용량 비율 배열
        initial_params: 초기 파라미터
        use_numba: Numba 사용 여부
        
    Returns:
        FittingResult 객체
    """
    model = CapacityDegradationModel(
        initial_params=initial_params,
        use_numba=use_numba
    )
    return model.fit(cycles, temperatures, capacities)


# ============================================================
# 벤치마크 유틸리티
# ============================================================

def benchmark_fitting(n_points: int = 10000,
                      n_iterations: int = 100) -> Dict[str, float]:
    """
    Python vs Numba 성능 벤치마크
    
    Args:
        n_points: 데이터 포인트 수
        n_iterations: 반복 횟수
        
    Returns:
        벤치마크 결과 딕셔너리
    """
    import time
    
    # 테스트 데이터 생성
    cycles = np.random.uniform(1, 1000, n_points)
    temps = np.random.uniform(273 + 25, 273 + 45, n_points)
    
    # 파라미터
    a, b, b1, c, d, e, f, fd = 0.03, -18, 0.7, 2.3, -782, -0.28, 96, 1
    
    # Python 버전 벤치마크
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = capacityfit((cycles, temps), a, b, b1, c, d, e, f, fd)
    python_time = (time.perf_counter() - start) / n_iterations
    
    # Numba 버전 벤치마크
    if NUMBA_AVAILABLE:
        # Warmup (JIT 컴파일)
        _ = capacityfit_numba(cycles, temps, a, b, b1, c, d, e, f, fd)
        
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = capacityfit_numba(cycles, temps, a, b, b1, c, d, e, f, fd)
        numba_time = (time.perf_counter() - start) / n_iterations
    else:
        numba_time = None
    
    results = {
        'n_points': n_points,
        'n_iterations': n_iterations,
        'python_time_ms': python_time * 1000,
        'numba_time_ms': numba_time * 1000 if numba_time else None,
        'speedup': python_time / numba_time if numba_time else None,
        'numba_available': NUMBA_AVAILABLE
    }
    
    return results
