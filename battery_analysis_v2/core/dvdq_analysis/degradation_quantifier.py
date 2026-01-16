"""
Degradation Quantifier Module
=============================

dV/dQ 분석을 통한 배터리 열화 정량화 모듈입니다.

열화 메커니즘:
- LAM_PE (Loss of Active Material - Positive Electrode): 양극 활물질 손실
- LAM_NE (Loss of Active Material - Negative Electrode): 음극 활물질 손실  
- LLI (Loss of Lithium Inventory): 리튬 재고 손실

정량화 수식:
    LAM_PE = (1 - mass_ca) * 100%
    LAM_NE = (1 - mass_an) * 100%
    LLI = f(slip_ca, slip_an)  # 슬립 파라미터로부터 계산

Author: Battery Analysis Team
Date: 2026-01-14
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from scipy.optimize import minimize, curve_fit




class DegradationMetrics(NamedTuple):
    """열화 지표 결과"""
    lam_pe: float  # 양극 활물질 손실 (%)
    lam_ne: float  # 음극 활물질 손실 (%)
    lli: float     # 리튬 재고 손실 (%)
    capacity_loss: float  # 총 용량 손실 (%)


@dataclass
class FittingParams:
    """피팅 파라미터"""
    ca_mass: float = 1.0  # 양극 활물질 비율 (0-1)
    ca_slip: float = 0.0  # 양극 SOC 슬립
    an_mass: float = 1.0  # 음극 활물질 비율 (0-1)
    an_slip: float = 0.0  # 음극 SOC 슬립
    
    def to_array(self) -> np.ndarray:
        """배열로 변환"""
        return np.array([self.ca_mass, self.ca_slip, self.an_mass, self.an_slip])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'FittingParams':
        """배열에서 생성"""
        return cls(
            ca_mass=arr[0],
            ca_slip=arr[1],
            an_mass=arr[2],
            an_slip=arr[3]
        )


def calculate_lam(mass_ratio: float, electrode: str = 'cathode') -> float:
    """
    Loss of Active Material (LAM) 계산
    
    수식: LAM = (1 - mass_ratio) * 100%
    
    Args:
        mass_ratio: 활물질 비율 (0-1, 예: 0.95 = 95% 잔존)
        electrode: 'cathode' 또는 'anode'
        
    Returns:
        LAM 값 (%)
        
    Examples:
        >>> calculate_lam(0.95, 'cathode')  # 5% 양극 활물질 손실
        5.0
    """
    if not 0 <= mass_ratio <= 1:
        raise ValueError(f"mass_ratio must be between 0 and 1, got {mass_ratio}")
    
    lam = (1 - mass_ratio) * 100
    return lam


def calculate_lli(slip_ca: float, 
                  slip_an: float, 
                  rated_capacity: float = 100.0) -> float:
    """
    Loss of Lithium Inventory (LLI) 계산
    
    LLI는 양극과 음극의 slip 파라미터 차이로부터 추정됩니다.
    주로 SEI 성장에 의한 리튬 소모를 반영합니다.
    
    Args:
        slip_ca: 양극 SOC 슬립
        slip_an: 음극 SOC 슬립
        rated_capacity: 정격 용량
        
    Returns:
        LLI 값 (%)
    """
    # 슬립 차이를 리튬 손실로 해석
    lli = abs(slip_an - slip_ca) / rated_capacity * 100
    return lli


class DegradationQuantifier:
    """
    열화 정량화 클래스
    
    dV/dQ 피팅을 통해 열화 메커니즘을 분리/정량화합니다.
    
    사용 예시:
        >>> quantifier = DegradationQuantifier(cathode_profile, anode_profile)
        >>> metrics = quantifier.quantify(experimental_data)
        >>> print(f"LAM_PE: {metrics.lam_pe:.1f}%")
    """
    
    def __init__(self,
                 cathode_capacity: np.ndarray,
                 cathode_voltage: np.ndarray,
                 anode_capacity: np.ndarray,
                 anode_voltage: np.ndarray,
                 rated_capacity: float = 100.0):
        """
        Args:
            cathode_capacity: 양극 용량 데이터
            cathode_voltage: 양극 전압 데이터
            anode_capacity: 음극 용량 데이터
            anode_voltage: 음극 전압 데이터
            rated_capacity: 정격 용량
        """
        self.ca_capacity = np.asarray(cathode_capacity)
        self.ca_voltage = np.asarray(cathode_voltage)
        self.an_capacity = np.asarray(anode_capacity)
        self.an_voltage = np.asarray(anode_voltage)
        self.rated_capacity = rated_capacity
    
    def _simulate_voltage(self, 
                          params: FittingParams,
                          target_capacity: np.ndarray) -> np.ndarray:
        """주어진 파라미터로 전압 시뮬레이션"""
        # 열화 적용
        ca_cap_new = self.ca_capacity * params.ca_mass - params.ca_slip
        an_cap_new = self.an_capacity * params.an_mass - params.an_slip
        
        # 전압 보간
        ca_volt = np.interp(target_capacity, ca_cap_new, self.ca_voltage)
        an_volt = np.interp(target_capacity, an_cap_new, self.an_voltage)
        
        # Full-cell 전압
        full_volt = ca_volt - an_volt
        
        return full_volt
    
    def _objective(self, 
                   params_array: np.ndarray,
                   exp_capacity: np.ndarray,
                   exp_voltage: np.ndarray) -> float:
        """최적화 목적 함수 (RMSE)"""
        params = FittingParams.from_array(params_array)
        
        try:
            sim_voltage = self._simulate_voltage(params, exp_capacity)
            rmse = np.sqrt(np.nanmean((exp_voltage - sim_voltage) ** 2))
            return rmse
        except Exception:
            return np.inf
    
    def fit(self,
            experimental_capacity: np.ndarray,
            experimental_voltage: np.ndarray,
            initial_params: Optional[FittingParams] = None,
            bounds: Optional[Dict] = None) -> Tuple[FittingParams, float]:
        """
        실험 데이터에 피팅하여 열화 파라미터 추출
        
        Args:
            experimental_capacity: 실험 용량 데이터
            experimental_voltage: 실험 전압 데이터
            initial_params: 초기 파라미터
            bounds: 파라미터 범위 {'ca_mass': (min, max), ...}
            
        Returns:
            (최적 파라미터, RMSE) 튜플
        """
        # 초기값 설정
        if initial_params is None:
            initial_params = FittingParams()
        
        # 범위 설정
        if bounds is None:
            bounds_array = [
                (0.8, 1.0),   # ca_mass
                (-5, 5),      # ca_slip
                (0.8, 1.0),   # an_mass
                (-5, 5),      # an_slip
            ]
        else:
            bounds_array = [
                bounds.get('ca_mass', (0.8, 1.0)),
                bounds.get('ca_slip', (-5, 5)),
                bounds.get('an_mass', (0.8, 1.0)),
                bounds.get('an_slip', (-5, 5)),
            ]
        
        # 최적화
        result = minimize(
            self._objective,
            initial_params.to_array(),
            args=(experimental_capacity, experimental_voltage),
            method='L-BFGS-B',
            bounds=bounds_array
        )
        
        optimal_params = FittingParams.from_array(result.x)
        rmse = result.fun
        
        return optimal_params, rmse
    
    def quantify(self,
                 experimental_capacity: np.ndarray,
                 experimental_voltage: np.ndarray,
                 initial_params: Optional[FittingParams] = None) -> DegradationMetrics:
        """
        열화 정량화 수행
        
        Args:
            experimental_capacity: 실험 용량 데이터
            experimental_voltage: 실험 전압 데이터
            initial_params: 초기 파라미터
            
        Returns:
            DegradationMetrics 객체
        """
        # 피팅 수행
        params, rmse = self.fit(
            experimental_capacity, 
            experimental_voltage,
            initial_params
        )
        
        # 열화 지표 계산
        lam_pe = calculate_lam(params.ca_mass, 'cathode')
        lam_ne = calculate_lam(params.an_mass, 'anode')
        lli = calculate_lli(params.ca_slip, params.an_slip, self.rated_capacity)
        
        # 총 용량 손실 계산 (시뮬레이션 기반)
        initial_capacity = np.max(self.ca_capacity)
        capacity_with_degradation = initial_capacity * min(params.ca_mass, params.an_mass)
        capacity_loss = (1 - capacity_with_degradation / initial_capacity) * 100
        
        return DegradationMetrics(
            lam_pe=lam_pe,
            lam_ne=lam_ne,
            lli=lli,
            capacity_loss=capacity_loss
        )
    
    def generate_random_params(self,
                               ca_mass_range: Tuple[float, float] = (0.9, 1.0),
                               ca_slip_range: Tuple[float, float] = (-2, 2),
                               an_mass_range: Tuple[float, float] = (0.9, 1.0),
                               an_slip_range: Tuple[float, float] = (-2, 2),
                               seed: Optional[int] = None) -> FittingParams:
        """
        랜덤 파라미터 생성 (Monte Carlo 시뮬레이션용)
        
        원본: BatteryDataTool.py generate_params() 함수
        
        Args:
            ca_mass_range: 양극 mass 범위
            ca_slip_range: 양극 slip 범위
            an_mass_range: 음극 mass 범위
            an_slip_range: 음극 slip 범위
            seed: 랜덤 시드
            
        Returns:
            랜덤 생성된 FittingParams
        """
        if seed is not None:
            np.random.seed(seed)
        
        return FittingParams(
            ca_mass=np.random.uniform(*ca_mass_range),
            ca_slip=np.random.uniform(*ca_slip_range),
            an_mass=np.random.uniform(*an_mass_range),
            an_slip=np.random.uniform(*an_slip_range)
        )


# ============================================================
# 헬퍼 함수
# ============================================================

def analyze_degradation_trend(cycles: np.ndarray,
                              lam_pe_values: np.ndarray,
                              lam_ne_values: np.ndarray,
                              lli_values: np.ndarray) -> Dict[str, float]:
    """
    열화 트렌드 분석
    
    Args:
        cycles: 사이클 번호 배열
        lam_pe_values: 각 사이클의 LAM_PE 값
        lam_ne_values: 각 사이클의 LAM_NE 값
        lli_values: 각 사이클의 LLI 값
        
    Returns:
        각 열화 메커니즘의 기울기 (cycle당 변화율)
    """
    from scipy.stats import linregress
    
    results = {}
    
    # LAM_PE 트렌드
    slope, intercept, r, p, se = linregress(cycles, lam_pe_values)
    results['lam_pe_rate'] = slope
    results['lam_pe_r2'] = r ** 2
    
    # LAM_NE 트렌드
    slope, intercept, r, p, se = linregress(cycles, lam_ne_values)
    results['lam_ne_rate'] = slope
    results['lam_ne_r2'] = r ** 2
    
    # LLI 트렌드
    slope, intercept, r, p, se = linregress(cycles, lli_values)
    results['lli_rate'] = slope
    results['lli_r2'] = r ** 2
    
    return results
