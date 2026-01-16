"""
Approval Life Prediction Module
================================

승인 수명 예측 모듈입니다.
가속 시험 데이터를 기반으로 실사용 조건에서의 수명을 예측합니다.

핵심 개념:
- 가속 계수 (Acceleration Factor)
- Arrhenius 온도 의존성
- 다중 조건 외삽

Author: Battery Analysis Team
Date: 2026-01-14
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from scipy.optimize import curve_fit

from .capacity_fit import (
    CapacityDegradationModel,
    ModelParameters,
    FittingResult,
    capacityfit
)


@dataclass
class AccelerationFactor:
    """가속 계수 정보"""
    test_temperature: float    # 시험 온도 (K)
    target_temperature: float  # 목표 온도 (K)
    factor: float             # 가속 계수
    activation_energy: float  # 활성화 에너지 (J/mol)


@dataclass
class ApprovalResult:
    """승인 수명 예측 결과"""
    test_cycles: int              # 시험 사이클 수
    equivalent_real_cycles: int   # 실사용 환산 사이클 수
    predicted_years: float        # 예상 수명 (년)
    acceleration_factor: float    # 적용된 가속 계수
    meets_requirement: bool       # 승인 기준 충족 여부
    required_years: float         # 요구 수명 (년)
    model_parameters: ModelParameters
    confidence_interval: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict:
        return {
            'test_cycles': self.test_cycles,
            'equivalent_real_cycles': self.equivalent_real_cycles,
            'predicted_years': self.predicted_years,
            'acceleration_factor': self.acceleration_factor,
            'meets_requirement': self.meets_requirement,
            'required_years': self.required_years,
            **self.model_parameters.__dict__
        }


class ApprovalLifePredictor:
    """
    승인 수명 예측기
    
    가속 시험 결과를 실사용 조건으로 외삽하여 수명 예측
    
    사용 예시:
        >>> predictor = ApprovalLifePredictor(
        ...     test_temperature=318.15,  # 45°C
        ...     target_temperature=298.15  # 25°C
        ... )
        >>> result = predictor.fit_and_predict(cycles, temps, capacities)
    """
    
    # 물리 상수
    R = 8.314  # 기체 상수 (J/mol·K)
    
    def __init__(self,
                 test_temperature: float,
                 target_temperature: float = 298.15,
                 cycles_per_year: int = 365,  # 하루 1사이클 가정
                 required_years: float = 8.0):
        """
        Args:
            test_temperature: 시험 온도 (K)
            target_temperature: 목표 운용 온도 (K)
            cycles_per_year: 연간 예상 사이클 수
            required_years: 승인 요구 수명 (년)
        """
        self.test_temperature = test_temperature
        self.target_temperature = target_temperature
        self.cycles_per_year = cycles_per_year
        self.required_years = required_years
        
        self.model = CapacityDegradationModel()
        self._is_fitted = False
        self._acceleration_factor: Optional[float] = None
    
    def calculate_arrhenius_factor(self,
                                   activation_energy: float = 50000) -> float:
        """
        Arrhenius 기반 가속 계수 계산
        
        수식: AF = exp(Ea/R * (1/T_target - 1/T_test))
        
        Args:
            activation_energy: 활성화 에너지 (J/mol), 기본값 50 kJ/mol
            
        Returns:
            가속 계수
        """
        af = np.exp(
            activation_energy / self.R * (
                1 / self.target_temperature - 1 / self.test_temperature
            )
        )
        return af
    
    def fit(self,
            cycles: np.ndarray,
            temperatures: np.ndarray,
            capacities: np.ndarray,
            maxfev: int = 100000) -> FittingResult:
        """
        실험 데이터에 모델 피팅
        """
        result = self.model.fit(cycles, temperatures, capacities, maxfev)
        self._is_fitted = True
        return result
    
    def predict_real_life(self,
                          test_cycles: int,
                          test_soh: float,
                          activation_energy: float = 50000) -> Tuple[int, float]:
        """
        실사용 조건에서의 수명 예측
        
        Args:
            test_cycles: 시험 완료 사이클 수
            test_soh: 시험 완료 시점 SOH
            activation_energy: 활성화 에너지 (J/mol)
            
        Returns:
            (실사용 환산 사이클 수, 예상 수명 년수) 튜플
        """
        # 가속 계수 계산
        af = self.calculate_arrhenius_factor(activation_energy)
        self._acceleration_factor = af
        
        # 실사용 환산 사이클
        equivalent_cycles = int(test_cycles * af)
        
        # 년수 환산
        years = equivalent_cycles / self.cycles_per_year
        
        return equivalent_cycles, years
    
    def fit_and_predict(self,
                        cycles: np.ndarray,
                        temperatures: np.ndarray,
                        capacities: np.ndarray,
                        eol_threshold: float = 0.8,
                        activation_energy: float = 50000,
                        maxfev: int = 100000) -> ApprovalResult:
        """
        피팅 및 승인 수명 예측
        
        Args:
            cycles: 사이클 수 배열
            temperatures: 온도 배열 (K)
            capacities: 용량 비율 배열
            eol_threshold: EOL 기준 (0.8 = 80%)
            activation_energy: 활성화 에너지
            maxfev: 최대 함수 호출 횟수
            
        Returns:
            ApprovalResult 객체
        """
        # 피팅
        fitting_result = self.fit(cycles, temperatures, capacities, maxfev)
        
        # 시험 조건에서 EOL까지 사이클 예측
        test_cycles_to_eol = self.model.predict_cycle_to_eol(
            temperature=self.test_temperature,
            eol_threshold=eol_threshold
        )
        
        # 실사용 조건 환산
        equivalent_cycles, years = self.predict_real_life(
            test_cycles=test_cycles_to_eol,
            test_soh=eol_threshold,
            activation_energy=activation_energy
        )
        
        # 승인 기준 충족 여부
        meets_requirement = years >= self.required_years
        
        return ApprovalResult(
            test_cycles=test_cycles_to_eol,
            equivalent_real_cycles=equivalent_cycles,
            predicted_years=years,
            acceleration_factor=self._acceleration_factor,
            meets_requirement=meets_requirement,
            required_years=self.required_years,
            model_parameters=self.model.params
        )
    
    def sensitivity_analysis(self,
                             activation_energy_range: Tuple[float, float] = (40000, 80000),
                             n_points: int = 10) -> pd.DataFrame:
        """
        활성화 에너지에 대한 민감도 분석
        
        Args:
            activation_energy_range: 활성화 에너지 범위 (J/mol)
            n_points: 분석 포인트 수
            
        Returns:
            민감도 분석 결과 DataFrame
        """
        results = []
        
        for ea in np.linspace(activation_energy_range[0], 
                              activation_energy_range[1], 
                              n_points):
            af = self.calculate_arrhenius_factor(ea)
            results.append({
                'activation_energy_kJ': ea / 1000,
                'acceleration_factor': af,
                'temperature_test_C': self.test_temperature - 273.15,
                'temperature_target_C': self.target_temperature - 273.15
            })
        
        return pd.DataFrame(results)


def predict_approval_cycle_life(cycles: np.ndarray,
                                 temperatures: np.ndarray,
                                 capacities: np.ndarray,
                                 test_temperature: float,
                                 target_temperature: float = 298.15,
                                 required_years: float = 8.0) -> ApprovalResult:
    """
    승인 사이클 수명 예측 헬퍼 함수
    
    Args:
        cycles: 사이클 수 배열
        temperatures: 온도 배열 (K)
        capacities: 용량 비율 배열
        test_temperature: 시험 온도 (K)
        target_temperature: 목표 운용 온도 (K)
        required_years: 요구 수명 (년)
        
    Returns:
        ApprovalResult 객체
    """
    predictor = ApprovalLifePredictor(
        test_temperature=test_temperature,
        target_temperature=target_temperature,
        required_years=required_years
    )
    return predictor.fit_and_predict(
        cycles=cycles,
        temperatures=temperatures,
        capacities=capacities
    )
