"""
EU Life Prediction Module
==========================

EU 기준 사이클 수명 예측 모듈입니다.

EU 규정 기반 배터리 수명 요구사항을 충족하는지 평가합니다.

Author: Battery Analysis Team
Date: 2026-01-14
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from .capacity_fit import (
    CapacityDegradationModel,
    ModelParameters,
    FittingResult,
    capacityfit,
    capacityfit_numba,
    NUMBA_AVAILABLE
)


@dataclass
class EULifeResult:
    """EU 수명 예측 결과"""
    predicted_cycles_80: int      # 80% SOH까지의 사이클
    predicted_cycles_70: int      # 70% SOH까지의 사이클
    meets_eu_requirement: bool    # EU 기준 충족 여부
    required_cycles: int          # EU 요구 사이클 수
    model_parameters: ModelParameters
    rmse: float
    r_squared: float
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'predicted_cycles_80': self.predicted_cycles_80,
            'predicted_cycles_70': self.predicted_cycles_70,
            'meets_eu_requirement': self.meets_eu_requirement,
            'required_cycles': self.required_cycles,
            'rmse': self.rmse,
            'r_squared': self.r_squared,
            **self.model_parameters.__dict__
        }


class EULifePredictor:
    """
    EU 기준 수명 예측기
    
    EU 규정:
    - EV 배터리: 1000 사이클에서 70~80% SOH 유지
    - ESS 배터리: 더 긴 수명 요구
    
    사용 예시:
        >>> predictor = EULifePredictor()
        >>> result = predictor.fit_and_predict(cycles, temps, capacities)
        >>> print(f"80% SOH 도달: {result.predicted_cycles_80} cycles")
    """
    
    def __init__(self,
                 required_cycles: int = 1000,
                 eol_threshold: float = 0.8,
                 use_numba: bool = True):
        """
        Args:
            required_cycles: EU 요구 사이클 수
            eol_threshold: EOL 기준 (0.8 = 80%)
            use_numba: Numba 최적화 사용
        """
        self.required_cycles = required_cycles
        self.eol_threshold = eol_threshold
        self.use_numba = use_numba and NUMBA_AVAILABLE
        
        self.model = CapacityDegradationModel(use_numba=self.use_numba)
        self._is_fitted = False
    
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
            FittingResult
        """
        result = self.model.fit(cycles, temperatures, capacities, maxfev)
        self._is_fitted = True
        return result
    
    def predict_cycles_to_soh(self,
                              temperature: float,
                              target_soh: float = 0.8,
                              max_cycles: int = 10000) -> int:
        """
        특정 SOH까지의 사이클 수 예측
        
        Args:
            temperature: 동작 온도 (K)
            target_soh: 목표 SOH (예: 0.8)
            max_cycles: 최대 예측 사이클
            
        Returns:
            목표 SOH 도달 사이클 수
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first. Call fit() method.")
        
        return self.model.predict_cycle_to_eol(
            temperature=temperature,
            eol_threshold=target_soh,
            max_cycles=max_cycles
        )
    
    def fit_and_predict(self,
                        cycles: np.ndarray,
                        temperatures: np.ndarray,
                        capacities: np.ndarray,
                        operating_temperature: float = 298.15,  # 25°C
                        maxfev: int = 100000) -> EULifeResult:
        """
        피팅 및 EU 기준 예측 수행
        
        Args:
            cycles: 사이클 수 배열
            temperatures: 온도 배열 (K)
            capacities: 용량 비율 배열
            operating_temperature: 운용 온도 (K)
            maxfev: 최대 함수 호출 횟수
            
        Returns:
            EULifeResult 객체
        """
        # 피팅
        fitting_result = self.fit(cycles, temperatures, capacities, maxfev)
        
        # 80% SOH 도달 사이클 예측
        cycles_to_80 = self.predict_cycles_to_soh(
            temperature=operating_temperature,
            target_soh=0.8
        )
        
        # 70% SOH 도달 사이클 예측
        cycles_to_70 = self.predict_cycles_to_soh(
            temperature=operating_temperature,
            target_soh=0.7
        )
        
        # EU 기준 충족 여부
        meets_requirement = cycles_to_80 >= self.required_cycles
        
        return EULifeResult(
            predicted_cycles_80=cycles_to_80,
            predicted_cycles_70=cycles_to_70,
            meets_eu_requirement=meets_requirement,
            required_cycles=self.required_cycles,
            model_parameters=self.model.params,
            rmse=fitting_result.rmse,
            r_squared=fitting_result.r_squared
        )
    
    def predict_capacity_curve(self,
                               temperature: float,
                               max_cycles: int = 2000,
                               step: int = 10) -> pd.DataFrame:
        """
        용량 열화 곡선 생성
        
        Args:
            temperature: 온도 (K)
            max_cycles: 최대 사이클
            step: 사이클 스텝
            
        Returns:
            사이클별 용량 DataFrame
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first. Call fit() method.")
        
        cycles = np.arange(0, max_cycles + 1, step)
        temps = np.full_like(cycles, temperature, dtype=float)
        
        capacities = self.model.predict(cycles, temps)
        
        return pd.DataFrame({
            'cycle': cycles,
            'capacity': capacities,
            'soh_percent': capacities * 100
        })
    
    def compare_temperatures(self,
                             temperatures: list,
                             max_cycles: int = 2000) -> pd.DataFrame:
        """
        다양한 온도에서의 열화 비교
        
        Args:
            temperatures: 온도 리스트 (K)
            max_cycles: 최대 사이클
            
        Returns:
            온도별 비교 DataFrame
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first. Call fit() method.")
        
        results = []
        cycles = np.arange(0, max_cycles + 1, 10)
        
        for temp in temperatures:
            temps = np.full_like(cycles, temp, dtype=float)
            capacities = self.model.predict(cycles, temps)
            
            for i, (cyc, cap) in enumerate(zip(cycles, capacities)):
                results.append({
                    'temperature_K': temp,
                    'temperature_C': temp - 273.15,
                    'cycle': cyc,
                    'capacity': cap
                })
        
        return pd.DataFrame(results)


def predict_eu_cycle_life(cycles: np.ndarray,
                          temperatures: np.ndarray,
                          capacities: np.ndarray,
                          operating_temperature: float = 298.15,
                          required_cycles: int = 1000) -> EULifeResult:
    """
    EU 사이클 수명 예측 헬퍼 함수
    
    Args:
        cycles: 사이클 수 배열
        temperatures: 온도 배열 (K)
        capacities: 용량 비율 배열
        operating_temperature: 운용 온도 (K)
        required_cycles: EU 요구 사이클 수
        
    Returns:
        EULifeResult 객체
    """
    predictor = EULifePredictor(required_cycles=required_cycles)
    return predictor.fit_and_predict(
        cycles=cycles,
        temperatures=temperatures,
        capacities=capacities,
        operating_temperature=operating_temperature
    )
