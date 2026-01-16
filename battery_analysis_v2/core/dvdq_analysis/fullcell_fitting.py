"""
Full-Cell Fitting Module
========================

Full-cell 전압 시뮬레이션 및 dV/dQ 계산 모듈입니다.

원본 함수: BatteryDataTool.py - generate_simulation_full()

핵심 수식:
    V_fullcell(Q) = V_cathode(Q + slip_ca) * mass_ca - V_anode(Q + slip_an) * mass_an
    dV/dQ = d(V_fullcell)/dQ

Author: Battery Analysis Team
Date: 2026-01-14
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
from dataclasses import dataclass



from .halfcell_profiles import HalfCellProfile


@dataclass
class FittingResult:
    """피팅 결과 데이터 구조"""
    ca_mass: float
    ca_slip: float
    an_mass: float
    an_slip: float
    rmse: float
    r_squared: float
    simulated_voltage: np.ndarray
    simulated_dvdq: np.ndarray


def calculate_dvdq(voltage: np.ndarray, 
                   capacity: np.ndarray, 
                   period: int = 1) -> np.ndarray:
    """
    dV/dQ 계산 (미분 전압)
    
    Args:
        voltage: 전압 데이터 (V)
        capacity: 용량 데이터 (mAh 또는 %)
        period: 미분 윈도우 크기
        
    Returns:
        dV/dQ 배열
    """
    dv = np.diff(voltage, n=period, prepend=np.nan * np.ones(period))
    dq = np.diff(capacity, n=period, prepend=np.nan * np.ones(period))
    
    # 0으로 나누기 방지
    with np.errstate(divide='ignore', invalid='ignore'):
        dvdq = dv / dq
        dvdq[~np.isfinite(dvdq)] = np.nan
    
    return dvdq


def generate_simulation_full(
    ca_capacity: np.ndarray,
    ca_voltage: np.ndarray,
    an_capacity: np.ndarray,
    an_voltage: np.ndarray,
    real_capacity: np.ndarray,
    real_voltage: np.ndarray,
    ca_mass: float,
    ca_slip: float,
    an_mass: float,
    an_slip: float,
    full_cell_max_cap: float,
    rated_cap: float,
    full_period: int = 1
) -> pd.DataFrame:
    """
    Full-cell 시뮬레이션 결과 생성
    
    원본: BatteryDataTool.py generate_simulation_full() 함수
    
    열화 모델:
        ca_cap_new = ca_cap * ca_mass - ca_slip
        an_cap_new = an_cap * an_mass - an_slip
        V_full = V_cathode(Q) - V_anode(Q)
    
    Args:
        ca_capacity: 양극 용량 데이터
        ca_voltage: 양극 전압 데이터
        an_capacity: 음극 용량 데이터
        an_voltage: 음극 전압 데이터
        real_capacity: 실측 용량 데이터
        real_voltage: 실측 전압 데이터
        ca_mass: 양극 활물질 비율 (0-1)
        ca_slip: 양극 SOC 슬립
        an_mass: 음극 활물질 비율 (0-1)
        an_slip: 음극 SOC 슬립
        full_cell_max_cap: 풀셀 최대 용량
        rated_cap: 정격 용량
        full_period: dV/dQ 계산 주기
        
    Returns:
        시뮬레이션 결과 DataFrame
    """
    # 용량 보정 적용
    ca_cap_new = ca_capacity * ca_mass - ca_slip
    an_cap_new = an_capacity * an_mass - an_slip
    
    # 기준 용량 생성 (0.1 mAh 단위)
    simul_full_cap = np.arange(0, full_cell_max_cap, 0.1)
    
    # 각 전극 전압 보간
    simul_full_ca_volt = np.interp(simul_full_cap, ca_cap_new, ca_voltage)
    simul_full_an_volt = np.interp(simul_full_cap, an_cap_new, an_voltage)
    simul_full_real_volt = np.interp(simul_full_cap, real_capacity, real_voltage)
    
    # Full-cell 전압 계산
    simul_full_volt = simul_full_ca_volt - simul_full_an_volt
    
    # DataFrame 생성
    simul_full = pd.DataFrame({
        "full_cap": simul_full_cap,
        "an_volt": simul_full_an_volt,
        "ca_volt": simul_full_ca_volt,
        "full_volt": simul_full_volt,
        "real_volt": simul_full_real_volt
    })
    
    # 마지막 행 제거 (경계 조건)
    simul_full = simul_full.iloc[:-1]
    
    # 백분율로 용량 변환
    simul_full["full_cap"] = simul_full["full_cap"] / rated_cap * 100
    
    # dV/dQ 계산
    simul_full["an_dvdq"] = calculate_dvdq(
        simul_full["an_volt"].values, 
        simul_full["full_cap"].values, 
        full_period
    )
    simul_full["ca_dvdq"] = calculate_dvdq(
        simul_full["ca_volt"].values,
        simul_full["full_cap"].values,
        full_period
    )
    simul_full["real_dvdq"] = calculate_dvdq(
        simul_full["real_volt"].values,
        simul_full["full_cap"].values,
        full_period
    )
    simul_full["full_dvdq"] = simul_full["ca_dvdq"] - simul_full["an_dvdq"]
    
    return simul_full


class FullCellSimulator:
    """
    Full-cell 시뮬레이션 클래스
    
    사용 예시:
        >>> cathode = HalfCellProfile(...)
        >>> anode = HalfCellProfile(...)
        >>> simulator = FullCellSimulator(cathode, anode)
        >>> result = simulator.simulate(ca_mass=0.95, an_mass=0.98, ...)
    """
    
    def __init__(self, 
                 cathode_profile: HalfCellProfile,
                 anode_profile: HalfCellProfile,
                 rated_capacity: float = 100.0):
        """
        Args:
            cathode_profile: 양극 OCV 프로파일
            anode_profile: 음극 OCV 프로파일
            rated_capacity: 정격 용량 (mAh)
        """
        self.cathode = cathode_profile
        self.anode = anode_profile
        self.rated_capacity = rated_capacity
    
    def simulate(self,
                 ca_mass: float = 1.0,
                 ca_slip: float = 0.0,
                 an_mass: float = 1.0,
                 an_slip: float = 0.0,
                 capacity_step: float = 0.1,
                 dvdq_period: int = 1) -> pd.DataFrame:
        """
        Full-cell 전압 시뮬레이션 수행
        
        Args:
            ca_mass: 양극 활물질 비율
            ca_slip: 양극 SOC 슬립
            an_mass: 음극 활물질 비율
            an_slip: 음극 SOC 슬립
            capacity_step: 용량 스텝 크기
            dvdq_period: dV/dQ 미분 주기
            
        Returns:
            시뮬레이션 결과 DataFrame
        """
        # 열화 파라미터 적용
        ca_cap_new, ca_volt = self.cathode.apply_degradation(ca_mass, ca_slip)
        an_cap_new, an_volt = self.anode.apply_degradation(an_mass, an_slip)
        
        # 최대 용량 결정
        max_cap = min(np.max(ca_cap_new), np.max(an_cap_new))
        
        # 시뮬레이션 용량 범위
        sim_capacity = np.arange(0, max_cap, capacity_step)
        
        # 전압 보간
        sim_ca_volt = np.interp(sim_capacity, ca_cap_new, ca_volt)
        sim_an_volt = np.interp(sim_capacity, an_cap_new, an_volt)
        
        # Full-cell 전압
        sim_full_volt = sim_ca_volt - sim_an_volt
        
        # SOC 계산 (백분율)
        sim_soc = sim_capacity / self.rated_capacity * 100
        
        # DataFrame 생성
        result = pd.DataFrame({
            'capacity': sim_capacity,
            'soc': sim_soc,
            'ca_voltage': sim_ca_volt,
            'an_voltage': sim_an_volt,
            'full_voltage': sim_full_volt
        })
        
        # dV/dQ 계산
        result['ca_dvdq'] = calculate_dvdq(sim_ca_volt, sim_soc, dvdq_period)
        result['an_dvdq'] = calculate_dvdq(sim_an_volt, sim_soc, dvdq_period)
        result['full_dvdq'] = calculate_dvdq(sim_full_volt, sim_soc, dvdq_period)
        
        return result
    
    def calculate_rmse(self,
                       experimental_voltage: np.ndarray,
                       experimental_capacity: np.ndarray,
                       ca_mass: float,
                       ca_slip: float,
                       an_mass: float,
                       an_slip: float) -> float:
        """
        시뮬레이션과 실험 데이터 간 RMSE 계산
        
        Args:
            experimental_voltage: 실험 전압 데이터
            experimental_capacity: 실험 용량 데이터
            ca_mass, ca_slip, an_mass, an_slip: 열화 파라미터
            
        Returns:
            RMSE 값
        """
        # 시뮬레이션 수행
        sim_result = self.simulate(ca_mass, ca_slip, an_mass, an_slip)
        
        # 실험 용량 범위에서 시뮬레이션 전압 보간
        sim_voltage_interp = np.interp(
            experimental_capacity,
            sim_result['capacity'].values,
            sim_result['full_voltage'].values
        )
        
        # RMSE 계산
        rmse = np.sqrt(np.mean((experimental_voltage - sim_voltage_interp) ** 2))
        
        return rmse



