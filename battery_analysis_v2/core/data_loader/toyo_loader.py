"""
Toyo Loader Module
==================

Toyo 충방전기 데이터 로더입니다.

데이터 구조:
- capacity.log: 사이클 요약 데이터
- 000001, 000002, ...: 각 사이클의 프로파일 데이터

원본 함수: BatteryDataTool.py toyo_cycle_data(), toyo_Profile_import()

Author: Battery Analysis Team
Date: 2026-01-14
"""

import os
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np

from .base_loader import (
    BaseCyclerLoader,
    CycleData,
    ProfileData,
    CyclerType,
)


class ToyoLoader(BaseCyclerLoader):
    """
    Toyo 충방전기 데이터 로더
    
    사용 예시:
        >>> loader = ToyoLoader("/path/to/data")
        >>> cycle_data = loader.load_cycle_data()
        >>> profile = loader.load_profile_data(cycle=1)
    """
    
    @property
    def cycler_type(self) -> CyclerType:
        return CyclerType.TOYO
    
    def _read_csv(self, *args) -> Optional[pd.DataFrame]:
        """
        Toyo CSV 파일 읽기
        
        원본 함수: toyo_read_csv()
        """
        if len(args) == 0:
            filepath = self.raw_path / "capacity.log"
            skiprows = 0
        else:
            cycle = args[0]
            filepath = self.raw_path / f"{cycle:06d}"
            skiprows = 3
        
        if filepath.exists():
            try:
                df = pd.read_csv(
                    filepath,
                    sep=",",
                    skiprows=skiprows,
                    engine="c",
                    encoding="cp949",
                    on_bad_lines='skip'
                )
                return df
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                return None
        return None
    
    def _estimate_capacity(self) -> float:
        """
        첫 사이클 데이터에서 용량 추정
        
        원본 함수: toyo_min_cap()
        """
        # 경로에서 추출 시도
        capacity = self._extract_capacity_from_path()
        if capacity > 0:
            return capacity
        
        # 첫 사이클 데이터에서 추정
        first_cycle = self._read_csv(1)
        if first_cycle is not None and "Current[mA]" in first_cycle.columns:
            max_current = first_cycle["Current[mA]"].max()
            if max_current > 0:
                return int(round(max_current / self.initial_crate))
        
        return 0.0
    
    def load_cycle_data(self, 
                        check_dcir: bool = False,
                        **kwargs) -> CycleData:
        """
        사이클 데이터 로드
        
        원본 함수: toyo_cycle_data()
        
        Args:
            check_dcir: DCIR 계산 여부
            
        Returns:
            CycleData 객체
        """
        # 용량 확인
        if self._rated_capacity == 0:
            self._rated_capacity = self._estimate_capacity()
        
        mincapacity = self._rated_capacity
        
        # capacity.log 로드
        raw_data = self._read_csv()
        if raw_data is None or raw_data.empty:
            raise FileNotFoundError(f"capacity.log not found in {self.raw_path}")
        
        # 컬럼 정규화
        if "Cap[mAh]" in raw_data.columns:
            col_mapping = {
                "TotlCycle": "TotlCycle",
                "Condition": "Condition",
                "Cap[mAh]": "Cap",
                "Ocv": "Ocv",
                "Finish": "Finish",
                "Mode": "Mode",
                "PeakVolt[V]": "PeakVolt",
                "Pow[mWh]": "Power",
                "PeakTemp[Deg]": "PeakTemp",
                "AveVolt[V]": "AveVolt"
            }
        else:
            col_mapping = {
                "Total Cycle": "TotlCycle",
                "Condition": "Condition",
                "Capacity[mAh]": "Cap",
                "OCV[V]": "Ocv",
                "End Factor": "Finish",
                "Mode": "Mode",
                "Peak Volt.[V]": "PeakVolt",
                "Power[mWh]": "Power",
                "Peak Temp.[deg]": "PeakTemp",
                "Ave. Volt.[V]": "AveVolt"
            }
        
        # 필요한 컬럼만 선택 및 이름 변경
        available_cols = [c for c in col_mapping.keys() if c in raw_data.columns]
        cycleraw = raw_data[available_cols].copy()
        cycleraw.columns = [col_mapping[c] for c in available_cols]
        
        # 원본 사이클 저장
        cycleraw["OriCycle"] = cycleraw["TotlCycle"].copy()
        
        # 연속된 동일 조건 병합
        cycleraw = self._merge_consecutive_conditions(cycleraw, mincapacity)
        
        # 충전 데이터 추출
        charge_mask = (
            (cycleraw["Condition"] == 1) & 
            (cycleraw["Cap"] > mincapacity / 60)
        )
        if "Finish" in cycleraw.columns:
            charge_mask &= ~cycleraw["Finish"].isin(["Vol", "Volt", "                 Vol"])
        
        charge_data = cycleraw[charge_mask].copy()
        
        # 방전 데이터 추출
        discharge_mask = (
            (cycleraw["Condition"] == 2) & 
            (cycleraw["Cap"] > mincapacity / 60)
        )
        discharge_data = cycleraw[discharge_mask].copy()
        
        # 인덱스를 사이클로 설정
        if not charge_data.empty:
            charge_data = charge_data.set_index("TotlCycle")
        if not discharge_data.empty:
            discharge_data = discharge_data.set_index("TotlCycle")
        
        # 공통 사이클 찾기
        common_cycles = np.intersect1d(
            charge_data.index.values if not charge_data.empty else [],
            discharge_data.index.values if not discharge_data.empty else []
        )
        
        if len(common_cycles) == 0:
            raise ValueError("No valid cycle data found")
        
        # 데이터 정렬
        charge_cap = charge_data.loc[common_cycles, "Cap"].values
        discharge_cap = discharge_data.loc[common_cycles, "Cap"].values
        
        # 효율 계산
        efficiency = np.divide(
            discharge_cap, charge_cap,
            out=np.ones_like(discharge_cap),
            where=charge_cap > 0
        )
        
        # 결과 생성
        result = CycleData(
            cycle=common_cycles,
            charge_capacity=charge_cap,
            discharge_capacity=discharge_cap,
            efficiency=efficiency,
            rated_capacity=mincapacity
        )
        
        # 옵션 데이터
        if "Power" in discharge_data.columns:
            result.discharge_energy = discharge_data.loc[common_cycles, "Power"].values
        if "PeakTemp" in discharge_data.columns:
            result.temperature = discharge_data.loc[common_cycles, "PeakTemp"].values
        if "Ocv" in charge_data.columns:
            result.rest_voltage = charge_data.loc[common_cycles, "Ocv"].values
        if "AveVolt" in discharge_data.columns:
            result.average_voltage = discharge_data.loc[common_cycles, "AveVolt"].values
        if "OriCycle" in discharge_data.columns:
            result.original_cycle = discharge_data.loc[common_cycles, "OriCycle"].values
        
        return result
    
    def _merge_consecutive_conditions(self, df: pd.DataFrame, 
                                       mincapacity: float) -> pd.DataFrame:
        """연속된 동일 조건 (충전/방전) 병합"""
        i = 0
        while i < len(df) - 1:
            current_cond = df.iloc[i]["Condition"]
            next_cond = df.iloc[i + 1]["Condition"]
            
            if current_cond in (1, 2) and current_cond == next_cond:
                idx = df.index[i + 1]
                prev_idx = df.index[i]
                
                # 용량 합산
                df.loc[idx, "Cap"] += df.loc[prev_idx, "Cap"]
                
                if current_cond == 1:
                    # 충전: OCV 유지
                    if "Ocv" in df.columns:
                        df.loc[idx, "Ocv"] = df.loc[prev_idx, "Ocv"]
                else:
                    # 방전: 에너지 합산, 평균 전압 재계산
                    if "Power" in df.columns:
                        df.loc[idx, "Power"] += df.loc[prev_idx, "Power"]
                    if "AveVolt" in df.columns and df.loc[idx, "Cap"] > 0:
                        df.loc[idx, "AveVolt"] = df.loc[idx, "Power"] / df.loc[idx, "Cap"]
                
                df = df.drop(prev_idx).reset_index(drop=True)
            else:
                i += 1
        
        return df
    
    def load_profile_data(self, cycle: int, **kwargs) -> ProfileData:
        """
        특정 사이클의 프로파일 데이터 로드
        
        원본 함수: toyo_Profile_import()
        
        Args:
            cycle: 사이클 번호
            
        Returns:
            ProfileData 객체
        """
        raw_data = self._read_csv(cycle)
        
        if raw_data is None or raw_data.empty:
            raise FileNotFoundError(f"Profile data for cycle {cycle} not found")
        
        # 컬럼 정규화
        if "PassTime[Sec]" in raw_data.columns:
            time_col = "PassTime[Sec]"
            temp_col = "Temp1[Deg]"
        else:
            time_col = "Passed Time[Sec]"
            temp_col = "Temp1[deg]"
        
        # 필수 데이터
        time = raw_data[time_col].values
        voltage = raw_data["Voltage[V]"].values
        current = raw_data["Current[mA]"].values
        
        # 조건 (충전=1, 방전=2, 휴지=3)
        condition = raw_data["Condition"].values if "Condition" in raw_data.columns else None
        
        # 온도
        temperature = raw_data[temp_col].values if temp_col in raw_data.columns else None
        
        # 용량 계산 (누적)
        dt = np.diff(time, prepend=0)
        capacity = np.cumsum(np.abs(current) * dt / 3600)
        
        return ProfileData(
            time=time,
            voltage=voltage,
            current=current,
            capacity=capacity,
            temperature=temperature,
            condition=condition,
            cycle=cycle
        )


def toyo_cycle_data(raw_path: Union[str, Path],
                    rated_capacity: Optional[float] = None,
                    initial_crate: float = 0.2,
                    **kwargs) -> CycleData:
    """
    Toyo 사이클 데이터 로드 헬퍼 함수
    
    Args:
        raw_path: 데이터 폴더 경로
        rated_capacity: 정격 용량 (mAh)
        initial_crate: 초기 C-rate
        
    Returns:
        CycleData 객체
    """
    loader = ToyoLoader(raw_path, rated_capacity, initial_crate)
    return loader.load_cycle_data(**kwargs)


def toyo_profile_data(raw_path: Union[str, Path],
                      cycle: int,
                      rated_capacity: Optional[float] = None,
                      initial_crate: float = 0.2,
                      **kwargs) -> ProfileData:
    """
    Toyo 프로파일 데이터 로드 헬퍼 함수
    
    Args:
        raw_path: 데이터 폴더 경로
        cycle: 사이클 번호
        rated_capacity: 정격 용량 (mAh)
        initial_crate: 초기 C-rate
        
    Returns:
        ProfileData 객체
    """
    loader = ToyoLoader(raw_path, rated_capacity, initial_crate)
    return loader.load_profile_data(cycle, **kwargs)
