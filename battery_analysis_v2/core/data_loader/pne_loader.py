"""
PNE Loader Module
=================

PNE 충방전기 데이터 로더입니다.

데이터 구조:
- Pattern 폴더: 패턴 설정
- Restore/SaveEndData.csv: 사이클 요약 데이터
- Restore/*.csv: 프로파일 데이터

원본 함수: BatteryDataTool.py pne_cycle_data(), pne_data()

Author: Battery Analysis Team
Date: 2026-01-14
"""

import os
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
import numpy as np

from .base_loader import (
    BaseCyclerLoader,
    CycleData,
    ProfileData,
    CyclerType,
)


class PNELoader(BaseCyclerLoader):
    """
    PNE 충방전기 데이터 로더
    
    PNE 데이터 특징:
    - Pattern 폴더 존재
    - Restore/SaveEndData.csv에 사이클 데이터
    - 컬럼 인덱스 기반 데이터 (헤더 없음)
    
    사용 예시:
        >>> loader = PNELoader("/path/to/data")
        >>> cycle_data = loader.load_cycle_data()
        >>> profile = loader.load_profile_data(cycle=1)
    """
    
    # PNE 컬럼 인덱스 매핑
    # 0:Index 1:- 2:StepType 3:ChgDchg 4:- 5:충전
    # 6:EndState 7:Step 8:Voltage(mV) 9:Current(A) 10:ChgCapacity(mAh) 11:DchgCapacity(mAh)
    # 12:ChgPower(W) 13:DchgPower(W) 14:ChgWattHour(Wh) 15:DchgWattHour(Wh)
    # 17:StepTime(s) 20:imp 24:Temperature 27:TotalCycle 28:CurrCycle 29:AvgVoltage 45:voltage_max
    COL_TOTAL_CYCLE = 27
    COL_CONDITION = 2  # StepType: 1=충전, 2=방전, 3=휴지, 8=loop
    COL_CHG_CAP = 10
    COL_DCHG_CAP = 11
    COL_VOLTAGE = 8
    COL_IMP = 20
    COL_VOL_MAX = 45
    COL_DCHG_ENERGY = 15
    COL_STEP_TIME = 17
    COL_CURRENT = 9
    COL_TEMP = 24
    COL_AVG_V = 29
    COL_END_STATE = 6  # 66=충전완료, 65=방전완료, 64=휴지, 78=용량
    
    @property
    def cycler_type(self) -> CyclerType:
        return CyclerType.PNE
    
    def _get_restore_path(self) -> Path:
        """Restore 폴더 경로"""
        return self.raw_path / "Restore"
    
    def _find_save_end_data(self) -> Optional[Path]:
        """SaveEndData.csv 파일 찾기"""
        restore_path = self._get_restore_path()
        if not restore_path.is_dir():
            return None
        
        for f in restore_path.iterdir():
            if f.is_file() and "SaveEndData.csv" in f.name:
                return f
        return None
    
    def _estimate_capacity(self) -> float:
        """용량 추정"""
        # 경로에서 추출 시도
        capacity = self._extract_capacity_from_path()
        if capacity > 0:
            return capacity
        
        # 데이터에서 추정 시도
        save_end_file = self._find_save_end_data()
        if save_end_file and save_end_file.stat().st_size > 0:
            try:
                df = pd.read_csv(
                    save_end_file,
                    sep=",",
                    header=None,
                    encoding="cp949",
                    on_bad_lines='skip'
                )
                if self.COL_CURRENT < len(df.columns):
                    max_current = df.iloc[:, self.COL_CURRENT].abs().max()
                    if max_current > 0:
                        return max_current / self.initial_crate
            except Exception:
                pass
        
        return 0.0
    
    def _is_pne21_or_22(self) -> bool:
        """PNE21 또는 PNE22 여부 확인 (단위 변환 필요)"""
        path_str = str(self.raw_path)
        return 'PNE21' in path_str or 'PNE22' in path_str
    
    def load_cycle_data(self,
                        check_dcir: bool = False,
                        check_dcir2: bool = False,
                        make_dcir: bool = False,
                        **kwargs) -> CycleData:
        """
        사이클 데이터 로드
        
        원본 함수: pne_cycle_data()
        
        Args:
            check_dcir: 기본 DCIR 계산
            check_dcir2: 대체 DCIR 계산
            make_dcir: RSS/1s DCIR 계산
            
        Returns:
            CycleData 객체
        """
        # 용량 확인
        if self._rated_capacity == 0:
            self._rated_capacity = self._estimate_capacity()
        
        mincapacity = self._rated_capacity
        
        # SaveEndData.csv 찾기
        save_end_file = self._find_save_end_data()
        if save_end_file is None:
            raise FileNotFoundError(f"SaveEndData.csv not found in {self._get_restore_path()}")
        
        if save_end_file.stat().st_size == 0:
            raise ValueError("SaveEndData.csv is empty")
        
        # CSV 로드
        cycleraw = pd.read_csv(
            save_end_file,
            sep=",",
            header=None,
            encoding="cp949",
            on_bad_lines='skip'
        )
        
        # 필요한 컬럼 선택
        cols_needed = [self.COL_TOTAL_CYCLE, self.COL_CONDITION, self.COL_CHG_CAP, 
                       self.COL_DCHG_CAP, self.COL_VOLTAGE, self.COL_IMP, 
                       self.COL_VOL_MAX, self.COL_DCHG_ENERGY, self.COL_STEP_TIME,
                       self.COL_CURRENT, self.COL_TEMP, self.COL_AVG_V, self.COL_END_STATE]
        
        available_cols = [c for c in cols_needed if c < len(cycleraw.columns)]
        cycleraw = cycleraw.iloc[:, available_cols].copy()
        cycleraw.columns = ["TotlCycle", "Condition", "chgCap", "DchgCap", 
                           "Ocv", "imp", "volmax", "DchgEng", "steptime",
                           "Curr", "Temp", "AvgV", "EndState"][:len(available_cols)]
        
        # PNE21/22는 단위 변환 필요
        if self._is_pne21_or_22():
            for col in ["DchgCap", "chgCap", "Curr"]:
                if col in cycleraw.columns:
                    cycleraw[col] = cycleraw[col] / 1000
        
        # 충전 데이터: Condition=1, 충전용량 > 최소값
        min_cap_threshold = mincapacity / 60
        charge_mask = (cycleraw["Condition"] == 1) & (cycleraw["chgCap"] > min_cap_threshold)
        charge_data = cycleraw[charge_mask].copy()
        
        # 방전 데이터: Condition=2, 방전용량 > 최소값
        discharge_mask = (cycleraw["Condition"] == 2) & (cycleraw["DchgCap"] > min_cap_threshold)
        discharge_data = cycleraw[discharge_mask].copy()
        
        # 인덱스 설정
        if not charge_data.empty:
            charge_data = charge_data.set_index("TotlCycle")
        if not discharge_data.empty:
            discharge_data = discharge_data.set_index("TotlCycle")
        
        # 공통 사이클
        common_cycles = np.intersect1d(
            charge_data.index.values if not charge_data.empty else [],
            discharge_data.index.values if not discharge_data.empty else []
        )
        
        if len(common_cycles) == 0:
            raise ValueError("No valid cycle data found")
        
        # 데이터 추출
        charge_cap = charge_data.loc[common_cycles, "chgCap"].values
        discharge_cap = discharge_data.loc[common_cycles, "DchgCap"].values
        
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
        if "DchgEng" in discharge_data.columns:
            result.discharge_energy = discharge_data.loc[common_cycles, "DchgEng"].values
        if "Temp" in discharge_data.columns:
            result.temperature = discharge_data.loc[common_cycles, "Temp"].values
        if "Ocv" in charge_data.columns:
            result.rest_voltage = charge_data.loc[common_cycles, "Ocv"].values / 1000  # mV -> V
        if "AvgV" in discharge_data.columns:
            result.average_voltage = discharge_data.loc[common_cycles, "AvgV"].values / 1000
        
        # DCIR 계산 (옵션)
        if check_dcir and "imp" in cycleraw.columns and "volmax" in cycleraw.columns:
            dcir_mask = (cycleraw["Condition"] == 2) & (cycleraw["volmax"] > 4100000)
            dcir_data = cycleraw[dcir_mask].copy()
            if not dcir_data.empty:
                dcir_values = dcir_data["imp"].values / 1000
                # 사이클에 매핑 (간략화)
                result.dcir = dcir_values[:len(common_cycles)] if len(dcir_values) >= len(common_cycles) else None
        
        return result
    
    def load_profile_data(self, cycle: int, **kwargs) -> ProfileData:
        """
        특정 사이클의 프로파일 데이터 로드
        
        원본 함수: pne_data()
        
        Args:
            cycle: 사이클 번호
            
        Returns:
            ProfileData 객체
        """
        # 프로파일 파일 찾기
        restore_path = self._get_restore_path()
        if not restore_path.is_dir():
            raise FileNotFoundError(f"Restore folder not found: {restore_path}")
        
        # 사이클에 해당하는 파일 검색
        profile_file = self._find_profile_file(cycle)
        if profile_file is None:
            raise FileNotFoundError(f"Profile file for cycle {cycle} not found")
        
        # CSV 로드
        raw_data = pd.read_csv(
            profile_file,
            sep=",",
            header=None,
            encoding="cp949",
            on_bad_lines='skip'
        )
        
        # 컬럼 추출
        # 0:시간, 8:전압(mV), 9:전류(A), 24:온도, 2:조건
        time = raw_data.iloc[:, 0].values if 0 < len(raw_data.columns) else np.array([])
        voltage = raw_data.iloc[:, 8].values / 1000 if 8 < len(raw_data.columns) else np.array([])  # mV -> V
        current = raw_data.iloc[:, 9].values * 1000 if 9 < len(raw_data.columns) else np.array([])  # A -> mA
        
        temp_col = 24 if 24 < len(raw_data.columns) else None
        temperature = raw_data.iloc[:, temp_col].values if temp_col else None
        
        cond_col = 2 if 2 < len(raw_data.columns) else None  
        condition = raw_data.iloc[:, cond_col].values if cond_col else None
        
        step_col = 7 if 7 < len(raw_data.columns) else None
        step = raw_data.iloc[:, step_col].values if step_col else None
        
        # PNE21/22 단위 변환
        if self._is_pne21_or_22():
            current = current / 1000
        
        # 용량 계산
        if len(time) > 0:
            dt = np.diff(time, prepend=0)
            capacity = np.cumsum(np.abs(current) * dt / 3600)
        else:
            capacity = np.array([])
        
        return ProfileData(
            time=time,
            voltage=voltage,
            current=current,
            capacity=capacity,
            temperature=temperature,
            condition=condition,
            step=step,
            cycle=cycle
        )
    
    def _find_profile_file(self, cycle: int) -> Optional[Path]:
        """사이클에 해당하는 프로파일 파일 찾기"""
        restore_path = self._get_restore_path()
        
        # CSV 파일 목록
        csv_files = sorted([f for f in restore_path.iterdir() 
                           if f.is_file() and f.suffix.lower() == '.csv'
                           and 'SaveEndData' not in f.name])
        
        # 사이클 번호로 검색 (간략화)
        for f in csv_files:
            try:
                df = pd.read_csv(f, sep=",", header=None, encoding="cp949",
                               nrows=10, on_bad_lines='skip')
                if self.COL_TOTAL_CYCLE < len(df.columns):
                    cycles_in_file = df.iloc[:, self.COL_TOTAL_CYCLE].unique()
                    if cycle in cycles_in_file:
                        return f
            except Exception:
                continue
        
        # 찾지 못하면 인덱스로 시도
        if len(csv_files) > cycle - 1:
            return csv_files[cycle - 1]
        
        return None


def pne_cycle_data(raw_path: Union[str, Path],
                   rated_capacity: Optional[float] = None,
                   initial_crate: float = 0.2,
                   **kwargs) -> CycleData:
    """
    PNE 사이클 데이터 로드 헬퍼 함수
    
    Args:
        raw_path: 데이터 폴더 경로
        rated_capacity: 정격 용량 (mAh)
        initial_crate: 초기 C-rate
        
    Returns:
        CycleData 객체
    """
    loader = PNELoader(raw_path, rated_capacity, initial_crate)
    return loader.load_cycle_data(**kwargs)


def pne_profile_data(raw_path: Union[str, Path],
                     cycle: int,
                     rated_capacity: Optional[float] = None,
                     initial_crate: float = 0.2,
                     **kwargs) -> ProfileData:
    """
    PNE 프로파일 데이터 로드 헬퍼 함수
    
    Args:
        raw_path: 데이터 폴더 경로
        cycle: 사이클 번호
        rated_capacity: 정격 용량 (mAh)
        initial_crate: 초기 C-rate
        
    Returns:
        ProfileData 객체
    """
    loader = PNELoader(raw_path, rated_capacity, initial_crate)
    return loader.load_profile_data(cycle, **kwargs)
