"""
Cycle Classifier Module
=======================

사이클 유형 분류 모듈입니다.

주요 기능:
- RPT 사이클 식별
- 노화 패턴 분류
- 저항 측정 사이클 식별

원본 참조: cycle_categorizer.py, cycle_phase_categorizer.py

Author: Battery Analysis Team
Date: 2026-01-14
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto


class CycleType(Enum):
    """사이클 유형"""
    NORMAL = auto()           # 일반 충방전
    RPT = auto()              # Reference Performance Test
    AGING = auto()            # 가속 노화
    DCIR = auto()             # DCIR 측정
    SOC_CALIBRATION = auto()  # SOC 보정
    FORMATION = auto()        # 포메이션
    RATE_TEST = auto()        # Rate 테스트
    UNKNOWN = auto()


@dataclass
class CycleClassification:
    """사이클 분류 결과"""
    cycle: int
    cycle_type: CycleType
    crate: float
    confidence: float
    features: Dict


def classify_cycles(cycles: np.ndarray,
                    charge_capacity: np.ndarray,
                    discharge_capacity: np.ndarray,
                    current: Optional[np.ndarray] = None,
                    rated_capacity: Optional[float] = None,
                    step_time: Optional[np.ndarray] = None) -> List[CycleClassification]:
    """
    사이클 자동 분류
    
    Args:
        cycles: 사이클 배열
        charge_capacity: 충전 용량
        discharge_capacity: 방전 용량
        current: 전류 (C-rate 계산용)
        rated_capacity: 정격 용량
        step_time: 스텝 시간
        
    Returns:
        CycleClassification 리스트
    """
    classifier = CycleClassifier(
        cycles, charge_capacity, discharge_capacity,
        current, rated_capacity, step_time
    )
    return classifier.classify_all()


class CycleClassifier:
    """
    사이클 분류기 클래스
    
    분류 기준:
    - C-rate: 낮으면 RPT 가능성
    - 용량 변화: 급격한 변화는 테스트 사이클
    - 스텝 시간: 짧으면 DCIR
    
    사용 예시:
        >>> classifier = CycleClassifier(cycles, charge, discharge, crate=crates)
        >>> results = classifier.classify_all()
        >>> rpt_cycles = classifier.get_cycles_by_type(CycleType.RPT)
    """
    
    # 분류 임계값
    RPT_CRATE_THRESHOLD = 0.5      # RPT는 보통 0.5C 이하
    DCIR_DURATION_THRESHOLD = 60    # DCIR은 60초 이하
    AGING_CRATE_THRESHOLD = 1.0     # 노화는 1C 이상
    
    def __init__(self,
                 cycles: np.ndarray,
                 charge_capacity: np.ndarray,
                 discharge_capacity: np.ndarray,
                 current: Optional[np.ndarray] = None,
                 rated_capacity: Optional[float] = None,
                 step_time: Optional[np.ndarray] = None):
        """
        Args:
            cycles: 사이클 배열
            charge_capacity: 충전 용량
            discharge_capacity: 방전 용량
            current: 전류 배열 (mA)
            rated_capacity: 정격 용량 (mAh)
            step_time: 스텝 시간 (초)
        """
        self.cycles = np.asarray(cycles)
        self.charge_capacity = np.asarray(charge_capacity)
        self.discharge_capacity = np.asarray(discharge_capacity)
        self.current = current
        self.step_time = step_time
        
        # 정격 용량 추정
        if rated_capacity is not None:
            self.rated_capacity = rated_capacity
        else:
            self.rated_capacity = np.max(discharge_capacity)
        
        # C-rate 계산
        if current is not None and self.rated_capacity > 0:
            self.crate = np.abs(current) / self.rated_capacity
        else:
            self.crate = np.ones_like(cycles, dtype=float)
        
        # 분류 결과 저장
        self._classifications: Optional[List[CycleClassification]] = None
    
    def classify_all(self) -> List[CycleClassification]:
        """모든 사이클 분류"""
        if self._classifications is not None:
            return self._classifications
        
        results = []
        
        for i, cycle in enumerate(self.cycles):
            classification = self._classify_single(i)
            results.append(classification)
        
        self._classifications = results
        return results
    
    def _classify_single(self, idx: int) -> CycleClassification:
        """단일 사이클 분류"""
        cycle = int(self.cycles[idx])
        crate = float(self.crate[idx])
        
        features = {
            'charge_capacity': float(self.charge_capacity[idx]),
            'discharge_capacity': float(self.discharge_capacity[idx]),
            'crate': crate
        }
        
        if self.step_time is not None:
            features['step_time'] = float(self.step_time[idx])
        
        # 분류 로직
        cycle_type = CycleType.NORMAL
        confidence = 0.5
        
        # DCIR 사이클 (짧은 시간)
        if self.step_time is not None and self.step_time[idx] < self.DCIR_DURATION_THRESHOLD:
            cycle_type = CycleType.DCIR
            confidence = 0.8
        
        # RPT 사이클 (낮은 C-rate)
        elif crate < self.RPT_CRATE_THRESHOLD:
            cycle_type = CycleType.RPT
            confidence = 0.7 + 0.2 * (self.RPT_CRATE_THRESHOLD - crate) / self.RPT_CRATE_THRESHOLD
        
        # 노화 사이클 (높은 C-rate)
        elif crate >= self.AGING_CRATE_THRESHOLD:
            cycle_type = CycleType.AGING
            confidence = 0.6 + 0.2 * min(crate / 2, 1)
        
        # Rate 테스트 (다양한 C-rate)
        elif idx > 0 and abs(crate - self.crate[idx-1]) > 0.5:
            cycle_type = CycleType.RATE_TEST
            confidence = 0.6
        
        # 포메이션 (초기 사이클, 낮은 C-rate)
        if cycle <= 5 and crate < 0.3:
            cycle_type = CycleType.FORMATION
            confidence = 0.7
        
        return CycleClassification(
            cycle=cycle,
            cycle_type=cycle_type,
            crate=crate,
            confidence=confidence,
            features=features
        )
    
    def get_cycles_by_type(self, cycle_type: CycleType) -> np.ndarray:
        """특정 유형의 사이클 번호 반환"""
        if self._classifications is None:
            self.classify_all()
        
        return np.array([
            c.cycle for c in self._classifications 
            if c.cycle_type == cycle_type
        ])
    
    def get_rpt_cycles(self) -> np.ndarray:
        """RPT 사이클 반환"""
        return self.get_cycles_by_type(CycleType.RPT)
    
    def get_aging_cycles(self) -> np.ndarray:
        """노화 사이클 반환"""
        return self.get_cycles_by_type(CycleType.AGING)
    
    def get_dcir_cycles(self) -> np.ndarray:
        """DCIR 사이클 반환"""
        return self.get_cycles_by_type(CycleType.DCIR)
    
    def to_dataframe(self) -> pd.DataFrame:
        """분류 결과를 DataFrame으로 변환"""
        if self._classifications is None:
            self.classify_all()
        
        return pd.DataFrame([
            {
                'cycle': c.cycle,
                'type': c.cycle_type.name,
                'crate': c.crate,
                'confidence': c.confidence,
                **c.features
            }
            for c in self._classifications
        ])
    
    def summary(self) -> Dict:
        """분류 요약"""
        if self._classifications is None:
            self.classify_all()
        
        type_counts = {}
        for c in self._classifications:
            type_name = c.cycle_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            'total_cycles': len(self._classifications),
            'type_distribution': type_counts,
            'rpt_count': len(self.get_rpt_cycles()),
            'aging_count': len(self.get_aging_cycles())
        }
