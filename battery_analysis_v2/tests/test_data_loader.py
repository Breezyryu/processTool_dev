"""
Data Loader 모듈 단위 테스트
=============================
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestCyclerType:
    """충방전기 타입 테스트"""
    
    def test_cycler_type_enum(self):
        """CyclerType 열거형 테스트"""
        from battery_analysis_v2.core.data_loader import CyclerType
        
        assert CyclerType.PNE.name == "PNE"
        assert CyclerType.TOYO.name == "TOYO"
        assert CyclerType.UNKNOWN.name == "UNKNOWN"


class TestCycleData:
    """CycleData 구조 테스트"""
    
    def test_cycle_data_creation(self):
        """CycleData 생성 테스트"""
        from battery_analysis_v2.core.data_loader import CycleData
        
        data = CycleData(
            cycle=np.array([1, 2, 3]),
            charge_capacity=np.array([100, 99, 98]),
            discharge_capacity=np.array([99, 98, 97]),
            efficiency=np.array([0.99, 0.99, 0.99]),
            rated_capacity=100.0
        )
        
        assert len(data.cycle) == 3
        assert data.rated_capacity == 100.0
    
    def test_capacity_retention(self):
        """capacity_retention 프로퍼티 테스트"""
        from battery_analysis_v2.core.data_loader import CycleData
        
        data = CycleData(
            cycle=np.array([1, 2, 3]),
            charge_capacity=np.array([100, 99, 98]),
            discharge_capacity=np.array([100, 95, 90]),
            efficiency=np.array([1.0, 0.96, 0.92]),
            rated_capacity=100.0
        )
        
        retention = data.capacity_retention
        
        assert retention[0] == 1.0  # 첫 사이클은 100%
        assert retention[1] == 0.95
        assert retention[2] == 0.90
    
    def test_soh_calculation(self):
        """SOH 계산 테스트"""
        from battery_analysis_v2.core.data_loader import CycleData
        
        data = CycleData(
            cycle=np.array([1, 2]),
            charge_capacity=np.array([100, 99]),
            discharge_capacity=np.array([80, 78]),
            efficiency=np.array([0.8, 0.79]),
            rated_capacity=100.0
        )
        
        soh = data.soh
        
        assert soh[0] == 0.8  # 80%
        assert soh[1] == 0.78
    
    def test_to_dataframe(self):
        """DataFrame 변환 테스트"""
        from battery_analysis_v2.core.data_loader import CycleData
        
        data = CycleData(
            cycle=np.array([1, 2]),
            charge_capacity=np.array([100, 99]),
            discharge_capacity=np.array([99, 98]),
            efficiency=np.array([0.99, 0.99]),
            rated_capacity=100.0
        )
        
        df = data.to_dataframe()
        
        assert "cycle" in df.columns
        assert "discharge_capacity" in df.columns
        assert len(df) == 2


class TestProfileData:
    """ProfileData 구조 테스트"""
    
    def test_profile_data_creation(self):
        """ProfileData 생성 테스트"""
        from battery_analysis_v2.core.data_loader import ProfileData
        
        data = ProfileData(
            time=np.array([0, 1, 2, 3]),
            voltage=np.array([3.0, 3.5, 4.0, 4.2]),
            current=np.array([1000, 1000, 1000, 0]),
            capacity=np.array([0, 1, 2, 3])
        )
        
        assert len(data.time) == 4
        assert data.cycle is None
    
    def test_filter_by_condition(self):
        """조건별 필터링 테스트"""
        from battery_analysis_v2.core.data_loader import ProfileData
        
        data = ProfileData(
            time=np.array([0, 1, 2, 3]),
            voltage=np.array([3.0, 3.5, 4.0, 4.2]),
            current=np.array([1000, 1000, -1000, -1000]),
            capacity=np.array([0, 1, 2, 3]),
            condition=np.array([1, 1, 2, 2])  # 1=충전, 2=방전
        )
        
        charge = data.filter_by_condition(1)
        discharge = data.filter_by_condition(2)
        
        assert len(charge.time) == 2
        assert len(discharge.time) == 2


class TestDetectCyclerType:
    """충방전기 타입 감지 테스트"""
    
    def test_detect_unknown(self):
        """알 수 없는 타입 테스트"""
        from battery_analysis_v2.core.data_loader import detect_cycler_type, CyclerType
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = detect_cycler_type(tmpdir)
            assert result == CyclerType.UNKNOWN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
