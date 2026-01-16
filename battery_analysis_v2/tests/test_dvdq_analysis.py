"""
dV/dQ Analysis 모듈 단위 테스트
================================
"""

import pytest
import numpy as np
import sys
import os

# 모듈 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestHalfCellProfiles:
    """Half-cell 프로파일 테스트"""
    
    def test_halfcell_profile_creation(self):
        """HalfCellProfile 생성 테스트"""
        from battery_analysis_v2.core.dvdq_analysis import HalfCellProfile
        
        capacity = np.linspace(0, 100, 50)
        voltage = 4.2 - 0.5 * (1 - capacity / 100)
        
        profile = HalfCellProfile(
            name="Test",
            electrode_type="cathode",
            capacity=capacity,
            voltage=voltage
        )
        
        assert profile.name == "Test"
        assert profile.electrode_type == "cathode"
        assert len(profile.capacity) == 50
        assert profile.soc is not None
    
    def test_cathode_profile_helper(self):
        """create_cathode_profile 헬퍼 테스트"""
        from battery_analysis_v2.core.dvdq_analysis import create_cathode_profile
        
        capacity = np.array([0, 50, 100])
        voltage = np.array([4.2, 3.9, 3.5])
        
        profile = create_cathode_profile(capacity, voltage, "NMC")
        
        assert profile.electrode_type == "cathode"
        assert profile.name == "NMC"
    
    def test_anode_profile_helper(self):
        """create_anode_profile 헬퍼 테스트"""
        from battery_analysis_v2.core.dvdq_analysis import create_anode_profile
        
        capacity = np.array([0, 50, 100])
        voltage = np.array([0.3, 0.15, 0.05])
        
        profile = create_anode_profile(capacity, voltage)
        
        assert profile.electrode_type == "anode"
    
    def test_degradation_application(self):
        """열화 파라미터 적용 테스트"""
        from battery_analysis_v2.core.dvdq_analysis import HalfCellProfile
        
        capacity = np.array([0, 50, 100])
        voltage = np.array([4.2, 3.9, 3.5])
        
        profile = HalfCellProfile(
            name="Test",
            electrode_type="cathode",
            capacity=capacity,
            voltage=voltage
        )
        
        new_cap, new_volt = profile.apply_degradation(mass=0.9, slip=5.0)
        
        # mass=0.9, slip=5 적용: new_cap = cap * 0.9 - 5
        expected = capacity * 0.9 - 5.0
        np.testing.assert_array_almost_equal(new_cap, expected)
        np.testing.assert_array_equal(new_volt, voltage)


class TestFullCellFitting:
    """Full-cell 피팅 테스트"""
    
    def test_calculate_dvdq(self):
        """dV/dQ 계산 테스트"""
        from battery_analysis_v2.core.dvdq_analysis import calculate_dvdq
        
        voltage = np.array([3.0, 3.2, 3.4, 3.6, 3.8])
        capacity = np.array([0, 25, 50, 75, 100])
        
        dvdq = calculate_dvdq(voltage, capacity, period=1)
        
        assert len(dvdq) == len(voltage)
        assert np.isnan(dvdq[0])  # 첫 값은 NaN
        
        # dV/dQ = 0.2V / 25% = 0.008
        expected = 0.2 / 25
        np.testing.assert_almost_equal(dvdq[1], expected, decimal=5)
    
    def test_fullcell_simulator(self):
        """FullCellSimulator 테스트"""
        from battery_analysis_v2.core.dvdq_analysis import (
            FullCellSimulator, HalfCellProfile
        )
        
        capacity = np.linspace(0, 100, 50)
        ca_voltage = 4.2 - 0.5 * (1 - capacity / 100)
        an_voltage = 0.1 + 0.2 * (1 - capacity / 100)
        
        cathode = HalfCellProfile("CA", "cathode", capacity, ca_voltage)
        anode = HalfCellProfile("AN", "anode", capacity, an_voltage)
        
        simulator = FullCellSimulator(cathode, anode, rated_capacity=100)
        result = simulator.simulate(ca_mass=0.95, an_mass=0.98)
        
        assert "full_voltage" in result.columns
        assert "full_dvdq" in result.columns
        assert len(result) > 0


class TestDegradationQuantifier:
    """열화 정량화 테스트"""
    
    def test_calculate_lam(self):
        """LAM 계산 테스트"""
        from battery_analysis_v2.core.dvdq_analysis import calculate_lam
        
        # 95% 활물질 잔존 = 5% LAM
        lam = calculate_lam(0.95)
        assert lam == pytest.approx(5.0)
        
        # 100% 잔존 = 0% LAM
        lam = calculate_lam(1.0)
        assert lam == pytest.approx(0.0)
    
    def test_calculate_lli(self):
        """LLI 계산 테스트"""
        from battery_analysis_v2.core.dvdq_analysis import calculate_lli
        
        lli = calculate_lli(slip_ca=0.5, slip_an=0.3, rated_capacity=100)
        
        # LLI = |0.3 - 0.5| / 100 * 100 = 0.2%
        assert lli == pytest.approx(0.2)
    
    def test_lam_invalid_input(self):
        """LAM 잘못된 입력 테스트"""
        from battery_analysis_v2.core.dvdq_analysis import calculate_lam
        
        with pytest.raises(ValueError):
            calculate_lam(1.5)  # > 1 은 에러
        
        with pytest.raises(ValueError):
            calculate_lam(-0.1)  # < 0 은 에러


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
