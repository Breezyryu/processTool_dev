"""
Cycle Analysis 모듈 단위 테스트
================================
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestCapacityAnalyzer:
    """용량 분석기 테스트"""
    
    def test_capacity_retention(self):
        """용량 유지율 계산 테스트"""
        from battery_analysis_v2.core.cycle_analysis import calculate_capacity_retention
        
        capacity = np.array([100, 95, 90, 85, 80])
        retention = calculate_capacity_retention(capacity)
        
        assert retention[0] == 1.0
        assert retention[-1] == 0.8
    
    def test_capacity_retention_custom_reference(self):
        """사용자 정의 기준 용량 테스트"""
        from battery_analysis_v2.core.cycle_analysis import calculate_capacity_retention
        
        capacity = np.array([100, 90, 80])
        retention = calculate_capacity_retention(capacity, reference=100)
        
        np.testing.assert_array_equal(retention, [1.0, 0.9, 0.8])
    
    def test_fade_rate_calculation(self):
        """열화율 계산 테스트"""
        from battery_analysis_v2.core.cycle_analysis import calculate_capacity_fade_rate
        
        cycles = np.array([0, 100, 200, 300, 400])
        capacity = np.array([100, 98, 96, 94, 92])  # 100사이클당 2% 감소
        
        result = calculate_capacity_fade_rate(cycles, capacity, method='linear')
        
        assert result.fade_rate_per_100_cycles == pytest.approx(2.0, rel=0.1)
        assert result.total_fade == pytest.approx(8.0, rel=0.1)
    
    def test_eol_prediction(self):
        """EOL 예측 테스트"""
        from battery_analysis_v2.core.cycle_analysis import predict_eol_cycle
        
        cycles = np.array([0, 100, 200])
        capacity = np.array([100, 98, 96])  # 100사이클당 2% 감소
        
        result = predict_eol_cycle(cycles, capacity, eol_threshold=80)
        
        # 80%까지 20% 감소, 100사이클당 2% → ~1000 사이클
        assert result.predicted_cycle > 500
        assert result.eol_threshold == 80
    
    def test_capacity_analyzer_class(self):
        """CapacityAnalyzer 클래스 테스트"""
        from battery_analysis_v2.core.cycle_analysis import CapacityAnalyzer
        
        cycles = np.arange(0, 500, 10)
        capacity = 100 * (1 - 0.0001 * cycles)  # 선형 감소
        
        analyzer = CapacityAnalyzer(cycles, capacity, rated_capacity=100)
        
        assert len(analyzer.retention) == len(cycles)
        assert analyzer.soh[-1] < 1.0


class TestEfficiencyAnalyzer:
    """효율 분석기 테스트"""
    
    def test_coulombic_efficiency(self):
        """쿨롱 효율 계산 테스트"""
        from battery_analysis_v2.core.cycle_analysis import calculate_coulombic_efficiency
        
        charge = np.array([100, 100, 100])
        discharge = np.array([99, 98, 97])
        
        ce = calculate_coulombic_efficiency(charge, discharge)
        
        assert ce[0] == pytest.approx(0.99)
        assert ce[2] == pytest.approx(0.97)
    
    def test_energy_efficiency(self):
        """에너지 효율 계산 테스트"""
        from battery_analysis_v2.core.cycle_analysis import calculate_energy_efficiency
        
        charge_energy = np.array([400, 405, 410])
        discharge_energy = np.array([380, 382, 384])
        
        ee = calculate_energy_efficiency(charge_energy, discharge_energy)
        
        assert all(ee < 1.0)
        assert all(ee > 0.9)
    
    def test_efficiency_analyzer_class(self):
        """EfficiencyAnalyzer 클래스 테스트"""
        from battery_analysis_v2.core.cycle_analysis import EfficiencyAnalyzer
        
        charge = np.full(100, 100.0)
        discharge = np.full(100, 99.5)
        
        analyzer = EfficiencyAnalyzer(charge, discharge)
        
        assert np.mean(analyzer.coulombic_efficiency) == pytest.approx(0.995)
        
        stats = analyzer.coulombic_efficiency_stats()
        assert stats.mean == pytest.approx(0.995)


class TestDCIRAnalyzer:
    """DCIR 분석기 테스트"""
    
    def test_dcir_growth(self):
        """DCIR 성장 계산 테스트"""
        from battery_analysis_v2.core.cycle_analysis import calculate_dcir_growth
        
        cycles = np.array([0, 100, 200, 300, 400])
        dcir = np.array([10, 10.5, 11, 11.5, 12])  # 선형 증가
        
        result = calculate_dcir_growth(cycles, dcir)
        
        assert result.initial_dcir == 10
        assert result.current_dcir == 12
        assert result.total_growth == 2
        assert result.total_growth_percent == pytest.approx(20.0)
    
    def test_dcir_analyzer_class(self):
        """DCIRAnalyzer 클래스 테스트"""
        from battery_analysis_v2.core.cycle_analysis import DCIRAnalyzer
        
        cycles = np.arange(0, 500, 10)
        dcir = 10 + 0.01 * cycles  # 선형 증가
        
        analyzer = DCIRAnalyzer(cycles, dcir)
        growth = analyzer.analyze_growth()
        
        assert growth.initial_dcir == 10
        assert growth.current_dcir > growth.initial_dcir


class TestCycleClassifier:
    """사이클 분류기 테스트"""
    
    def test_cycle_type_enum(self):
        """CycleType 열거형 테스트"""
        from battery_analysis_v2.core.cycle_analysis import CycleType
        
        assert CycleType.NORMAL.name == "NORMAL"
        assert CycleType.RPT.name == "RPT"
        assert CycleType.AGING.name == "AGING"
    
    def test_classifier_basic(self):
        """기본 분류 테스트"""
        from battery_analysis_v2.core.cycle_analysis import CycleClassifier, CycleType
        
        cycles = np.array([1, 2, 3, 4, 5])
        charge = np.array([100, 100, 100, 100, 100])
        discharge = np.array([99, 99, 99, 99, 99])
        current = np.array([50, 50, 50, 50, 50])  # 낮은 C-rate
        
        classifier = CycleClassifier(cycles, charge, discharge, current, rated_capacity=100)
        results = classifier.classify_all()
        
        assert len(results) == 5
    
    def test_classify_function(self):
        """classify_cycles 함수 테스트"""
        from battery_analysis_v2.core.cycle_analysis import classify_cycles
        
        cycles = np.array([1, 2, 3])
        charge = np.array([100, 100, 100])
        discharge = np.array([99, 99, 99])
        
        results = classify_cycles(cycles, charge, discharge)
        
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
