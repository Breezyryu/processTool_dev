"""
Life Prediction 모듈 단위 테스트
=================================
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestCapacityFit:
    """용량 피팅 테스트"""
    
    def test_capacityfit_basic(self):
        """capacityfit 기본 계산 테스트"""
        from battery_analysis_v2.core.life_prediction.empirical import capacityfit
        
        cycles = np.array([100, 200, 300])
        temps = np.array([298.15, 298.15, 298.15])  # 25°C
        
        # 기본 파라미터
        result = capacityfit(
            (cycles, temps),
            a=0.03, b=-18, b1=0.7,
            c=2.3, d=-782, e=-0.28, f=96, fd=1
        )
        
        assert len(result) == 3
        assert all(result <= 1.0)  # 용량은 1 이하
        assert all(result > 0)     # 용량은 양수
    

    def test_model_parameters_default(self):
        """ModelParameters 기본값 테스트"""
        from battery_analysis_v2.core.life_prediction.empirical.capacity_fit import ModelParameters
        
        params = ModelParameters.default()
        
        assert params.a == 0.03
        assert params.b == -18
        assert params.fd == 1
    
    def test_model_parameters_array_conversion(self):
        """ModelParameters 배열 변환 테스트"""
        from battery_analysis_v2.core.life_prediction.empirical.capacity_fit import ModelParameters
        
        params = ModelParameters(a=0.1, b=-10, b1=0.5, c=1.0, d=-500, e=-0.2, f=50, fd=2)
        arr = params.to_array()
        
        assert len(arr) == 8
        assert arr[0] == 0.1
        assert arr[7] == 2
        
        restored = ModelParameters.from_array(arr)
        assert restored.a == params.a
        assert restored.fd == params.fd


class TestCapacityDegradationModel:
    """용량 열화 모델 클래스 테스트"""
    
    def test_model_prediction(self):
        """모델 예측 테스트"""
        from battery_analysis_v2.core.life_prediction.empirical import CapacityDegradationModel
        
        model = CapacityDegradationModel()
        
        cycles = np.array([100, 500, 1000])
        temps = np.array([298, 298, 298])
        
        predictions = model.predict(cycles, temps)
        
        assert len(predictions) == 3
        # 사이클이 증가할수록 용량 감소
        assert predictions[0] > predictions[2]
    
    def test_eol_prediction(self):
        """EOL 예측 테스트"""
        from battery_analysis_v2.core.life_prediction.empirical import CapacityDegradationModel
        
        model = CapacityDegradationModel()
        
        eol_cycle = model.predict_cycle_to_eol(
            temperature=298.15,
            eol_threshold=0.8,
            max_cycles=5000
        )
        
        assert eol_cycle > 0
        assert eol_cycle <= 5000


class TestEULifePredictor:
    """EU 수명 예측기 테스트"""
    
    def test_eu_predictor_creation(self):
        """EULifePredictor 생성 테스트"""
        from battery_analysis_v2.core.life_prediction.empirical import EULifePredictor
        
        predictor = EULifePredictor(required_cycles=1000)
        
        assert predictor.required_cycles == 1000
        assert predictor.eol_threshold == 0.8


class TestApprovalLifePredictor:
    """승인 수명 예측기 테스트"""
    
    def test_arrhenius_factor(self):
        """Arrhenius 가속 계수 테스트"""
        from battery_analysis_v2.core.life_prediction.empirical import ApprovalLifePredictor
        
        predictor = ApprovalLifePredictor(
            test_temperature=318.15,    # 45°C
            target_temperature=298.15   # 25°C
        )
        
        af = predictor.calculate_arrhenius_factor(activation_energy=50000)
        
        # 고온에서 저온으로: 가속 계수 > 1
        assert af > 1
    
    def test_sensitivity_analysis(self):
        """민감도 분석 테스트"""
        from battery_analysis_v2.core.life_prediction.empirical import ApprovalLifePredictor
        
        predictor = ApprovalLifePredictor(
            test_temperature=318.15,
            target_temperature=298.15
        )
        
        result = predictor.sensitivity_analysis(
            activation_energy_range=(40000, 80000),
            n_points=5
        )
        
        assert len(result) == 5
        assert "acceleration_factor" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
