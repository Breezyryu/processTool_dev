"""
모듈 기능 테스트 스크립트
========================

생성된 모듈들의 기본 기능을 검증합니다.
"""

import sys
import numpy as np

print("=" * 60)
print("Battery Analysis v2 Module Test")
print("=" * 60)

# 1. dV/dQ Analysis 모듈 테스트
print("\n[1] dV/dQ Analysis Module Test")
print("-" * 40)

try:
    from battery_analysis_v2.core.dvdq_analysis import (
        HalfCellProfile,
        create_cathode_profile,
        create_anode_profile,
        calculate_dvdq,
        FullCellSimulator,
        calculate_lam,
        calculate_lli,
        DegradationQuantifier
    )
    print("✅ dvdq_analysis 모듈 import 성공")
    
    # Half-cell 프로파일 테스트
    capacity = np.linspace(0, 100, 100)
    ca_voltage = 4.2 - 0.5 * (1 - capacity/100)
    an_voltage = 0.1 + 0.2 * (1 - capacity/100)
    
    cathode = create_cathode_profile(capacity, ca_voltage, "Test_Cathode")
    anode = create_anode_profile(capacity, an_voltage, "Test_Anode")
    print(f"✅ HalfCellProfile 생성: {cathode.name}, {anode.name}")
    
    # LAM 계산 테스트
    lam_pe = calculate_lam(0.95)
    print(f"✅ LAM 계산: mass=0.95 → LAM={lam_pe:.1f}%")
    
    # LLI 계산 테스트
    lli = calculate_lli(0.5, 0.3)
    print(f"✅ LLI 계산: slip_ca=0.5, slip_an=0.3 → LLI={lli:.2f}%")
    
except Exception as e:
    print(f"❌ dvdq_analysis 모듈 오류: {e}")

# 2. Life Prediction 모듈 테스트
print("\n[2] Life Prediction Module Test")
print("-" * 40)

try:
    from battery_analysis_v2.core.life_prediction.empirical import (
        capacityfit,
        capacityfit_numba,
        CapacityDegradationModel,
        EULifePredictor,
        ApprovalLifePredictor,
        NUMBA_AVAILABLE
    )
    print("✅ life_prediction.empirical 모듈 import 성공")
    print(f"   Numba 사용 가능: {NUMBA_AVAILABLE}")
    
    # capacityfit 테스트
    cycles = np.array([100, 200, 300])
    temps = np.array([298, 298, 298])  # 25°C in K
    
    # 기본 파라미터
    params = (0.03, -18, 0.7, 2.3, -782, -0.28, 96, 1)
    
    result = capacityfit((cycles, temps), *params)
    print(f"✅ capacityfit 실행: {result[:3]}")
    
    if NUMBA_AVAILABLE:
        result_numba = capacityfit_numba(cycles.astype(float), temps.astype(float), *params)
        print(f"✅ capacityfit_numba 실행: {result_numba[:3]}")
    
except Exception as e:
    print(f"❌ life_prediction 모듈 오류: {e}")
    import traceback
    traceback.print_exc()

# 3. 벤치마크 테스트
print("\n[3] Benchmark Test")
print("-" * 40)

try:
    from battery_analysis_v2.core.life_prediction.empirical.capacity_fit import benchmark_fitting
    
    results = benchmark_fitting(n_points=1000, n_iterations=10)
    print(f"✅ 벤치마크 완료:")
    print(f"   데이터 포인트: {results['n_points']}")
    print(f"   Python 시간: {results['python_time_ms']:.3f} ms")
    if results['numba_time_ms']:
        print(f"   Numba 시간: {results['numba_time_ms']:.3f} ms")
        print(f"   속도 향상: {results['speedup']:.1f}x")
    
except Exception as e:
    print(f"❌ 벤치마크 오류: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
