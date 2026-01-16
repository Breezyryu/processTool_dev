# Julia BatteryAnalysis 모듈

> 최종 업데이트: 2026-01-14

Python `battery_analysis_v2` 패키지의 Julia 버전입니다.

## 파일 구조

```
julia/
├── src/
│   ├── BatteryAnalysis.jl      # 메인 모듈
│   ├── Types.jl                # 공통 타입 정의
│   ├── LifePrediction/
│   │   ├── CapacityFit.jl      # 용량 열화 모델
│   │   └── EUPrediction.jl     # EU/승인 수명 예측
│   ├── DVDQAnalysis/
│   │   ├── FullCellFitting.jl  # Full-cell dV/dQ 시뮬레이션
│   │   └── DegradationQuantifier.jl  # LAM/LLI 정량화
│   ├── CycleAnalysis/
│   │   ├── CapacityAnalyzer.jl # 용량 분석
│   │   ├── EfficiencyAnalyzer.jl # 효율 분석
│   │   └── DCIRAnalyzer.jl     # DCIR 분석
│   └── DataLoader/
│       └── BaseLoader.jl       # 데이터 로더
├── test/
│   └── runtests.jl             # 단위 테스트
├── benchmarks/                  # 벤치마크 코드
│   ├── dvdq/
│   └── life_prediction/
└── quick_test.jl               # 빠른 테스트
```

## 사용법

### 기본 사용

```julia
# 모듈 로드
include("src/BatteryAnalysis.jl")
using .BatteryAnalysis

# 용량 열화 예측
params = ModelParameters()
cap = capacityfit(100.0, 298.15, params)
println("Cycle 100 @ 25°C: ", cap * 100, "% remaining")

# 벡터화 계산 (SIMD 최적화)
cycles = collect(100.0:100.0:1000.0)
temps = fill(298.15, length(cycles))
caps = capacityfit_vectorized(cycles, temps, params)

# LAM/LLI 계산
lam_pe = calculate_lam(0.95)  # 5% LAM
lli = calculate_lli(0.5, 0.3)  # 0.2% LLI
```

### 테스트 실행

```bash
cd julia
julia quick_test.jl
```

## 핵심 수식

### 용량 열화 모델
```
capacity = 1 - exp(a*T + b) * (cycle*fd)^b1 - exp(c*T + d) * (cycle*fd)^(e*T + f)
```

### 열화 메커니즘
```julia
LAM = (1 - mass_ratio) × 100%
LLI = |slip_an - slip_ca| / rated_capacity × 100%
```

## Python vs Julia 비교

| 항목 | Python | Julia |
|------|--------|-------|
| 파일 수 | 16개 | 10개 |
| 최적화 | Numba JIT | SIMD, @inbounds |
| 타입 시스템 | 동적 | 정적 (parametric) |
| 테스트 | pytest | Test.jl |

## 테스트 결과

```
1. capacityfit(100, 298.15) = 0.9971 ✅ PASS
2. calculate_lam(0.95) = 5.0 ✅ PASS
3. calculate_lli(0.5, 0.3) = 0.2 ✅ PASS
4. capacity_retention = [1.0, 0.95, 0.9] ✅ PASS
5. coulombic_efficiency = [0.99, 0.98] ✅ PASS
6. capacityfit_vectorized ✅ PASS
```
