"""
BatteryAnalysis.jl - 배터리 분석 모듈
======================================

Python battery_analysis_v2 패키지의 Julia 버전입니다.

모듈 구조:
- DataLoader: PNE/Toyo 충방전기 데이터 로드
- CycleAnalysis: 용량/효율/DCIR 분석
- DVDQAnalysis: dV/dQ 분석, LAM/LLI 정량화
- LifePrediction: 수명 예측 모델

Author: Battery Analysis Team
Date: 2026-01-14
"""
module BatteryAnalysis

using LinearAlgebra
using Statistics

# 하위 모듈 include
include("Types.jl")
include("LifePrediction/CapacityFit.jl")
include("LifePrediction/EUPrediction.jl")
include("DVDQAnalysis/FullCellFitting.jl")
include("DVDQAnalysis/DegradationQuantifier.jl")
include("CycleAnalysis/CapacityAnalyzer.jl")
include("CycleAnalysis/EfficiencyAnalyzer.jl")
include("CycleAnalysis/DCIRAnalyzer.jl")
include("DataLoader/BaseLoader.jl")

# 타입 export
export CycleData, ProfileData, CyclerType
export FadeAnalysisResult, EOLPrediction
export DegradationMetrics, FittingParams

# Life Prediction export
export capacityfit, capacityfit_vectorized
export swellingfit
export predict_eol_cycle, calculate_fade_rate
export ModelParameters

# dV/dQ Analysis export
export calculate_dvdq, simulate_fullcell
export calculate_lam, calculate_lli

# Cycle Analysis export
export calculate_capacity_retention
export calculate_coulombic_efficiency, calculate_energy_efficiency
export calculate_dcir_growth

# Data Loader export
export detect_cycler_type, load_cycle_data

end # module
