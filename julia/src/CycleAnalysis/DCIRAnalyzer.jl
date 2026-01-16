"""
DCIRAnalyzer.jl - DCIR 분석
============================

Python dcir_analyzer.py의 Julia 버전입니다.
"""

"""
    DCIRGrowthResult

DCIR 성장 분석 결과
"""
struct DCIRGrowthResult
    growth_rate_per_cycle::Float64
    growth_rate_percent::Float64
    initial_dcir::Float64
    current_dcir::Float64
    total_growth::Float64
    total_growth_percent::Float64
    r_squared::Float64
end


"""
    calculate_dcir_growth(cycles, dcir)

DCIR 증가율 계산

# Arguments
- `cycles`: 사이클 배열
- `dcir`: DCIR 배열 (mΩ)

# Returns
- `DCIRGrowthResult` 구조체
"""
function calculate_dcir_growth(cycles::Vector{<:Real}, dcir::Vector{<:Real})
    # NaN 제거
    valid_mask = .!isnan.(dcir)
    x = Float64.(cycles[valid_mask])
    y = Float64.(dcir[valid_mask])

    if length(x) < 2
        return DCIRGrowthResult(0, 0, y[1], y[end], 0, 0, 0)
    end

    # 선형 회귀
    x_mean = mean(x)
    y_mean = mean(y)

    ss_xy = sum((x .- x_mean) .* (y .- y_mean))
    ss_xx = sum((x .- x_mean) .^ 2)

    slope = ss_xx > 0 ? ss_xy / ss_xx : 0.0
    intercept = y_mean - slope * x_mean

    # R² 계산
    y_pred = slope .* x .+ intercept
    ss_res = sum((y .- y_pred) .^ 2)
    ss_tot = sum((y .- y_mean) .^ 2)
    r_squared = ss_tot > 0 ? 1 - ss_res / ss_tot : 0.0

    initial = y[1]
    current = y[end]
    total_growth = current - initial
    growth_pct = initial > 0 ? total_growth / initial * 100 : 0

    return DCIRGrowthResult(
        slope,
        initial > 0 ? slope / initial * 100 : 0,
        initial,
        current,
        total_growth,
        growth_pct,
        r_squared
    )
end


"""
    DCIRAnalyzer

DCIR 분석기 구조체
"""
struct DCIRAnalyzer{T<:AbstractFloat}
    cycles::Vector{Int}
    dcir::Vector{T}
    capacity::Union{Vector{T},Nothing}
    temperature::Union{Vector{T},Nothing}
end

function DCIRAnalyzer(cycles::Vector{<:Integer}, dcir::Vector{T};
    capacity::Union{Vector{T},Nothing}=nothing,
    temperature::Union{Vector{T},Nothing}=nothing) where T<:AbstractFloat
    return DCIRAnalyzer{T}(Int.(cycles), dcir, capacity, temperature)
end


"""
    analyze_growth(analyzer::DCIRAnalyzer)

DCIR 성장 분석
"""
function analyze_growth(analyzer::DCIRAnalyzer)
    return calculate_dcir_growth(analyzer.cycles, analyzer.dcir)
end


"""
    dcir_ratio(analyzer::DCIRAnalyzer)

DCIR 비율 (초기 대비)
"""
function dcir_ratio(analyzer::DCIRAnalyzer{T})::Vector{T} where T
    valid_idx = findfirst(x -> !isnan(x), analyzer.dcir)
    if valid_idx === nothing
        return ones(T, length(analyzer.dcir))
    end
    initial = analyzer.dcir[valid_idx]
    return initial > 0 ? analyzer.dcir ./ initial : ones(T, length(analyzer.dcir))
end


"""
    temperature_correction(analyzer::DCIRAnalyzer; 
                           reference_temp=25.0, activation_energy=30000.0)

온도 보정된 DCIR

Arrhenius 기반: DCIR_corr = DCIR * exp(Ea/R * (1/T - 1/T_ref))
"""
function temperature_correction(analyzer::DCIRAnalyzer{T};
    reference_temp::Float64=25.0,
    activation_energy::Float64=30000.0)::Vector{T} where T
    if analyzer.temperature === nothing
        return analyzer.dcir
    end

    R = 8.314  # 기체 상수
    T_ref = reference_temp + 273.15
    T = analyzer.temperature .+ 273.15

    correction = exp.(activation_energy / R .* (1.0 ./ T .- 1.0 / T_ref))

    return analyzer.dcir .* correction
end


"""
    predict_dcir_at_cycle(analyzer::DCIRAnalyzer, target_cycle)

특정 사이클에서의 DCIR 예측
"""
function predict_dcir_at_cycle(analyzer::DCIRAnalyzer{T}, target_cycle::Int)::T where T
    valid_mask = .!isnan.(analyzer.dcir)
    x = Float64.(analyzer.cycles[valid_mask])
    y = Float64.(analyzer.dcir[valid_mask])

    if length(x) < 2
        return y[end]
    end

    x_mean = mean(x)
    y_mean = mean(y)

    ss_xy = sum((x .- x_mean) .* (y .- y_mean))
    ss_xx = sum((x .- x_mean) .^ 2)

    slope = ss_xx > 0 ? ss_xy / ss_xx : 0.0
    intercept = y_mean - slope * x_mean

    return T(slope * target_cycle + intercept)
end
