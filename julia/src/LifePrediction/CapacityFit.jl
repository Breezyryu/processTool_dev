"""
CapacityFit.jl - 용량 열화 모델
================================

Python capacity_fit.py의 Julia 버전입니다.

핵심 수식:
    capacity = 1 - exp(a*T + b) * (cycle*fd)^b1 - exp(c*T + d) * (cycle*fd)^(e*T + f)
"""

"""
    capacityfit(cycle, temp, params::ModelParameters)
    capacityfit(cycle, temp, a, b, b1, c, d, e, f, fd)

경험적 용량 열화 모델

# Arguments
- `cycle`: 사이클 수
- `temp`: 온도 (K)
- `params` 또는 개별 파라미터: 피팅 파라미터

# Returns
- 잔존 용량 비율 (0-1)

# Examples
```julia
params = ModelParameters()
capacity = capacityfit(100.0, 298.15, params)
```
"""
function capacityfit(cycle::Float64, temp::Float64, params::ModelParameters)
    return capacityfit(cycle, temp, params.a, params.b, params.b1,
        params.c, params.d, params.e, params.f, params.fd)
end

function capacityfit(cycle::T, temp::T,
    a::T, b::T, b1::T,
    c::T, d::T, e::T,
    f::T, fd::T)::T where T<:AbstractFloat
    # Calendar aging term
    term1 = exp(a * temp + b) * (cycle * fd)^b1

    # Cycle aging term
    term2 = exp(c * temp + d) * (cycle * fd)^(e * temp + f)

    # 잔존 용량
    return 1.0 - term1 - term2
end


"""
    capacityfit_vectorized(cycles, temps, params::ModelParameters)
    capacityfit_vectorized(cycles, temps, a, b, b1, c, d, e, f, fd)

벡터화된 용량 열화 모델 (SIMD 최적화)

# Examples
```julia
cycles = [100.0, 200.0, 300.0]
temps = fill(298.15, 3)
caps = capacityfit_vectorized(cycles, temps, ModelParameters())
```
"""
function capacityfit_vectorized(cycles::Vector{T}, temps::Vector{T},
    params::ModelParameters)::Vector{T} where T<:AbstractFloat
    return capacityfit_vectorized(cycles, temps,
        params.a, params.b, params.b1,
        params.c, params.d, params.e,
        params.f, params.fd)
end

function capacityfit_vectorized(cycles::Vector{T}, temps::Vector{T},
    a::T, b::T, b1::T,
    c::T, d::T, e::T,
    f::T, fd::T)::Vector{T} where T<:AbstractFloat
    n = length(cycles)
    result = Vector{T}(undef, n)

    @inbounds @simd for i in 1:n
        cycle = cycles[i]
        temp = temps[i]

        # Calendar aging term
        term1 = exp(a * temp + b) * (cycle * fd)^b1

        # Cycle aging term
        term2 = exp(c * temp + d) * (cycle * fd)^(e * temp + f)

        result[i] = 1.0 - term1 - term2
    end

    return result
end


"""
    swellingfit(cycle, temp, a, b, b1, c, d, e, f, fd)

스웰링 열화 모델
"""
function swellingfit(cycle::T, temp::T,
    a::T, b::T, b1::T,
    c::T, d::T, e::T,
    f::T, fd::T)::T where T<:AbstractFloat
    term1 = exp(a * temp + b) * (cycle * fd)^b1
    term2 = exp(c * temp + d) * (cycle * fd)^(e * temp + f)
    return term1 + term2
end


"""
    predict_eol_cycle(cycles, capacities; eol_threshold=0.8, max_cycle=10000)

EOL 사이클 예측 (선형 외삽)

# Arguments
- `cycles`: 사이클 배열
- `capacities`: 용량 배열 (0-1 또는 %)
- `eol_threshold`: EOL 기준 (기본값 80%)
- `max_cycle`: 최대 예측 사이클

# Returns
- `EOLPrediction` 구조체
"""
function predict_eol_cycle(cycles::Vector{<:Real}, capacities::Vector{<:Real};
    eol_threshold::Float64=0.8, max_cycle::Int=10000)
    n = length(cycles)

    # 용량 정규화 (% -> 비율)
    if capacities[1] > 10
        cap_norm = capacities ./ capacities[1]
    elseif maximum(capacities) > 1
        cap_norm = capacities ./ 100
    else
        cap_norm = capacities
    end

    current_soh = cap_norm[end] * 100

    # 선형 회귀
    x = Float64.(cycles)
    y = Float64.(cap_norm)

    x_mean = mean(x)
    y_mean = mean(y)

    ss_xy = sum((x .- x_mean) .* (y .- y_mean))
    ss_xx = sum((x .- x_mean) .^ 2)

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # R² 계산
    y_pred = slope .* x .+ intercept
    ss_res = sum((y .- y_pred) .^ 2)
    ss_tot = sum((y .- y_mean) .^ 2)
    r_squared = ss_tot > 0 ? 1 - ss_res / ss_tot : 0.0

    # EOL 예측
    if slope >= 0
        eol_cycle = max_cycle
    else
        eol_cycle = min(Int(round((eol_threshold - intercept) / slope)), max_cycle)
    end

    eol_cycle = max(eol_cycle, cycles[end])

    # 신뢰구간 (단순화)
    conf_low = max(Int(round(eol_cycle * 0.9)), cycles[end])
    conf_high = min(Int(round(eol_cycle * 1.1)), max_cycle)

    return EOLPrediction(
        eol_cycle,
        conf_low,
        conf_high,
        eol_threshold * 100,
        current_soh,
        eol_cycle - cycles[end],
        "linear"
    )
end


"""
    calculate_fade_rate(cycles, capacities)

용량 열화율 계산

# Returns
- `FadeAnalysisResult` 구조체
"""
function calculate_fade_rate(cycles::Vector{<:Real}, capacities::Vector{<:Real})
    n = length(cycles)

    # 용량 정규화
    if capacities[1] > 10
        cap_pct = capacities ./ capacities[1] .* 100
    elseif maximum(capacities) <= 1
        cap_pct = capacities .* 100
    else
        cap_pct = Float64.(capacities)
    end

    # 선형 회귀
    x = Float64.(cycles)
    y = Float64.(cap_pct)

    x_mean = mean(x)
    y_mean = mean(y)

    ss_xy = sum((x .- x_mean) .* (y .- y_mean))
    ss_xx = sum((x .- x_mean) .^ 2)

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # R² 계산
    y_pred = slope .* x .+ intercept
    ss_res = sum((y .- y_pred) .^ 2)
    ss_tot = sum((y .- y_mean) .^ 2)
    r_squared = ss_tot > 0 ? 1 - ss_res / ss_tot : 0.0

    fade_rate = -slope  # 양수로 변환

    return FadeAnalysisResult(
        fade_rate,
        fade_rate * 100,
        r_squared,
        cap_pct[1],
        cap_pct[end],
        cap_pct[1] - cap_pct[end],
        n
    )
end
