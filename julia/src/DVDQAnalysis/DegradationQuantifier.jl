"""
DegradationQuantifier.jl - 열화 정량화
=======================================

Python degradation_quantifier.py의 Julia 버전입니다.

열화 메커니즘:
- LAM_PE: 양극 활물질 손실
- LAM_NE: 음극 활물질 손실
- LLI: 리튬 재고 손실
"""

"""
    calculate_lam(mass_ratio; electrode="cathode")

Loss of Active Material (LAM) 계산

수식: LAM = (1 - mass_ratio) × 100%

# Arguments
- `mass_ratio`: 활물질 비율 (0-1)
- `electrode`: "cathode" 또는 "anode"

# Returns
- LAM 값 (%)

# Examples
```julia
lam = calculate_lam(0.95)  # 5% LAM
```
"""
function calculate_lam(mass_ratio::Float64; electrode::String="cathode")
    if mass_ratio < 0 || mass_ratio > 1
        throw(ArgumentError("mass_ratio must be between 0 and 1, got $mass_ratio"))
    end
    return (1 - mass_ratio) * 100
end


"""
    calculate_lli(slip_ca, slip_an; rated_capacity=100.0)

Loss of Lithium Inventory (LLI) 계산

# Arguments
- `slip_ca`: 양극 SOC 슬립
- `slip_an`: 음극 SOC 슬립
- `rated_capacity`: 정격 용량

# Returns
- LLI 값 (%)
"""
function calculate_lli(slip_ca::Float64, slip_an::Float64;
    rated_capacity::Float64=100.0)
    return abs(slip_an - slip_ca) / rated_capacity * 100
end


"""
    quantify_degradation(params::FittingParams; rated_capacity=100.0)

열화 메커니즘 정량화

# Arguments
- `params`: 피팅 파라미터
- `rated_capacity`: 정격 용량

# Returns
- `DegradationMetrics` 구조체
"""
function quantify_degradation(params::FittingParams;
    rated_capacity::Float64=100.0)
    lam_pe = calculate_lam(params.ca_mass)
    lam_ne = calculate_lam(params.an_mass)
    lli = calculate_lli(params.ca_slip, params.an_slip;
        rated_capacity=rated_capacity)

    # 총 용량 손실 추정
    capacity_loss = max(lam_pe, lam_ne)

    return DegradationMetrics(lam_pe, lam_ne, lli, capacity_loss)
end


"""
    generate_random_params(; ca_mass_range=(0.9, 1.0), 
                            ca_slip_range=(-2.0, 2.0),
                            an_mass_range=(0.9, 1.0),
                            an_slip_range=(-2.0, 2.0))

랜덤 파라미터 생성 (Monte Carlo용)
"""
function generate_random_params(;
    ca_mass_range::Tuple{Float64,Float64}=(0.9, 1.0),
    ca_slip_range::Tuple{Float64,Float64}=(-2.0, 2.0),
    an_mass_range::Tuple{Float64,Float64}=(0.9, 1.0),
    an_slip_range::Tuple{Float64,Float64}=(-2.0, 2.0))
    ca_mass = rand() * (ca_mass_range[2] - ca_mass_range[1]) + ca_mass_range[1]
    ca_slip = rand() * (ca_slip_range[2] - ca_slip_range[1]) + ca_slip_range[1]
    an_mass = rand() * (an_mass_range[2] - an_mass_range[1]) + an_mass_range[1]
    an_slip = rand() * (an_slip_range[2] - an_slip_range[1]) + an_slip_range[1]

    return FittingParams(ca_mass=ca_mass, ca_slip=ca_slip,
        an_mass=an_mass, an_slip=an_slip)
end


"""
    analyze_degradation_trend(cycles, lam_pe, lam_ne, lli)

열화 트렌드 분석
"""
function analyze_degradation_trend(cycles::Vector{<:Real},
    lam_pe::Vector{<:Real},
    lam_ne::Vector{<:Real},
    lli::Vector{<:Real})
    x = Float64.(cycles)

    results = Dict{String,Float64}()

    # LAM_PE 트렌드
    if length(lam_pe) > 1
        slope_pe = linreg_slope(x, Float64.(lam_pe))
        results["lam_pe_rate"] = slope_pe
    end

    # LAM_NE 트렌드
    if length(lam_ne) > 1
        slope_ne = linreg_slope(x, Float64.(lam_ne))
        results["lam_ne_rate"] = slope_ne
    end

    # LLI 트렌드
    if length(lli) > 1
        slope_lli = linreg_slope(x, Float64.(lli))
        results["lli_rate"] = slope_lli
    end

    return results
end


"""
    linreg_slope(x, y)

선형 회귀 기울기만 계산
"""
function linreg_slope(x::Vector{Float64}, y::Vector{Float64})
    x_mean = mean(x)
    y_mean = mean(y)

    ss_xy = sum((x .- x_mean) .* (y .- y_mean))
    ss_xx = sum((x .- x_mean) .^ 2)

    return ss_xx > 0 ? ss_xy / ss_xx : 0.0
end
