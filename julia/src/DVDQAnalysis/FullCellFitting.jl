"""
FullCellFitting.jl - Full-cell dV/dQ 피팅
==========================================

Python fullcell_fitting.py의 Julia 버전입니다.

핵심 수식:
    V_fullcell(Q) = V_cathode(Q) - V_anode(Q)
    dV/dQ = d(V_fullcell)/dQ
"""

"""
    calculate_dvdq(voltage, capacity; period=1)

dV/dQ 계산 (미분 전압)

# Arguments
- `voltage`: 전압 배열 (V)
- `capacity`: 용량 배열 (mAh 또는 %)
- `period`: 미분 윈도우 크기

# Returns
- dV/dQ 배열
"""
function calculate_dvdq(voltage::Vector{T}, capacity::Vector{T};
    period::Int=1)::Vector{T} where T<:AbstractFloat
    n = length(voltage)
    result = Vector{T}(undef, n)

    @inbounds for i in 1:n
        if i <= period
            result[i] = NaN
        else
            dv = voltage[i] - voltage[i-period]
            dq = capacity[i] - capacity[i-period]
            result[i] = dq != 0 ? dv / dq : NaN
        end
    end

    return result
end


"""
    simulate_fullcell(ca_cap, ca_volt, an_cap, an_volt, params::FittingParams;
                      max_cap=100.0, step=0.1)

Full-cell 전압 시뮬레이션

# Arguments
- `ca_cap, ca_volt`: 양극 용량/전압 데이터
- `an_cap, an_volt`: 음극 용량/전압 데이터
- `params`: 열화 파라미터 (FittingParams)
- `max_cap`: 최대 용량
- `step`: 용량 스텝

# Returns
- (capacity, ca_voltage, an_voltage, full_voltage, full_dvdq) 튜플
"""
function simulate_fullcell(ca_cap::Vector{T}, ca_volt::Vector{T},
    an_cap::Vector{T}, an_volt::Vector{T},
    params::FittingParams;
    max_cap::T=100.0, step::T=0.1) where T<:AbstractFloat

    # 열화 적용
    ca_cap_new = ca_cap .* params.ca_mass .- params.ca_slip
    an_cap_new = an_cap .* params.an_mass .- params.an_slip

    # 시뮬레이션 용량 범위
    sim_capacity = collect(0.0:step:max_cap)
    n = length(sim_capacity)

    sim_ca_volt = Vector{T}(undef, n)
    sim_an_volt = Vector{T}(undef, n)
    sim_full_volt = Vector{T}(undef, n)

    # 보간 수행
    @inbounds for i in 1:n
        q = sim_capacity[i]
        sim_ca_volt[i] = linear_interp(q, ca_cap_new, ca_volt)
        sim_an_volt[i] = linear_interp(q, an_cap_new, an_volt)
        sim_full_volt[i] = sim_ca_volt[i] - sim_an_volt[i]
    end

    # dV/dQ 계산
    sim_full_dvdq = calculate_dvdq(sim_full_volt, sim_capacity)

    return (sim_capacity, sim_ca_volt, sim_an_volt, sim_full_volt, sim_full_dvdq)
end


"""
    linear_interp(x, xp, fp)

선형 보간 (수동 구현)
"""
@inline function linear_interp(x::T, xp::Vector{T}, fp::Vector{T})::T where T<:AbstractFloat
    n = length(xp)

    # 경계 처리
    if x <= xp[1]
        return fp[1]
    elseif x >= xp[n]
        return fp[n]
    end

    # 이진 검색
    lo, hi = 1, n
    while lo < hi - 1
        mid = (lo + hi) ÷ 2
        if xp[mid] <= x
            lo = mid
        else
            hi = mid
        end
    end

    # 선형 보간
    t = (x - xp[lo]) / (xp[hi] - xp[lo])
    return fp[lo] + t * (fp[hi] - fp[lo])
end


"""
    calculate_rmse(exp_voltage, exp_capacity, ca_cap, ca_volt, an_cap, an_volt, params)

시뮬레이션과 실험 데이터 간 RMSE 계산
"""
function calculate_rmse(exp_voltage::Vector{T}, exp_capacity::Vector{T},
    ca_cap::Vector{T}, ca_volt::Vector{T},
    an_cap::Vector{T}, an_volt::Vector{T},
    params::FittingParams) where T<:AbstractFloat

    # 시뮬레이션 수행
    sim_cap, sim_ca, sim_an, sim_full, _ = simulate_fullcell(
        ca_cap, ca_volt, an_cap, an_volt, params;
        max_cap=maximum(exp_capacity)
    )

    # 실험 용량에서 시뮬레이션 전압 보간
    sim_interp = [linear_interp(q, sim_cap, sim_full) for q in exp_capacity]

    # RMSE 계산
    rmse = sqrt(mean((exp_voltage .- sim_interp) .^ 2))

    return rmse
end
