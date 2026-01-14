#=
Full-Cell dV/dQ Simulation - Julia Implementation  
===================================================

Python/Numba 벤치마크 비교를 위한 Julia 구현입니다.

수식:
    V_fullcell(Q) = V_cathode(Q + slip_ca) * mass_ca - V_anode(Q + slip_an) * mass_an
    dV/dQ = d(V_fullcell)/dQ

Author: Battery Analysis Team
Date: 2026-01-14
=#

module FullCellSimulation

using BenchmarkTools
using Interpolations

export simulate_fullcell, calculate_dvdq, run_benchmark

"""
    calculate_dvdq(voltage, capacity, period=1)

dV/dQ 계산

# Arguments
- `voltage`: 전압 배열
- `capacity`: 용량 배열
- `period`: 미분 윈도우

# Returns
- dV/dQ 배열
"""
function calculate_dvdq(voltage::Vector{T}, 
                        capacity::Vector{T}, 
                        period::Int=1)::Vector{T} where T<:AbstractFloat
    n = length(voltage)
    result = Vector{T}(undef, n)
    
    @inbounds for i in 1:n
        if i <= period
            result[i] = NaN
        else
            dv = voltage[i] - voltage[i - period]
            dq = capacity[i] - capacity[i - period]
            result[i] = dq != 0 ? dv / dq : NaN
        end
    end
    
    return result
end


"""
    simulate_fullcell(ca_cap, ca_volt, an_cap, an_volt, 
                      ca_mass, ca_slip, an_mass, an_slip, 
                      max_cap, step=0.1)

Full-cell 전압 시뮬레이션

# Arguments
- `ca_cap, ca_volt`: 양극 용량/전압 데이터
- `an_cap, an_volt`: 음극 용량/전압 데이터
- `ca_mass, ca_slip`: 양극 열화 파라미터
- `an_mass, an_slip`: 음극 열화 파라미터
- `max_cap`: 최대 용량
- `step`: 용량 스텝

# Returns
- (capacity, ca_volt, an_volt, full_volt) 튜플
"""
function simulate_fullcell(ca_cap::Vector{T}, ca_volt::Vector{T},
                           an_cap::Vector{T}, an_volt::Vector{T},
                           ca_mass::T, ca_slip::T,
                           an_mass::T, an_slip::T,
                           max_cap::T, step::T=0.1) where T<:AbstractFloat
    # 열화 적용
    ca_cap_new = ca_cap .* ca_mass .- ca_slip
    an_cap_new = an_cap .* an_mass .- an_slip
    
    # 시뮬레이션 용량 범위
    sim_capacity = collect(0.0:step:max_cap)
    n = length(sim_capacity)
    
    # 보간 생성
    ca_interp = LinearInterpolation(ca_cap_new, ca_volt, extrapolation_bc=Flat())
    an_interp = LinearInterpolation(an_cap_new, an_volt, extrapolation_bc=Flat())
    
    # 보간 수행
    sim_ca_volt = ca_interp.(sim_capacity)
    sim_an_volt = an_interp.(sim_capacity)
    
    # Full-cell 전압
    sim_full_volt = sim_ca_volt .- sim_an_volt
    
    return (sim_capacity, sim_ca_volt, sim_an_volt, sim_full_volt)
end


"""
    simulate_fullcell_fast(ca_cap, ca_volt, an_cap, an_volt,
                           ca_mass, ca_slip, an_mass, an_slip,
                           sim_capacity)

최적화된 Full-cell 시뮬레이션 (사전 할당된 용량 배열 사용)
"""
function simulate_fullcell_fast(ca_cap::Vector{T}, ca_volt::Vector{T},
                                 an_cap::Vector{T}, an_volt::Vector{T},
                                 ca_mass::T, ca_slip::T,
                                 an_mass::T, an_slip::T,
                                 sim_capacity::Vector{T})::Tuple{Vector{T}, Vector{T}, Vector{T}} where T<:AbstractFloat
    n = length(sim_capacity)
    sim_ca_volt = Vector{T}(undef, n)
    sim_an_volt = Vector{T}(undef, n)
    sim_full_volt = Vector{T}(undef, n)
    
    # 열화 적용된 용량 (미리 계산)
    ca_cap_new = ca_cap .* ca_mass .- ca_slip
    an_cap_new = an_cap .* an_mass .- an_slip
    
    # 선형 보간 (수동 구현 - 최적화)
    @inbounds for i in 1:n
        q = sim_capacity[i]
        
        # 양극 보간
        sim_ca_volt[i] = linear_interp(q, ca_cap_new, ca_volt)
        
        # 음극 보간
        sim_an_volt[i] = linear_interp(q, an_cap_new, an_volt)
        
        # Full-cell
        sim_full_volt[i] = sim_ca_volt[i] - sim_an_volt[i]
    end
    
    return (sim_ca_volt, sim_an_volt, sim_full_volt)
end


"""
    linear_interp(x, xp, fp)

수동 선형 보간 (성능 최적화)
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
    run_benchmark(n_points=1000)

벤치마크 실행
"""
function run_benchmark(n_points::Int=1000)
    # 테스트 데이터 생성
    capacity = collect(0.0:0.1:100.0)
    n_half = length(capacity)
    
    # 양극/음극 프로파일 (예시)
    ca_volt = 4.2 .- 0.5 .* (1 .- capacity ./ 100) .+ 0.1 .* sin.(π .* capacity ./ 100)
    an_volt = 0.1 .+ 0.2 .* (1 .- capacity ./ 100) .- 0.05 .* sin.(3π .* capacity ./ 100)
    
    # 파라미터
    ca_mass, ca_slip = 0.95, 0.5
    an_mass, an_slip = 0.98, 0.3
    
    # 시뮬레이션 용량 범위
    sim_capacity = collect(0.0:0.1:80.0)
    
    println("Julia Full-Cell Simulation Benchmark")
    println("=" ^ 50)
    println("Half-cell data points: $n_half")
    println("Simulation points: $(length(sim_capacity))")
    println()
    
    # 빠른 버전 벤치마크
    println("Fast simulation (manual interp):")
    b_result = @benchmark simulate_fullcell_fast($capacity, $ca_volt,
                                                  $capacity, $an_volt,
                                                  $ca_mass, $ca_slip,
                                                  $an_mass, $an_slip,
                                                  $sim_capacity)
    display(b_result)
    println()
    
    # dV/dQ 계산 벤치마크
    println("dV/dQ calculation:")
    _, _, _, full_volt = simulate_fullcell(capacity, ca_volt, capacity, an_volt,
                                           ca_mass, ca_slip, an_mass, an_slip,
                                           80.0, 0.1)
    b_dvdq = @benchmark calculate_dvdq($full_volt, $sim_capacity, 1)
    display(b_dvdq)
    
    return Dict(
        "simulation_mean_ns" => mean(b_result.times),
        "dvdq_mean_ns" => mean(b_dvdq.times)
    )
end

end  # module

# 직접 실행 시 벤치마크
if abspath(PROGRAM_FILE) == @__FILE__
    using .FullCellSimulation
    results = run_benchmark()
    println("\nResults: ", results)
end
