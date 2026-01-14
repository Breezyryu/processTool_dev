#=
Capacity Degradation Model - Julia Implementation
==================================================

Python/Numba 벤치마크 비교를 위한 Julia 구현입니다.

수식:
    capacity(x, T) = 1 - exp(a*T + b) * (x*fd)^b1 - exp(c*T + d) * (x*fd)^(e*T + f)

Author: Battery Analysis Team
Date: 2026-01-14
=#

module CapacityFit

using BenchmarkTools

export capacityfit, capacityfit_vectorized, run_benchmark

"""
    capacityfit(cycle, temp, a, b, b1, c, d, e, f, fd)

경험적 용량 열화 모델 (스칼라 버전)

# Arguments
- `cycle`: 사이클 수
- `temp`: 온도 (K)
- `a, b, b1, c, d, e, f, fd`: 피팅 파라미터

# Returns
- 잔존 용량 비율 (0-1)
"""
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
    capacityfit_vectorized(cycles, temps, a, b, b1, c, d, e, f, fd)

경험적 용량 열화 모델 (벡터화 버전)

# Arguments
- `cycles`: 사이클 수 배열
- `temps`: 온도 배열 (K)
- `a, b, b1, c, d, e, f, fd`: 피팅 파라미터

# Returns
- 잔존 용량 비율 배열
"""
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
    capacityfit_broadcast(cycles, temps, a, b, b1, c, d, e, f, fd)

브로드캐스팅 기반 구현 (Julia 스타일)
"""
function capacityfit_broadcast(cycles::Vector{T}, temps::Vector{T},
                                a::T, b::T, b1::T,
                                c::T, d::T, e::T,
                                f::T, fd::T)::Vector{T} where T<:AbstractFloat
    term1 = @. exp(a * temps + b) * (cycles * fd)^b1
    term2 = @. exp(c * temps + d) * (cycles * fd)^(e * temps + f)
    return @. 1.0 - term1 - term2
end


"""
    run_benchmark(n_points=10000, n_iterations=100)

성능 벤치마크 실행

# Returns
- 벤치마크 결과 딕셔너리
"""
function run_benchmark(n_points::Int=10000, n_iterations::Int=100)
    # 테스트 데이터 생성
    cycles = rand(Float64, n_points) .* 1000 .+ 1
    temps = rand(Float64, n_points) .* 20 .+ (273.0 + 25.0)
    
    # 파라미터
    a, b, b1 = 0.03, -18.0, 0.7
    c, d, e, f, fd = 2.3, -782.0, -0.28, 96.0, 1.0
    
    println("Julia Capacity Fit Benchmark")
    println("=" ^ 50)
    println("Data points: $n_points")
    println()
    
    # 벡터화 버전 벤치마크
    println("Vectorized (SIMD) version:")
    b1_result = @benchmark capacityfit_vectorized($cycles, $temps, 
                                                   $a, $b, $b1, 
                                                   $c, $d, $e, $f, $fd)
    display(b1_result)
    println()
    
    # 브로드캐스트 버전 벤치마크
    println("Broadcast version:")
    b2_result = @benchmark capacityfit_broadcast($cycles, $temps,
                                                  $a, $b, $b1,
                                                  $c, $d, $e, $f, $fd)
    display(b2_result)
    println()
    
    # 결과 반환
    return Dict(
        "n_points" => n_points,
        "vectorized_mean_ns" => mean(b1_result.times),
        "broadcast_mean_ns" => mean(b2_result.times),
        "vectorized_min_ns" => minimum(b1_result.times),
        "broadcast_min_ns" => minimum(b2_result.times)
    )
end


# 스웰링 모델
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

end  # module

# 직접 실행 시 벤치마크
if abspath(PROGRAM_FILE) == @__FILE__
    using .CapacityFit
    results = run_benchmark()
    println("\nResults: ", results)
end
