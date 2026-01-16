"""
CapacityAnalyzer.jl - 용량 분석
================================

Python capacity_analyzer.py의 Julia 버전입니다.
"""

"""
    calculate_capacity_retention(capacity; reference=nothing)

용량 유지율 계산

# Arguments
- `capacity`: 용량 배열
- `reference`: 기준 용량 (없으면 첫 값 사용)

# Returns
- 용량 유지율 배열 (0-1)
"""
function calculate_capacity_retention(capacity::Vector{T};
    reference::Union{T,Nothing}=nothing)::Vector{T} where T<:AbstractFloat
    if reference === nothing
        ref = length(capacity) > 0 ? capacity[1] : one(T)
    else
        ref = reference
    end

    if ref == 0
        return ones(T, length(capacity))
    end

    return capacity ./ ref
end


"""
    CapacityAnalyzer

용량 분석기 구조체
"""
struct CapacityAnalyzer{T<:AbstractFloat}
    cycles::Vector{Int}
    capacity::Vector{T}
    rated_capacity::T
end

function CapacityAnalyzer(cycles::Vector{<:Integer}, capacity::Vector{T};
    rated_capacity::Union{T,Nothing}=nothing) where T<:AbstractFloat
    rc = rated_capacity === nothing ? capacity[1] : rated_capacity
    return CapacityAnalyzer{T}(Int.(cycles), capacity, rc)
end


"""
    retention(analyzer::CapacityAnalyzer)

용량 유지율
"""
function retention(analyzer::CapacityAnalyzer{T})::Vector{T} where T
    return calculate_capacity_retention(analyzer.capacity)
end


"""
    soh(analyzer::CapacityAnalyzer)

SOH (정격 대비)
"""
function soh(analyzer::CapacityAnalyzer{T})::Vector{T} where T
    return analyzer.capacity ./ analyzer.rated_capacity
end


"""
    analyze_fade(analyzer::CapacityAnalyzer)

열화율 분석
"""
function analyze_fade(analyzer::CapacityAnalyzer)
    return calculate_fade_rate(analyzer.cycles, analyzer.capacity)
end


"""
    predict_eol(analyzer::CapacityAnalyzer; threshold=80.0)

EOL 예측
"""
function predict_eol(analyzer::CapacityAnalyzer; threshold::Float64=80.0)
    return predict_eol_cycle(analyzer.cycles, analyzer.capacity;
        eol_threshold=threshold / 100)
end


"""
    get_cycle_at_soh(analyzer::CapacityAnalyzer, target_soh)

특정 SOH에 도달한 사이클 찾기
"""
function get_cycle_at_soh(analyzer::CapacityAnalyzer{T}, target_soh::T)::Union{Int,Nothing} where T
    ret = retention(analyzer) .* 100

    idx = findfirst(x -> x <= target_soh, ret)
    return idx !== nothing ? analyzer.cycles[idx] : nothing
end
