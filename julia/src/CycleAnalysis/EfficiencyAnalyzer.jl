"""
EfficiencyAnalyzer.jl - 효율 분석
==================================

Python efficiency_analyzer.py의 Julia 버전입니다.
"""

"""
    calculate_coulombic_efficiency(charge, discharge)

쿨롱 효율 계산

CE = Discharge / Charge
"""
function calculate_coulombic_efficiency(charge::Vector{T},
    discharge::Vector{T})::Vector{T} where T<:AbstractFloat
    n = length(charge)
    result = Vector{T}(undef, n)

    @inbounds @simd for i in 1:n
        result[i] = charge[i] > 0 ? discharge[i] / charge[i] : one(T)
    end

    return result
end


"""
    calculate_energy_efficiency(charge_energy, discharge_energy)

에너지 효율 계산

EE = Discharge Energy / Charge Energy
"""
function calculate_energy_efficiency(charge_energy::Vector{T},
    discharge_energy::Vector{T})::Vector{T} where T<:AbstractFloat
    n = length(charge_energy)
    result = Vector{T}(undef, n)

    @inbounds @simd for i in 1:n
        result[i] = charge_energy[i] > 0 ? discharge_energy[i] / charge_energy[i] : one(T)
    end

    return result
end


"""
    EfficiencyAnalyzer

효율 분석기 구조체
"""
struct EfficiencyAnalyzer{T<:AbstractFloat}
    charge_capacity::Vector{T}
    discharge_capacity::Vector{T}
    charge_energy::Union{Vector{T},Nothing}
    discharge_energy::Union{Vector{T},Nothing}
    cycles::Vector{Int}
end

function EfficiencyAnalyzer(charge::Vector{T}, discharge::Vector{T};
    charge_energy::Union{Vector{T},Nothing}=nothing,
    discharge_energy::Union{Vector{T},Nothing}=nothing,
    cycles::Union{Vector{<:Integer},Nothing}=nothing) where T<:AbstractFloat
    cyc = cycles === nothing ? collect(1:length(charge)) : Int.(cycles)
    return EfficiencyAnalyzer{T}(charge, discharge, charge_energy, discharge_energy, cyc)
end


"""
    coulombic_efficiency(analyzer::EfficiencyAnalyzer)

쿨롱 효율 계산
"""
function coulombic_efficiency(analyzer::EfficiencyAnalyzer{T})::Vector{T} where T
    return calculate_coulombic_efficiency(analyzer.charge_capacity,
        analyzer.discharge_capacity)
end


"""
    energy_efficiency(analyzer::EfficiencyAnalyzer)

에너지 효율 계산
"""
function energy_efficiency(analyzer::EfficiencyAnalyzer{T})::Union{Vector{T},Nothing} where T
    if analyzer.charge_energy !== nothing && analyzer.discharge_energy !== nothing
        return calculate_energy_efficiency(analyzer.charge_energy,
            analyzer.discharge_energy)
    end
    return nothing
end


"""
    EfficiencyStats

효율 통계
"""
struct EfficiencyStats
    mean::Float64
    std::Float64
    min::Float64
    max::Float64
    median::Float64
end


"""
    coulombic_efficiency_stats(analyzer)

쿨롱 효율 통계 계산
"""
function coulombic_efficiency_stats(analyzer::EfficiencyAnalyzer)
    ce = coulombic_efficiency(analyzer)
    return EfficiencyStats(
        mean(ce),
        std(ce),
        minimum(ce),
        maximum(ce),
        median(ce)
    )
end


"""
    detect_anomalies(analyzer; threshold=3.0)

Z-score 기반 이상치 탐지

# Returns
- 이상치 사이클 배열
"""
function detect_anomalies(analyzer::EfficiencyAnalyzer{T};
    threshold::Float64=3.0)::Vector{Int} where T
    ce = coulombic_efficiency(analyzer)
    m = mean(ce)
    s = std(ce)

    if s == 0
        return Int[]
    end

    z_scores = abs.(ce .- m) ./ s
    anomaly_indices = findall(z -> z > threshold, z_scores)

    return analyzer.cycles[anomaly_indices]
end
