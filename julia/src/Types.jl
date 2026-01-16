"""
Types.jl - 공통 데이터 타입 정의
================================
"""

# 충방전기 타입
@enum CyclerType begin
    PNE
    TOYO
    UNKNOWN
end

"""
    CycleData

사이클 데이터 구조체

# Fields
- `cycle`: 사이클 번호 배열
- `charge_capacity`: 충전 용량 (mAh)
- `discharge_capacity`: 방전 용량 (mAh)
- `efficiency`: 쿨롱 효율
- `rated_capacity`: 정격 용량 (mAh)
- `temperature`: 온도 (°C) [optional]
- `dcir`: DC 내부저항 (mΩ) [optional]
"""
struct CycleData
    cycle::Vector{Int}
    charge_capacity::Vector{Float64}
    discharge_capacity::Vector{Float64}
    efficiency::Vector{Float64}
    rated_capacity::Float64
    temperature::Union{Vector{Float64},Nothing}
    dcir::Union{Vector{Float64},Nothing}

    # 기본 생성자
    function CycleData(cycle, charge_capacity, discharge_capacity, efficiency,
        rated_capacity; temperature=nothing, dcir=nothing)
        new(cycle, charge_capacity, discharge_capacity, efficiency,
            rated_capacity, temperature, dcir)
    end
end

"""
    capacity_retention(data::CycleData)

용량 유지율 계산 (첫 사이클 대비)
"""
function capacity_retention(data::CycleData)
    if length(data.discharge_capacity) > 0 && data.discharge_capacity[1] > 0
        return data.discharge_capacity ./ data.discharge_capacity[1]
    end
    return ones(length(data.discharge_capacity))
end

"""
    soh(data::CycleData)

SOH 계산 (정격 용량 대비)
"""
function soh(data::CycleData)
    return data.discharge_capacity ./ data.rated_capacity
end


"""
    ProfileData

프로파일 데이터 구조체

# Fields
- `time`: 시간 (초)
- `voltage`: 전압 (V)
- `current`: 전류 (mA)
- `capacity`: 용량 (mAh)
- `temperature`: 온도 (°C) [optional]
- `condition`: 상태 (1=충전, 2=방전, 3=휴지) [optional]
"""
struct ProfileData
    time::Vector{Float64}
    voltage::Vector{Float64}
    current::Vector{Float64}
    capacity::Vector{Float64}
    temperature::Union{Vector{Float64},Nothing}
    condition::Union{Vector{Int},Nothing}
    cycle::Union{Int,Nothing}

    function ProfileData(time, voltage, current, capacity;
        temperature=nothing, condition=nothing, cycle=nothing)
        new(time, voltage, current, capacity, temperature, condition, cycle)
    end
end


"""
    FadeAnalysisResult

열화 분석 결과 구조체
"""
struct FadeAnalysisResult
    fade_rate_per_cycle::Float64
    fade_rate_per_100_cycles::Float64
    r_squared::Float64
    initial_capacity::Float64
    current_capacity::Float64
    total_fade::Float64
    cycles_analyzed::Int
end


"""
    EOLPrediction

EOL 예측 결과 구조체
"""
struct EOLPrediction
    predicted_cycle::Int
    confidence_low::Int
    confidence_high::Int
    eol_threshold::Float64
    current_soh::Float64
    remaining_cycles::Int
    method::String
end


"""
    DegradationMetrics

열화 지표 구조체
"""
struct DegradationMetrics
    lam_pe::Float64   # 양극 활물질 손실 (%)
    lam_ne::Float64   # 음극 활물질 손실 (%)
    lli::Float64      # 리튬 재고 손실 (%)
    capacity_loss::Float64  # 총 용량 손실 (%)
end


"""
    FittingParams

dV/dQ 피팅 파라미터 구조체
"""
struct FittingParams
    ca_mass::Float64
    ca_slip::Float64
    an_mass::Float64
    an_slip::Float64

    FittingParams(; ca_mass=1.0, ca_slip=0.0, an_mass=1.0, an_slip=0.0) =
        new(ca_mass, ca_slip, an_mass, an_slip)
end


"""
    ModelParameters

용량 열화 모델 파라미터
"""
struct ModelParameters
    a::Float64
    b::Float64
    b1::Float64
    c::Float64
    d::Float64
    e::Float64
    f::Float64
    fd::Float64

    # 기본값 생성자
    ModelParameters() = new(0.03, -18.0, 0.7, 2.3, -782.0, -0.28, 96.0, 1.0)
    ModelParameters(a, b, b1, c, d, e, f, fd) = new(a, b, b1, c, d, e, f, fd)
end
