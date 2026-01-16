"""
BaseLoader.jl - 데이터 로더 기반
==================================

Python base_loader.py의 Julia 버전입니다.
"""

using DelimitedFiles


"""
    detect_cycler_type(raw_path)

충방전기 타입 자동 감지

Pattern 폴더 존재 → PNE
capacity.log 존재 → TOYO
그 외 → UNKNOWN
"""
function detect_cycler_type(raw_path::String)::CyclerType
    pattern_path = joinpath(raw_path, "Pattern")

    if isdir(pattern_path)
        return PNE
    end

    capacity_log = joinpath(raw_path, "capacity.log")
    if isfile(capacity_log)
        return TOYO
    end

    # 숫자 이름 파일 확인
    for item in readdir(raw_path)
        if all(isdigit, item)
            return TOYO
        end
    end

    return UNKNOWN
end


"""
    extract_capacity_from_path(path)

경로에서 용량 추출 (예: "4500mAh_cell" → 4500.0)
"""
function extract_capacity_from_path(path::String)::Float64
    # mAh 패턴
    m = match(r"(\d+(?:[\-\.])?\d*)\s*mAh"i, path)
    if m !== nothing
        cap_str = replace(m.captures[1], "-" => ".")
        return parse(Float64, cap_str)
    end

    # Ah 패턴
    m = match(r"(\d+(?:[\-\.])?\d*)\s*Ah"i, path)
    if m !== nothing
        cap_str = replace(m.captures[1], "-" => ".")
        return parse(Float64, cap_str) * 1000
    end

    return 0.0
end


"""
    load_cycle_data(raw_path; rated_capacity=nothing, initial_crate=0.2)

사이클 데이터 로드 (자동 충방전기 타입 감지)

# Arguments
- `raw_path`: 데이터 폴더 경로
- `rated_capacity`: 정격 용량 (nothing이면 자동 추출)
- `initial_crate`: 초기 C-rate

# Returns
- `CycleData` 구조체
"""
function load_cycle_data(raw_path::String;
    rated_capacity::Union{Float64,Nothing}=nothing,
    initial_crate::Float64=0.2)::CycleData
    cycler_type = detect_cycler_type(raw_path)

    if cycler_type == TOYO
        return load_toyo_cycle_data(raw_path; rated_capacity=rated_capacity,
            initial_crate=initial_crate)
    elseif cycler_type == PNE
        return load_pne_cycle_data(raw_path; rated_capacity=rated_capacity,
            initial_crate=initial_crate)
    else
        error("Unknown cycler type at: $raw_path")
    end
end


"""
    load_toyo_cycle_data(raw_path; rated_capacity=nothing, initial_crate=0.2)

Toyo 사이클 데이터 로드
"""
function load_toyo_cycle_data(raw_path::String;
    rated_capacity::Union{Float64,Nothing}=nothing,
    initial_crate::Float64=0.2)::CycleData
    cap = rated_capacity === nothing ? extract_capacity_from_path(raw_path) : rated_capacity

    # capacity.log 로드
    capacity_log = joinpath(raw_path, "capacity.log")

    if !isfile(capacity_log)
        error("capacity.log not found in $raw_path")
    end

    # CSV 로드 (간략화된 버전)
    try
        # 실제 구현에서는 CSV.jl 사용 권장
        raw_data = readdlm(capacity_log, ','; skipstart=0)

        # 컬럼 추출 (Toyo 형식에 따라 조정 필요)
        cycles = Int.(raw_data[:, 1])
        charge_cap = Float64.(raw_data[:, 3])
        discharge_cap = Float64.(raw_data[:, 4])

        efficiency = discharge_cap ./ charge_cap

        return CycleData(cycles, charge_cap, discharge_cap, efficiency, cap)
    catch e
        # 더미 데이터 반환 (테스트용)
        @warn "Failed to load Toyo data: $e. Returning dummy data."
        return create_dummy_cycle_data(cap)
    end
end


"""
    load_pne_cycle_data(raw_path; rated_capacity=nothing, initial_crate=0.2)

PNE 사이클 데이터 로드
"""
function load_pne_cycle_data(raw_path::String;
    rated_capacity::Union{Float64,Nothing}=nothing,
    initial_crate::Float64=0.2)::CycleData
    cap = rated_capacity === nothing ? extract_capacity_from_path(raw_path) : rated_capacity

    restore_path = joinpath(raw_path, "Restore")

    if !isdir(restore_path)
        error("Restore folder not found in $raw_path")
    end

    # SaveEndData.csv 찾기
    save_end_file = ""
    for f in readdir(restore_path)
        if occursin("SaveEndData.csv", f)
            save_end_file = joinpath(restore_path, f)
            break
        end
    end

    if save_end_file == ""
        error("SaveEndData.csv not found")
    end

    try
        raw_data = readdlm(save_end_file, ','; skipstart=0)

        # PNE 컬럼 인덱스 (27=cycle, 10=chgCap, 11=dchgCap)
        cycles = Int.(raw_data[:, 28])  # 1-indexed in Julia
        charge_cap = Float64.(raw_data[:, 11])
        discharge_cap = Float64.(raw_data[:, 12])

        efficiency = discharge_cap ./ charge_cap

        return CycleData(cycles, charge_cap, discharge_cap, efficiency, cap)
    catch e
        @warn "Failed to load PNE data: $e. Returning dummy data."
        return create_dummy_cycle_data(cap)
    end
end


"""
    create_dummy_cycle_data(rated_capacity; n_cycles=100)

테스트용 더미 데이터 생성
"""
function create_dummy_cycle_data(rated_capacity::Float64; n_cycles::Int=100)::CycleData
    cycles = collect(1:n_cycles)

    # 선형 열화 시뮬레이션
    fade_rate = 0.0002  # 사이클당 0.02%
    capacity = rated_capacity .* (1 .- fade_rate .* cycles)

    charge = capacity .* 1.01  # 약간 높은 충전
    efficiency = capacity ./ charge

    return CycleData(cycles, charge, capacity, efficiency, rated_capacity)
end
