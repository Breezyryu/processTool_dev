"""
EUPrediction.jl - EU 기준 수명 예측
====================================

Python eu_prediction.py의 Julia 버전입니다.
"""

"""
    EULifeResult

EU 수명 예측 결과
"""
struct EULifeResult
    predicted_cycles_80::Int
    predicted_cycles_70::Int
    meets_eu_requirement::Bool
    required_cycles::Int
    params::ModelParameters
end


"""
    predict_eu_life(cycles, temps, capacities; 
                    required_cycles=1000, operating_temp=298.15)

EU 기준 수명 예측

# Arguments
- `cycles`: 사이클 배열
- `temps`: 온도 배열 (K)
- `capacities`: 용량 배열 (0-1)
- `required_cycles`: EU 요구 사이클 (기본 1000)
- `operating_temp`: 운용 온도 (K)

# Returns
- `EULifeResult` 구조체
"""
function predict_eu_life(cycles::Vector{<:Real},
    temps::Vector{<:Real},
    capacities::Vector{<:Real};
    required_cycles::Int=1000,
    operating_temp::Float64=298.15)

    params = ModelParameters()  # 기본 파라미터 사용

    # 80% SOH까지 사이클 예측
    cycles_to_80 = predict_cycles_to_soh(operating_temp, 0.8, params)

    # 70% SOH까지 사이클 예측
    cycles_to_70 = predict_cycles_to_soh(operating_temp, 0.7, params)

    # EU 기준 충족 여부
    meets_requirement = cycles_to_80 >= required_cycles

    return EULifeResult(
        cycles_to_80,
        cycles_to_70,
        meets_requirement,
        required_cycles,
        params
    )
end


"""
    predict_cycles_to_soh(temp, target_soh, params; max_cycles=10000)

특정 SOH까지의 사이클 수 예측
"""
function predict_cycles_to_soh(temp::Float64, target_soh::Float64,
    params::ModelParameters; max_cycles::Int=10000)
    for cycle in 1:max_cycles
        cap = capacityfit(Float64(cycle), temp, params)
        if cap < target_soh
            return cycle
        end
    end
    return max_cycles
end


"""
    ApprovalLifeResult

승인 수명 예측 결과
"""
struct ApprovalLifeResult
    test_cycles::Int
    equivalent_real_cycles::Int
    predicted_years::Float64
    acceleration_factor::Float64
    meets_requirement::Bool
    required_years::Float64
end


"""
    calculate_arrhenius_factor(test_temp, target_temp; activation_energy=50000)

Arrhenius 가속 계수 계산

수식: AF = exp(Ea/R * (1/T_target - 1/T_test))

# Arguments
- `test_temp`: 시험 온도 (K)
- `target_temp`: 목표 온도 (K)
- `activation_energy`: 활성화 에너지 (J/mol)

# Returns
- 가속 계수
"""
function calculate_arrhenius_factor(test_temp::Float64, target_temp::Float64;
    activation_energy::Float64=50000.0)
    R = 8.314  # 기체 상수 (J/mol·K)
    return exp(activation_energy / R * (1 / target_temp - 1 / test_temp))
end


"""
    predict_approval_life(test_temp, target_temp, test_cycles;
                          activation_energy=50000, cycles_per_year=365, required_years=8)

승인 수명 예측

# Arguments
- `test_temp`: 시험 온도 (K)
- `target_temp`: 목표 운용 온도 (K)
- `test_cycles`: 시험 사이클 수
"""
function predict_approval_life(test_temp::Float64, target_temp::Float64,
    test_cycles::Int;
    activation_energy::Float64=50000.0,
    cycles_per_year::Int=365,
    required_years::Float64=8.0)

    af = calculate_arrhenius_factor(test_temp, target_temp;
        activation_energy=activation_energy)

    equivalent_cycles = Int(round(test_cycles * af))
    predicted_years = equivalent_cycles / cycles_per_year
    meets_requirement = predicted_years >= required_years

    return ApprovalLifeResult(
        test_cycles,
        equivalent_cycles,
        predicted_years,
        af,
        meets_requirement,
        required_years
    )
end
