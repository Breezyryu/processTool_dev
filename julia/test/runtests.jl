"""
BatteryAnalysis.jl 테스트
==========================

실행: julia test/runtests.jl
"""

using Test

# 모듈 경로 추가
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

include("../src/Types.jl")
include("../src/LifePrediction/CapacityFit.jl")
include("../src/LifePrediction/EUPrediction.jl")
include("../src/DVDQAnalysis/FullCellFitting.jl")
include("../src/DVDQAnalysis/DegradationQuantifier.jl")
include("../src/CycleAnalysis/CapacityAnalyzer.jl")
include("../src/CycleAnalysis/EfficiencyAnalyzer.jl")
include("../src/CycleAnalysis/DCIRAnalyzer.jl")
include("../src/DataLoader/BaseLoader.jl")


@testset "BatteryAnalysis Tests" begin

    @testset "Types" begin
        @testset "CycleData" begin
            data = CycleData(
                [1, 2, 3],
                [100.0, 99.0, 98.0],
                [99.0, 98.0, 97.0],
                [0.99, 0.99, 0.99],
                100.0
            )

            @test length(data.cycle) == 3
            @test data.rated_capacity == 100.0

            ret = capacity_retention(data)
            @test ret[1] ≈ 1.0
            @test ret[3] ≈ 97 / 99 atol = 0.001

            s = soh(data)
            @test s[1] ≈ 0.99
        end

        @testset "ModelParameters" begin
            params = ModelParameters()
            @test params.a == 0.03
            @test params.fd == 1.0
        end

        @testset "FittingParams" begin
            fp = FittingParams(ca_mass=0.95, an_mass=0.98)
            @test fp.ca_mass == 0.95
            @test fp.ca_slip == 0.0
        end
    end


    @testset "Life Prediction" begin
        @testset "capacityfit" begin
            params = ModelParameters()

            cap = capacityfit(100.0, 298.15, params)
            @test cap < 1.0
            @test cap > 0.9

            # 사이클 증가 → 용량 감소
            cap2 = capacityfit(500.0, 298.15, params)
            @test cap2 < cap
        end

        @testset "capacityfit_vectorized" begin
            params = ModelParameters()
            cycles = [100.0, 200.0, 300.0]
            temps = fill(298.15, 3)

            caps = capacityfit_vectorized(cycles, temps, params)

            @test length(caps) == 3
            @test caps[1] > caps[3]  # 열화
        end

        @testset "predict_eol_cycle" begin
            cycles = [0, 100, 200, 300, 400]
            capacities = [100.0, 98.0, 96.0, 94.0, 92.0]

            result = predict_eol_cycle(cycles, capacities; eol_threshold=0.8)

            @test result.predicted_cycle > 400
            @test result.eol_threshold == 80.0
            @test result.remaining_cycles > 0
        end

        @testset "calculate_fade_rate" begin
            cycles = [0, 100, 200, 300, 400]
            capacities = [100.0, 98.0, 96.0, 94.0, 92.0]

            result = calculate_fade_rate(cycles, capacities)

            @test result.fade_rate_per_100_cycles ≈ 2.0 atol = 0.01
            @test result.total_fade ≈ 8.0 atol = 0.1
        end
    end


    @testset "dV/dQ Analysis" begin
        @testset "calculate_dvdq" begin
            voltage = [3.0, 3.2, 3.4, 3.6, 3.8]
            capacity = [0.0, 25.0, 50.0, 75.0, 100.0]

            dvdq = calculate_dvdq(voltage, capacity)

            @test length(dvdq) == 5
            @test isnan(dvdq[1])
            @test dvdq[2] ≈ 0.2 / 25 atol = 0.001
        end

        @testset "calculate_lam" begin
            @test calculate_lam(0.95) ≈ 5.0
            @test calculate_lam(1.0) ≈ 0.0

            @test_throws ArgumentError calculate_lam(1.5)
            @test_throws ArgumentError calculate_lam(-0.1)
        end

        @testset "calculate_lli" begin
            lli = calculate_lli(0.5, 0.3; rated_capacity=100.0)
            @test lli ≈ 0.2
        end

        @testset "simulate_fullcell" begin
            ca_cap = collect(0.0:10.0:100.0)
            ca_volt = 4.2 .- 0.5 .* (1 .- ca_cap ./ 100)
            an_cap = collect(0.0:10.0:100.0)
            an_volt = 0.1 .+ 0.2 .* (1 .- an_cap ./ 100)

            params = FittingParams(ca_mass=0.95, an_mass=0.98)

            cap, ca_v, an_v, full_v, dvdq = simulate_fullcell(
                ca_cap, ca_volt, an_cap, an_volt, params
            )

            @test length(cap) > 0
            @test length(full_v) == length(cap)
        end
    end


    @testset "Cycle Analysis" begin
        @testset "calculate_capacity_retention" begin
            capacity = [100.0, 95.0, 90.0, 85.0, 80.0]
            ret = calculate_capacity_retention(capacity)

            @test ret[1] == 1.0
            @test ret[end] == 0.8
        end

        @testset "CapacityAnalyzer" begin
            cycles = collect(1:100)
            capacity = 100.0 .* (1 .- 0.001 .* cycles)

            analyzer = CapacityAnalyzer(cycles, capacity; rated_capacity=100.0)

            @test length(retention(analyzer)) == 100
            @test all(soh(analyzer) .<= 1.0)
        end

        @testset "calculate_coulombic_efficiency" begin
            charge = [100.0, 100.0, 100.0]
            discharge = [99.0, 98.0, 97.0]

            ce = calculate_coulombic_efficiency(charge, discharge)

            @test ce[1] == 0.99
            @test ce[3] == 0.97
        end

        @testset "calculate_dcir_growth" begin
            cycles = [0, 100, 200, 300, 400]
            dcir = [10.0, 10.5, 11.0, 11.5, 12.0]

            result = calculate_dcir_growth(cycles, dcir)

            @test result.initial_dcir == 10.0
            @test result.current_dcir == 12.0
            @test result.total_growth_percent ≈ 20.0 atol = 0.1
        end
    end


    @testset "Data Loader" begin
        @testset "extract_capacity_from_path" begin
            @test extract_capacity_from_path("/path/4500mAh_cell01") == 4500.0
            @test extract_capacity_from_path("/path/4-5Ah_test") == 4500.0
            @test extract_capacity_from_path("/path/no_capacity") == 0.0
        end

        @testset "create_dummy_cycle_data" begin
            data = create_dummy_cycle_data(4500.0; n_cycles=50)

            @test length(data.cycle) == 50
            @test data.rated_capacity == 4500.0
            @test data.discharge_capacity[1] > data.discharge_capacity[end]
        end
    end

end

println("\n✅ All tests completed!")
