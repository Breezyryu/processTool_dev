# Minimal test script
println("Starting minimal BatteryAnalysis tests...")

# Load core modules
include("src/Types.jl")
println("✅ Types.jl loaded")

include("src/LifePrediction/CapacityFit.jl")
println("✅ CapacityFit.jl loaded")

include("src/DVDQAnalysis/DegradationQuantifier.jl")
println("✅ DegradationQuantifier.jl loaded")

include("src/CycleAnalysis/CapacityAnalyzer.jl")
println("✅ CapacityAnalyzer.jl loaded")

include("src/CycleAnalysis/EfficiencyAnalyzer.jl")
println("✅ EfficiencyAnalyzer.jl loaded")

# Basic tests
println("\n--- Running basic tests ---")

# 1. Test capacityfit
params = ModelParameters()
cap = capacityfit(100.0, 298.15, params)
println("1. capacityfit(100, 298.15) = ", round(cap, digits=4))
println("   Result: ", cap < 1.0 && cap > 0.9 ? "✅ PASS" : "❌ FAIL")

# 2. Test LAM
lam = calculate_lam(0.95)
println("2. calculate_lam(0.95) = ", lam)
println("   Result: ", lam ≈ 5.0 ? "✅ PASS" : "❌ FAIL")

# 3. Test LLI
lli = calculate_lli(0.5, 0.3; rated_capacity=100.0)
println("3. calculate_lli(0.5, 0.3) = ", lli)
println("   Result: ", lli ≈ 0.2 ? "✅ PASS" : "❌ FAIL")

# 4. Test capacity retention
ret = calculate_capacity_retention([100.0, 95.0, 90.0])
println("4. capacity_retention = ", ret)
println("   Result: ", ret[1] == 1.0 && ret[3] == 0.9 ? "✅ PASS" : "❌ FAIL")

# 5. Test coulombic efficiency
ce = calculate_coulombic_efficiency([100.0, 100.0], [99.0, 98.0])
println("5. coulombic_efficiency = ", ce)
println("   Result: ", ce[1] == 0.99 ? "✅ PASS" : "❌ FAIL")

# 6. Test vectorized capacityfit
cycles = [100.0, 200.0, 300.0]
temps = fill(298.15, 3)
caps = capacityfit_vectorized(cycles, temps, params)
println("6. capacityfit_vectorized = ", round.(caps, digits=4))
println("   Result: ", caps[1] > caps[3] ? "✅ PASS" : "❌ FAIL")

println("\n✅ All tests completed!")
