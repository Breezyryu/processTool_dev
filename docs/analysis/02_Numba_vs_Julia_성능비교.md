# Numba vs Julia 성능 비교 분석

> 작성일: 2026-01-14

---

## 1. 벤치마크 결과 요약

다양한 소스에서 수집한 벤치마크 결과:

| 테스트 케이스 | Numba | Julia | 비고 |
|--------------|-------|-------|------|
| 일반 루프 연산 | 1x | **1.1-1.3x 빠름** | Julia 10-30% 우세 |
| Monte Carlo (π 계산) | 1x | **3x 빠름** | 대규모 반복 연산 |
| 최적화된 코드 재작성 | +20% 개선 | **14x 빠름** | 실제 프로젝트 사례 |
| 마이크로벤치마크 | 1x | **3x 이상 빠름** | SIMD 최적화 차이 |
| 고도 반복 루프 | **비슷~우세** | 비슷 | Numba 선전 영역 |

---

## 2. 기술적 차이점

### 2.1 Numba

```
특징:
├── Python 함수 데코레이터 방식 (@jit, @njit)
├── LLVM 기반 JIT 컴파일
├── NumPy 배열에 최적화
├── 함수 단위 최적화 (로컬 최적화)
├── 병렬화: @parallel, prange 지원
└── 기존 Python 코드베이스와 쉬운 통합
```

### 2.2 Julia

```
특징:
├── 언어 자체가 JIT 컴파일 (네이티브)
├── LLVM 기반 + 전역 최적화
├── Multiple Dispatch 기반 타입 시스템
├── 프로그램 전체에 걸친 최적화 (글로벌 최적화)
├── SIMD 자동 벡터화 우수
└── 메타프로그래밍 & 매크로 지원
```

---

## 3. 성능 차이의 핵심 원인

| 요소 | Numba | Julia |
|------|-------|-------|
| **최적화 범위** | 함수 단위 (로컬) | 프로그램 전체 (글로벌) |
| **SIMD 벡터화** | 제한적 | 자동화 우수 |
| **타입 추론** | NumPy 타입 기반 | 네이티브 타입 시스템 |
| **컴파일 오버헤드** | 첫 호출 시 발생 | 첫 호출 시 발생 |
| **고수준 구조 처리** | 재컴파일 이슈 가능 | 안정적 처리 |

---

## 4. BatteryDataTool 적용 시 예상 성능

### 4.1 curve_fit 기반 비선형 최적화

```python
# 현재 Python 코드 (scipy.optimize.curve_fit)
popt, pcov = curve_fit(capacityfit, (dfall.x, dfall.t), dfall.y, p0, maxfev=100000)
```

| 기술 | 예상 속도 향상 | 구현 난이도 |
|------|---------------|-------------|
| **순수 Python** | 1x (기준) | - |
| **Numba** | **5-20x** | ⭐⭐ (낮음) |
| **Julia (LsqFit.jl)** | **10-50x** | ⭐⭐⭐ (중간) |

### 4.2 Numba 적용 예시

```python
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def capacityfit_numba(x_cycle, x_temp, a, b, b1, c, d, e, f, fd):
    """
    Empirical capacity degradation model - Numba 최적화
    capacity = 1 - exp(a*T + b)*(cycle*fd)^b1 - exp(c*T + d)*(cycle*fd)^(e*T + f)
    """
    n = len(x_cycle)
    result = np.empty(n)
    
    for i in prange(n):  # 병렬 처리
        cycle = x_cycle[i]
        temp = x_temp[i]
        
        # Calendar aging term
        term1 = np.exp(a * temp + b) * (cycle * fd) ** b1
        # Cycle aging term  
        term2 = np.exp(c * temp + d) * (cycle * fd) ** (e * temp + f)
        
        result[i] = 1.0 - term1 - term2
    
    return result
```

### 4.3 Julia 적용 예시

```julia
using LsqFit

function capacityfit(x, p)
    # p = [a, b, b1, c, d, e, f, fd]
    cycle, temp = x[1, :], x[2, :]
    
    term1 = @. exp(p[1] * temp + p[2]) * (cycle * p[8])^p[3]
    term2 = @. exp(p[4] * temp + p[5]) * (cycle * p[8])^(p[6] * temp + p[7])
    
    return @. 1.0 - term1 - term2
end

# 피팅
fit = curve_fit(capacityfit, xdata, ydata, p0)
```

---

## 5. 실용적 비교

### 5.1 종합 비교표

| 기준 | Numba | Julia | 승자 |
|------|-------|-------|------|
| **성능** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Julia |
| **구현 용이성** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Numba |
| **기존 코드 통합** | ⭐⭐⭐⭐⭐ | ⭐⭐ | Numba |
| **학습 곡선** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Numba |
| **장기 유지보수** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 동등 |
| **생태계 성숙도** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Numba (Python) |

### 5.2 예상 속도 개선 (BatteryDataTool 기준)

| 기능 | 현재 (Python) | Numba 적용 | Julia 마이그레이션 |
|------|--------------|-----------|------------------|
| dV/dQ 1회 피팅 | ~5초 | ~0.5초 | ~0.2초 |
| dV/dQ 10회 반복 | ~50초 | ~5초 | ~2초 |
| EU 피팅 (maxfev=100000) | ~30초 | ~3초 | ~1초 |
| 배치 처리 (100 파일) | ~1시간 | ~6분 | ~2분 |

---

## 6. 권장사항

### 6.1 BatteryDataTool 고도화의 경우

**Numba 선택 권장**

이유:
1. ✅ PyBaMM이 Python 전용 (Julia 미지원)
2. ✅ PINN 생태계가 Python 중심 (PyTorch, DeepXDE)
3. ✅ 기존 14,000줄 코드 활용 가능
4. ✅ 점진적 최적화 가능
5. ✅ 5-20배 성능 향상으로 충분

### 6.2 Julia가 더 적합한 경우

- 대규모 ODE/PDE 시뮬레이션이 메인인 경우
- 새로 시작하는 프로젝트
- Julia 생태계 (DifferentialEquations.jl) 활용 필요
- 최대 성능이 필수인 경우

---

## 7. 결론

| 시나리오 | 성능 이득 | 개발 비용 | 권장도 |
|---------|----------|----------|--------|
| 전체 Julia 마이그레이션 | 2-5배 (일부 영역) | **매우 높음** | ❌ |
| 핵심 연산만 Julia | 2-5배 (fitting 영역) | **중간** | ⚠️ |
| **Python + Numba 최적화** | 5-20배 | **낮음** | ✅ |

**최종 권장:** Python 유지 + Numba 최적화가 가장 효율적인 선택
