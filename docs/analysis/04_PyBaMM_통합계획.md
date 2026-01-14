# PyBaMM 통합 계획

> 작성일: 2026-01-14

---

## 1. PyBaMM 개요

### 1.1 PyBaMM이란?

**PyBaMM (Python Battery Mathematical Modelling)**은 물리 기반 배터리 시뮬레이션의 표준 라이브러리.

- **개발:** Oxford University 주도
- **라이선스:** BSD 3-Clause
- **특징:** 모듈형, 확장 가능, 연구/산업 모두 활용

### 1.2 핵심 기능

```
PyBaMM 기능
├── 전기화학 모델
│   ├── SPM (Single Particle Model)
│   ├── SPMe (SPM with Electrolyte)
│   ├── DFN (Doyle-Fuller-Newman, P2D)
│   └── 사용자 정의 모델
│
├── 열화 모델 (서브모델)
│   ├── SEI 성장 (여러 형태)
│   ├── 리튬 플레이팅
│   ├── LAM (Active Material Loss)
│   ├── 전해질 분해
│   └── 크랙킹
│
├── 실험 프로토콜
│   ├── CCCV 충전
│   ├── 방전 프로파일
│   ├── GITT, HPPC
│   └── 사용자 정의 프로토콜
│
├── 파라미터
│   ├── 기본 파라미터 셋 (Chen 2020, etc.)
│   ├── 사용자 정의 파라미터
│   └── 파라미터 추정 (fitting)
│
└── 솔버
    ├── CasADi (기본, 빠름)
    ├── IDA (DAE 솔버)
    └── JAX (GPU 가속)
```

---

## 2. BatteryDataTool 통합 전략

### 2.1 통합 목표

| 목표 | 설명 | 우선순위 |
|------|------|----------|
| 물리 기반 예측 | 경험적 모델 보완 | 🔴 높음 |
| 파라미터 추정 | 실험 데이터로 파라미터 피팅 | 🔴 높음 |
| What-if 분석 | 다양한 조건 시뮬레이션 | 🟡 중간 |
| 가상 데이터 생성 | PINN 학습용 데이터 증강 | 🟡 중간 |

### 2.2 통합 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                BatteryDataTool v2                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  실험 데이터  │    │  PyBaMM     │    │  분석 결과   │  │
│  │  (Cycle,     │───▶│  시뮬레이션  │───▶│  (비교,      │  │
│  │   Profile)   │    │             │    │   파라미터)  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                 │                   │          │
│         ▼                 ▼                   ▼          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              PyBaMM Wrapper Module                │   │
│  ├──────────────────────────────────────────────────┤   │
│  │  • 모델 설정 (SPM, DFN)                           │   │
│  │  • 파라미터 관리                                   │   │
│  │  • 실험 프로토콜 생성                              │   │
│  │  • 시뮬레이션 실행                                 │   │
│  │  • 결과 비교/피팅                                  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 구현 계획

### 3.1 모듈 구조

```
battery_analysis_v2/
└── pybamm_integration/
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   ├── spm_model.py          # SPM 래퍼
    │   ├── dfn_model.py          # DFN 래퍼
    │   └── degradation_models.py # 열화 모델 설정
    │
    ├── parameters/
    │   ├── __init__.py
    │   ├── parameter_sets.py     # 기본 파라미터 셋
    │   ├── custom_parameters.py  # 사용자 정의 파라미터
    │   └── parameter_fitting.py  # 파라미터 추정
    │
    ├── experiments/
    │   ├── __init__.py
    │   ├── protocols.py          # 실험 프로토콜 정의
    │   └── builder.py            # 프로토콜 빌더
    │
    ├── simulations/
    │   ├── __init__.py
    │   ├── runner.py             # 시뮬레이션 실행
    │   └── comparison.py         # 실험 vs 시뮬레이션 비교
    │
    └── utils/
        ├── __init__.py
        └── converters.py         # 데이터 변환 유틸
```

### 3.2 핵심 클래스 설계

```python
# pybamm_integration/models/spm_model.py

import pybamm

class SPMModelWrapper:
    """Single Particle Model 래퍼"""
    
    def __init__(self, parameter_set="Chen2020"):
        """
        Args:
            parameter_set: 파라미터 셋 이름 또는 dict
        """
        self.model = pybamm.lithium_ion.SPM()
        self.parameter_values = pybamm.ParameterValues(parameter_set)
        self.solver = pybamm.CasadiSolver()
        self.simulation = None
    
    def add_degradation(self, mechanisms: list):
        """
        열화 메커니즘 추가
        
        Args:
            mechanisms: ["SEI", "LAM", "lithium_plating"] 등
        """
        options = {}
        if "SEI" in mechanisms:
            options["SEI"] = "solvent-diffusion limited"
        if "LAM" in mechanisms:
            options["loss of active material"] = "stress-driven"
        if "lithium_plating" in mechanisms:
            options["lithium plating"] = "irreversible"
        
        self.model = pybamm.lithium_ion.SPM(options)
    
    def set_experiment(self, protocol):
        """실험 프로토콜 설정"""
        self.experiment = pybamm.Experiment(protocol)
    
    def run(self, t_eval=None):
        """시뮬레이션 실행"""
        self.simulation = pybamm.Simulation(
            self.model,
            parameter_values=self.parameter_values,
            experiment=self.experiment,
            solver=self.solver
        )
        return self.simulation.solve(t_eval=t_eval)
    
    def get_capacity_fade(self, solution):
        """용량 열화 추출"""
        return solution["Discharge capacity [A.h]"].entries
    
    def compare_with_experiment(self, exp_data, sim_solution):
        """실험 데이터와 시뮬레이션 비교"""
        # 보간 및 비교 로직
        pass
```

```python
# pybamm_integration/parameters/parameter_fitting.py

import pybamm
import pybop  # PyBOP: PyBaMM Optimization

class ParameterEstimator:
    """실험 데이터로 PyBaMM 파라미터 추정"""
    
    def __init__(self, model_wrapper):
        self.model = model_wrapper
    
    def fit_capacity_fade(self, exp_cycles, exp_capacity, 
                          fit_params=None):
        """
        용량 데이터로 열화 파라미터 피팅
        
        Args:
            exp_cycles: 사이클 수 배열
            exp_capacity: 용량 배열
            fit_params: 피팅할 파라미터 리스트
        """
        if fit_params is None:
            fit_params = [
                "SEI reaction exchange current density [A.m-2]",
                "SEI kinetic rate constant [m.s-1]"
            ]
        
        # PyBOP 또는 scipy로 최적화
        # ...
        
        return optimized_params
    
    def fit_ocv_curve(self, soc, ocv, electrode="positive"):
        """OCV 곡선으로 전극 열역학 파라미터 피팅"""
        pass
```

### 3.3 사용 예시

```python
# 사용 예시
from battery_analysis_v2.pybamm_integration import SPMModelWrapper

# 1. 모델 설정
model = SPMModelWrapper(parameter_set="Chen2020")
model.add_degradation(["SEI", "LAM"])

# 2. 실험 프로토콜 설정
protocol = [
    "Discharge at 1C until 2.5V",
    "Rest for 10 minutes",
    "Charge at 1C until 4.2V",
    "Hold at 4.2V until C/50",
    "Rest for 10 minutes"
] * 100  # 100 사이클

model.set_experiment(protocol)

# 3. 시뮬레이션 실행
solution = model.run()

# 4. 용량 열화 추출
capacity = model.get_capacity_fade(solution)

# 5. 실험 데이터와 비교
comparison = model.compare_with_experiment(exp_data, solution)
```

---

## 4. 핵심 기능 상세

### 4.1 열화 시뮬레이션

**지원 열화 메커니즘:**

| 메커니즘 | PyBaMM 옵션 | 파라미터 |
|---------|------------|---------|
| SEI 성장 | `"SEI": "solvent-diffusion limited"` | k_SEI, D_SEI |
| 리튬 플레이팅 | `"lithium plating": "irreversible"` | i_0_plating |
| 양극 LAM | `"loss of active material": "stress-driven"` | β_LAM |
| 음극 LAM | 동일 | 동일 |
| 크랙킹 | 서브모델 추가 | 크랙 상수 |

### 4.2 파라미터 추정

**워크플로우:**

```
1. 실험 데이터 준비
   ├── 사이클 데이터 (용량 vs 사이클)
   ├── EIS 데이터 (옵션)
   └── OCV 데이터 (옵션)

2. 피팅 파라미터 선택
   ├── 초기 파라미터 (기본 셋)
   └── 피팅 대상 파라미터 지정

3. 최적화 실행
   ├── 목적함수: RMSE(실험, 시뮬레이션)
   └── 알고리즘: L-BFGS-B, Differential Evolution

4. 결과 검증
   ├── 피팅 품질 확인
   └── 다른 데이터셋으로 검증
```

### 4.3 시뮬레이션 vs 실험 비교

```python
def compare_simulation(exp_data, sim_solution):
    """
    실험 데이터와 시뮬레이션 결과 비교
    
    Returns:
        dict: {
            'rmse': RMSE 값,
            'r2': R² 값,
            'residuals': 잔차,
            'plot': 비교 그래프
        }
    """
    # 시간/사이클 축 정렬
    # 보간
    # 메트릭 계산
    # 시각화
    pass
```

---

## 5. 성능 고려사항

### 5.1 시뮬레이션 속도

| 모델 | 1 사이클 | 100 사이클 | GPU 가속 |
|------|---------|-----------|---------|
| **SPM** | ~0.1초 | ~10초 | 가능 (JAX) |
| **SPMe** | ~0.5초 | ~50초 | 가능 |
| **DFN** | ~5초 | ~10분 | 가능 |

### 5.2 최적화 전략

1. **SPM 우선 사용** - 빠른 프로토타이핑
2. **JAX 백엔드** - GPU 가속 필요시
3. **병렬 처리** - 다중 조건 시뮬레이션
4. **캐싱** - 동일 조건 결과 재사용

```python
# JAX 백엔드 사용
import pybamm

pybamm.set_options({"jax": True})

# GPU 가속 시뮬레이션
solver = pybamm.JaxSolver()
```

---

## 6. 구현 로드맵

### Phase 1: 기본 통합 (4주)

- [ ] SPM 래퍼 모듈
- [ ] 기본 실험 프로토콜 지원
- [ ] 용량 시뮬레이션
- [ ] 실험 데이터 비교

### Phase 2: 열화 모델 (4주)

- [ ] SEI 성장 모델 통합
- [ ] LAM 모델 통합
- [ ] 리튬 플레이팅 모델
- [ ] 열화 파라미터 피팅

### Phase 3: 고급 기능 (4주)

- [ ] DFN 모델 지원
- [ ] 파라미터 추정 자동화
- [ ] UI 통합 (Streamlit)
- [ ] 배치 시뮬레이션

---

## 7. 참고 자료

- [PyBaMM Documentation](https://pybamm.readthedocs.io/)
- [PyBaMM GitHub](https://github.com/pybamm-team/PyBaMM)
- [PyBOP (Parameterization)](https://github.com/pybamm-team/PyBOP)
- Chen et al. (2020) - 기본 파라미터 셋
