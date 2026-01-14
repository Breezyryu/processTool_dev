# PINN 수명예측 구현 계획

> 작성일: 2026-01-14

---

## 1. PINN 개요

### 1.1 PINN이란?

**Physics-Informed Neural Network (PINN)**은 물리 법칙을 신경망 학습에 통합하는 방법론.

```
일반 신경망:
Loss = L_data (데이터 손실만)

PINN:
Loss = L_data + λ * L_physics (물리 손실 추가)
```

### 1.2 배터리 수명 예측에서 PINN의 장점

| 장점 | 설명 |
|------|------|
| **데이터 효율성** | 적은 실험 데이터로 일반화 |
| **물리적 일관성** | 비현실적 예측 방지 |
| **해석 가능성** | 물리 파라미터 추출 가능 |
| **외삽 능력** | 학습 범위 외 조건 예측 |
| **불확실성 정량화** | 예측 신뢰도 제공 가능 |

### 1.3 PINN vs 기존 방법론

| 방법 | 장점 | 단점 |
|------|------|------|
| **경험적 모델** | 단순, 빠름 | 외삽 제한 |
| **물리 모델 (PyBaMM)** | 정확, 해석 가능 | 파라미터 필요, 느림 |
| **순수 ML** | 복잡한 패턴 학습 | 데이터 많이 필요, 블랙박스 |
| **PINN** | 데이터 효율, 외삽 | 학습 어려움 |

---

## 2. 배터리 수명 PINN 설계

### 2.1 문제 정의

**목표:** 사이클/시간에 따른 용량 열화 예측

**입력 (x):**
- 사이클 수 (n)
- 시간 (t)
- 온도 (T)
- C-rate
- SOC 범위

**출력 (y):**
- 잔존 용량 (C/C₀)
- (선택) 열화 파라미터 (SEI 두께, LAM 등)

### 2.2 물리 법칙 (Physics Constraints)

**1. Arrhenius 온도 의존성:**
```
k(T) = A * exp(-Ea / RT)

여기서:
- k: 열화 속도
- A: 빈도 인자
- Ea: 활성화 에너지 (40-80 kJ/mol)
- R: 8.314 J/mol·K
- T: 절대 온도
```

**2. SEI 성장 (Calendar Aging):**
```
δ_SEI(t) = √(2 * k_SEI * D * t)

dQ_loss/dt = n_F * dδ_SEI/dt
           = n_F * √(k_SEI * D / 2t)

→ Q_loss ∝ √t (Parabolic growth)
```

**3. Cycle Aging:**
```
dC/dn = -α * exp(-Ea_cyc / RT) * |I|^β

여기서:
- n: 사이클 수
- α, β: 피팅 상수
- I: 전류
```

**4. 용량 열화 총합:**
```
C(t, n, T) = C₀ - Q_calendar(t, T) - Q_cycle(n, T, I)

또는:
C/C₀ = 1 - a*√t - b*n^c (경험적 형태)
```

### 2.3 PINN 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    PINN Architecture                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input Layer          Hidden Layers          Output      │
│  ┌─────────┐         ┌───────────┐         ┌────────┐  │
│  │ n (cycle)│         │           │         │        │  │
│  │ t (time) │ ───────▶│  MLP      │────────▶│ C/C₀   │  │
│  │ T (temp) │         │ (64-128)  │         │        │  │
│  │ C-rate   │         │ x 4-6     │         │(열화   │  │
│  │ SOC range│         │           │         │ 파라미터)│ │
│  └─────────┘         └───────────┘         └────────┘  │
│                            │                             │
│                            │ 자동 미분                    │
│                            ▼                             │
│                    ┌───────────────┐                     │
│                    │ Physics Loss   │                     │
│                    │ - Arrhenius    │                     │
│                    │ - SEI 성장      │                     │
│                    │ - 용량 보존     │                     │
│                    └───────────────┘                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 구현 상세

### 3.1 기술 스택

```python
# 핵심 라이브러리
torch          # PyTorch (딥러닝)
deepxde        # PINN 전문 라이브러리
numpy
pandas
matplotlib
scikit-learn   # 데이터 전처리

# 선택적
optuna         # 하이퍼파라미터 튜닝
wandb          # 실험 추적
```

### 3.2 모듈 구조

```
battery_analysis_v2/
└── pinn_engine/
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   ├── base_pinn.py          # 기본 PINN 클래스
    │   ├── capacity_pinn.py      # 용량 열화 PINN
    │   └── multi_output_pinn.py  # 다중 출력 PINN
    │
    ├── physics_loss/
    │   ├── __init__.py
    │   ├── arrhenius.py          # Arrhenius 손실
    │   ├── sei_growth.py         # SEI 성장 손실
    │   ├── cycle_aging.py        # Cycle aging 손실
    │   └── constraints.py        # 물리적 제약 (C>0 등)
    │
    ├── training/
    │   ├── __init__.py
    │   ├── trainer.py            # 학습 파이프라인
    │   ├── callbacks.py          # 콜백 함수
    │   └── scheduler.py          # 학습률 스케줄러
    │
    ├── data/
    │   ├── __init__.py
    │   ├── preprocessor.py       # 데이터 전처리
    │   └── augmentation.py       # 데이터 증강 (PyBaMM)
    │
    └── utils/
        ├── __init__.py
        ├── visualization.py      # 결과 시각화
        └── metrics.py            # 평가 메트릭
```

### 3.3 핵심 코드 설계

```python
# pinn_engine/models/capacity_pinn.py

import torch
import torch.nn as nn

class CapacityPINN(nn.Module):
    """배터리 용량 열화 예측 PINN"""
    
    def __init__(self, 
                 input_dim=5,      # [n, t, T, C-rate, SOC]
                 hidden_dim=128,
                 num_layers=5,
                 output_dim=1):    # C/C₀
        super().__init__()
        
        # 신경망 구조
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())  # 0-1 출력
        
        self.network = nn.Sequential(*layers)
        
        # 학습 가능한 물리 파라미터
        self.Ea = nn.Parameter(torch.tensor(50000.0))  # 활성화 에너지
        self.A = nn.Parameter(torch.tensor(1e10))       # 빈도 인자
        
    def forward(self, x):
        """순전파"""
        return self.network(x)
    
    def physics_loss(self, x, y_pred):
        """물리 손실 계산"""
        n = x[:, 0:1]   # 사이클
        t = x[:, 1:2]   # 시간
        T = x[:, 2:3]   # 온도 (K)
        
        R = 8.314  # 기체상수
        
        # 자동 미분으로 dC/dt, dC/dn 계산
        y_pred.requires_grad_(True)
        
        # dC/dt (Calendar aging)
        dC_dt = torch.autograd.grad(
            y_pred, t, 
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True
        )[0]
        
        # 물리 제약: dC/dt = -k(T) / (2*√t)  (SEI 성장)
        k_T = self.A * torch.exp(-self.Ea / (R * T))
        physics_calendar = dC_dt + k_T / (2 * torch.sqrt(t + 1e-6))
        
        # dC/dn (Cycle aging)
        dC_dn = torch.autograd.grad(
            y_pred, n,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True
        )[0]
        
        # 물리 제약: dC/dn = -α * k(T)
        alpha = 0.001  # 피팅 상수
        physics_cycle = dC_dn + alpha * k_T
        
        # 총 물리 손실
        loss_physics = (
            torch.mean(physics_calendar**2) + 
            torch.mean(physics_cycle**2)
        )
        
        return loss_physics
    
    def boundary_loss(self, y_pred):
        """경계 조건 손실"""
        # C/C₀는 0과 1 사이
        loss_bound = torch.mean(torch.relu(-y_pred) + torch.relu(y_pred - 1))
        
        # 초기 조건: t=0, n=0 → C/C₀ = 1
        # (별도 처리 필요)
        
        return loss_bound
```

```python
# pinn_engine/training/trainer.py

import torch
import torch.optim as optim

class PINNTrainer:
    """PINN 학습 관리자"""
    
    def __init__(self, model, 
                 lambda_physics=1.0,
                 lambda_boundary=1.0,
                 lr=1e-3):
        self.model = model
        self.lambda_physics = lambda_physics
        self.lambda_boundary = lambda_boundary
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=100, factor=0.5
        )
        
    def compute_loss(self, x, y_true):
        """총 손실 계산"""
        y_pred = self.model(x)
        
        # 데이터 손실 (MSE)
        loss_data = torch.mean((y_pred - y_true)**2)
        
        # 물리 손실
        loss_physics = self.model.physics_loss(x, y_pred)
        
        # 경계 조건 손실
        loss_boundary = self.model.boundary_loss(y_pred)
        
        # 총 손실
        total_loss = (
            loss_data + 
            self.lambda_physics * loss_physics +
            self.lambda_boundary * loss_boundary
        )
        
        return total_loss, {
            'data': loss_data.item(),
            'physics': loss_physics.item(),
            'boundary': loss_boundary.item()
        }
    
    def train_epoch(self, dataloader):
        """1 에폭 학습"""
        self.model.train()
        total_loss = 0
        
        for x_batch, y_batch in dataloader:
            self.optimizer.zero_grad()
            
            loss, loss_dict = self.compute_loss(x_batch, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def fit(self, train_loader, val_loader=None, epochs=1000):
        """전체 학습"""
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            if val_loader:
                val_loss = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.6f}")
        
        return history
```

### 3.4 데이터 증강 (PyBaMM 연동)

```python
# pinn_engine/data/augmentation.py

import pybamm
import numpy as np

class PyBaMMDataAugmentor:
    """PyBaMM으로 가상 데이터 생성"""
    
    def __init__(self):
        self.model = pybamm.lithium_ion.SPM()
        self.param = pybamm.ParameterValues("Chen2020")
    
    def generate_degradation_data(self, 
                                   temperatures=[25, 35, 45],
                                   c_rates=[0.5, 1.0, 2.0],
                                   n_cycles=500):
        """
        다양한 조건에서 열화 데이터 생성
        
        Returns:
            DataFrame with columns: [cycle, time, temp, c_rate, capacity]
        """
        data = []
        
        for T in temperatures:
            for C in c_rates:
                # 실험 프로토콜 생성
                experiment = pybamm.Experiment([
                    f"Discharge at {C}C until 2.5V",
                    "Rest for 5 minutes",
                    f"Charge at {C}C until 4.2V",
                    "Hold at 4.2V until C/50",
                    "Rest for 5 minutes"
                ] * n_cycles)
                
                # 시뮬레이션
                sim = pybamm.Simulation(
                    self.model,
                    parameter_values=self.param,
                    experiment=experiment
                )
                solution = sim.solve()
                
                # 용량 추출
                capacities = solution["Discharge capacity [A.h]"].entries
                
                for i, cap in enumerate(capacities):
                    data.append({
                        'cycle': i,
                        'time': i * 2,  # 예시
                        'temp': T + 273,  # K
                        'c_rate': C,
                        'capacity': cap / capacities[0]
                    })
        
        return pd.DataFrame(data)
```

---

## 4. 학습 전략

### 4.1 Two-Phase Training

```
Phase 1: 물리 손실 중심 (Pre-training)
├── 높은 λ_physics (10-100)
├── 물리적 일관성 학습
└── 데이터 없이도 가능

Phase 2: 데이터 피팅 (Fine-tuning)
├── 낮은 λ_physics (0.1-1)
├── 실제 데이터에 맞춤
└── 물리 제약 유지
```

### 4.2 Multi-Task Learning

```python
# 다중 출력 PINN
outputs = {
    'capacity': C/C₀,
    'sei_thickness': δ_SEI,
    'lam_fraction': LAM%,
    'resistance': R/R₀
}

# 공유 인코더 + 개별 디코더
class MultiOutputPINN(nn.Module):
    def __init__(self):
        self.encoder = SharedEncoder()
        self.capacity_head = CapacityDecoder()
        self.sei_head = SEIDecoder()
        # ...
```

### 4.3 불확실성 정량화

```python
# MC Dropout으로 불확실성 추정
def predict_with_uncertainty(model, x, n_samples=100):
    model.train()  # Dropout 활성화
    
    predictions = []
    for _ in range(n_samples):
        pred = model(x)
        predictions.append(pred.detach())
    
    predictions = torch.stack(predictions)
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)
    
    return mean, std
```

---

## 5. 구현 로드맵

### Phase 1: 기본 구현 (4주)

- [ ] 기본 PINN 클래스 구현
- [ ] Arrhenius 물리 손실
- [ ] 학습 파이프라인
- [ ] 단일 조건 학습/예측

### Phase 2: 고급 기능 (4주)

- [ ] 다중 조건 학습
- [ ] PyBaMM 데이터 증강
- [ ] 불확실성 정량화
- [ ] 하이퍼파라미터 튜닝

### Phase 3: 통합 및 배포 (4주)

- [ ] BatteryDataTool 통합
- [ ] UI (Streamlit) 연동
- [ ] 모델 저장/로드
- [ ] 배치 예측

---

## 6. 예상 성능

| 메트릭 | 경험적 모델 | 순수 ML | PINN |
|--------|-----------|--------|------|
| RMSE (학습 범위) | ~0.02 | ~0.01 | ~0.01 |
| RMSE (외삽) | ~0.10 | ~0.20 | ~0.05 |
| 필요 데이터량 | 중간 | 많음 | 적음 |
| 물리적 일관성 | 높음 | 낮음 | 높음 |

---

## 7. 참고 자료

- [Physics-Informed Neural Networks (Raissi et al., 2019)](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [DeepXDE: PINN Library](https://deepxde.readthedocs.io/)
- [Battery PINN Papers](https://scholar.google.com/scholar?q=physics+informed+neural+network+battery)
