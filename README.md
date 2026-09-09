# 🏎️ F1TENTH ROS 2 Foxy Simulation

본 패키지는 ROS 2 Foxy 기반의 F1TENTH 시뮬레이션 환경과 FGM(Follow the Gap Method) 자율주행 알고리즘을 포함하고 있습니다. 제공된 Docker 환경을 통해 실행하는 것을 원칙으로 합니다. 

최신 업데이트를 통해 **멀티 로봇 네임스페이스 분리**가 지원되며, 라이다(Scan)와 조향(Drive) 데이터 간의 정밀한 **타임스탬프 동기화 기반 데이터 로깅**을 제공합니다.

```bash
# -b 옵션으로 foxy 브랜치를 바로 가져옵니다.
git clone -b foxy https://github.com/joshuaha0702/f1tenth_ws_foxy
```

## 1. 환경 구축 및 빌드

### 1. host connection
```bash
xhost +local:docker
```

### 2. 컨테이너 실행

본 패키지의 기본 도커 설정은 **NVIDIA GPU 사용**을 권장하고 있습니다.
GPU 유무에 따라 아래 명령어 중 본인 환경에 맞는 것을 선택해 실행하세요.

**🏎️ GPU가 있는 PC (기본 설정):**
```bash
docker compose build
docker compose up -d
```

**💻 GPU가 없는 PC (CPU 전용 모드):**
GPU를 찾을 수 없다는 에러(`could not select device driver "nvidia"`)가 발생하면 `cpu` 프로필을 사용하여 실행하세요.
```bash
docker compose build
docker compose --profile cpu up -d ros2_cpu
```

> **Tip:** 구버전 Docker를 사용하신다면 `docker compose` 대신 `docker-compose`를 입력하시면 됩니다. 컨테이너를 종료하실 때는 `docker compose down`을 사용하세요.

### 3. 컨테이너 접속

```bash
docker exec -it f1tenth_foxy bash
```

### 4. alias

```bash
# 컨테이너 내부 전용 단축어 (Dockerfile에 기입됨)
cb       # 빌드 + 환경 소싱까지 한 번에
cs       # 빌드 없이 환경 소싱만
extract  # 저장된 rosbag에서 CSV 데이터 추출 (Lattice Planner 툴 활용)
```

## 2. 패키지 빌드

컨테이너 내부(`/root/f1tenth_ws`)에서 빌드를 진행합니다. (alias `cb` 사용 가능)

### 기본 명령어
```bash
colcon build --symlink-install
source install/setup.bash
```

### 또는 설정된 단축어 사용
```bash
cb
```

---

# 🚀 실행 방법 (데모 주행)

모든 노드는 기본적으로 `car1`이라는 네임스페이스(robot_name)를 사용하도록 설정되어 있습니다. 또한, 최신 업데이트를 통해 제어 메시지 규격이 **AckermannDriveStamped**로 표준화되었습니다.

### 터미널 1: 가제보 월드 및 차량 스폰

가제보 시뮬레이터를 실행하고 racecar_walker 트랙에 차량을 배치합니다.

```bash
ros2 launch racecar_description spawn_car.launch.py
```

### 터미널 2: FGM 알고리즘 노드 실행

차량의 라이다 데이터를 분석해 조향 명령을 내리는 메인 알고리즘 노드입니다. 멀티 로봇 구동 시 `robot_name` 파라미터를 변경하여 실행할 수 있습니다. `ackermann_to_twist` 노드가 자동으로 실행되어 시뮬레이션의 Twist 입력과 호환됩니다.

```bash
# 기본 실행 (car1)
ros2 run f1tenth_fgm_ros2 fgm_node

# 특정 네임스페이스 지정 실행 예시
ros2 run f1tenth_fgm_ros2 fgm_node --ros-args -p robot_name:=car2
```

### 터미널 3: 시각화 (RViz2)
시뮬레이션이 실행 중인 상태에서 새로운 터미널을 열고 접속하여 실행합니다. `car1`과 `car2`를 동시에 모니터링할 수 있도록 설정되어 있습니다.

```bash
rviz2 -d src/racecar_description/rviz/f1tenth_default.rviz
```

### 통합 런쳐 실행 
현재 통합 런처에는 위 3가지 기능을 한 번에 실행할 수 있도록 만들었습니다.

```bash
ros2 launch f1tenth_fgm_ros2 f1tenth_foxy_gazebo.launch.py
```

---

# 🤖 End2Race Agent 실행 방법

학습된 신경망 모델(`.pth`)을 이용해 car1을 자율주행하는 패키지입니다.  
car2는 Lattice Planner가 정속으로 주행하며, car1은 라이다 데이터를 GRU 기반 모델에 입력해 조향/속도를 출력합니다.

### 통합 런처 실행

```bash
ros2 launch f1tenth_end2race_ros2 f1tenth_end2race_gazebo.launch.py \
  head2head:=true \
  models:=src/f1tenth_end2race_ros2/models/origin_20260616.pth
```

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `head2head` | `true` | Lattice 기반 정속 주행 car2 함께 스폰 |
| `models` (= `model_path`) | 패키지 내 `models/end2race.pth` | 추론에 사용할 모델 가중치 경로 (`.pth`) |
| `namespace` | `car1` | car1 로봇 네임스페이스 |
| `x` / `y` | `6.4` / `16.0` | car1 스폰 위치 (m) |
| `x2` / `y2` | `6.4` / `12.0` | car2 스폰 위치 (m) |

> **Note:** 런처가 기동되면 Gazebo와 RViz2가 함께 열립니다. car1이 라이다 신호를 수신하는 즉시 추론·주행이 시작됩니다.

### 모델 파일 관리

```
src/f1tenth_end2race_ros2/models/
├── end2race.pth          # 패키지 기본 모델
└── origin_20260616.pth   # 날짜별 학습 체크포인트 예시
```

`model_path` 인자에 절대경로 또는 워크스페이스 상대경로를 넘기면 됩니다.

### 주요 에이전트 파라미터

`config/end2race_config.yaml`에서 조정합니다.

| 파라미터 | 설명 |
|---|---|
| `control.max_speed` | 최대 속도 (m/s), 기본 4.0 |
| `control.max_steer` / `min_steer` | 조향 클리핑 범위 (rad), 기본 ±0.52 |
| `hidden_scale` | GRU 히든 사이즈 스케일 (모델 학습 시 사용 값과 일치해야 함) |

---

## 🚗 End2Race 실차(Real Hardware) 실행

시뮬레이터 없이 **실제 F1TENTH 차량**에서 학습된 모델을 구동하는 런처입니다.  
LiDAR 드라이버와 VESC가 제공하는 `/scan`, `/drive` 토픽을 그대로 사용합니다.

### 전제 조건

- LiDAR 드라이버 노드와 VESC 드라이버 노드가 이미 실행 중이어야 합니다.  
- 두 드라이버는 네임스페이스 없이 `/scan`(센서 입력), `/drive`(제어 출력) 토픽을 사용해야 합니다.

### 실행 명령어

```bash
# 기본 실행 (모델 기본 경로 사용)
ros2 launch f1tenth_end2race_ros2 f1tenth_end2race_real.launch.py

# 모델 경로 직접 지정
ros2 launch f1tenth_end2race_ros2 f1tenth_end2race_real.launch.py \
  model_path:=src/f1tenth_end2race_ros2/models/origin_20260616.pth

# 원격 PC에서 RViz2로 모니터링할 경우
ros2 launch f1tenth_end2race_ros2 f1tenth_end2race_real.launch.py \
  use_rviz:=true
```

### 런치 인자

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `namespace` | `car1` | 로봇 네임스페이스 (내부적으로 토픽 경로에 사용) |
| `model_path` | 패키지 내 `models/end2race.pth` | 추론에 사용할 모델 가중치 경로 (`.pth`) |
| `use_rviz` | `false` | RViz2 실행 여부 (실차에선 보통 `false`, 개발 PC 모니터링 시 `true`) |

### 토픽 리맵

`agent_node`는 내부적으로 `/{namespace}/scan`, `/{namespace}/drive` 형태로 토픽을 처리하지만, 런처가 실차 하드웨어 스택의 토픽 이름에 맞게 자동으로 리맵합니다.

| agent_node 내부 토픽 | 실차 하드웨어 토픽 |
|---|---|
| `/{namespace}/scan` | `/scan` |
| `/{namespace}/drive` | `/drive` |

> **Note:** `use_rviz:=true` 시 `racecar_description` 패키지의 기본 RViz 설정이 열립니다. TF와 scan은 하드웨어 드라이버 스택이 제공한다고 가정합니다.

---

# 🏁 Lattice Planner 실행 방법

Clothoid 기반 Lattice 플래너를 이용해 레이스라인을 추종하는 단일 로봇 주행 모드입니다.  
가제보 스폰 → 플래너 노드 → RViz2까지 통합 런처 한 번으로 실행할 수 있습니다.

### 통합 런처 실행

```bash
ros2 launch f1tenth_lattice_ros2 f1tenth_lattice_gazebo.launch.py
```

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `namespace` | `car1` | 로봇 네임스페이스 |
| `x` / `y` | `6.4` / `16.0` | 스폰 위치 (m) |
| `yaw_deg` | `-90.0` | 스폰 초기 방향 (도) |
| `map` | `Simple` | 로드할 맵 이름 (`maps/<맵이름>/` 폴더 기준) |
| `config` | `config/lattice_config.yaml` | 플래너 설정 파일 경로. 지정 시 스폰 좌표 기본값도 이 파일에서 읽음 |
| `raceline` | `maps/<맵이름>/raceline1.csv` | 플래너가 추종할 raceline CSV. `episodes:=` 지정 시 episodes.yaml의 `raceline_path`가 자동 적용됨 |

### 주행 시작

런처 실행 후 시뮬레이터와 RViz2가 열리면, **RViz2 상단의 `2D Nav Goal` 버튼을 클릭**하여 맵 위 임의의 지점을 지정합니다. 해당 신호를 수신하는 순간 차량이 주행을 시작합니다.

### 레이스라인 변경

각 맵 폴더(`maps/<맵이름>/`)에 3개의 레인(`raceline0.csv` ~ `raceline2.csv`)이 포함되어 있습니다.  
런처 실행 시 `raceline:=` 인자로 지정합니다 (파일 수정 불필요).

```bash
ros2 launch f1tenth_lattice_ros2 f1tenth_lattice_gazebo.launch.py \
  raceline:=src/f1tenth_lattice_ros2/maps/Simple/raceline0.csv
```

```
inner  : maps/<맵이름>/raceline0.csv
center : maps/<맵이름>/raceline1.csv  ← 기본값
outer  : maps/<맵이름>/raceline2.csv
```

> 우선순위: `raceline:=` 명시 > `episodes:=`로 지정한 episodes.yaml의 `raceline_path` > 기본값(`maps/<맵이름>/raceline1.csv`).  
> 에피소드 수집 시 episode_manager의 스폰 기준 raceline과 플래너 추종 raceline이 자동으로 일치합니다.

### 주요 파라미터 수정

`config/lattice_config.yaml` 에서 플래너 동작을 조정할 수 있습니다.

| 파라미터 | 설명 |
|---|---|
| `lh_grid_lb` / `lh_grid_ub` | lookahead 거리 범위 (m) |
| `cost_weights` | 비용 함수 가중치 `[raceline추종, 속도보상, 곡률페널티, 충돌]` |
| `traj_v_span_min/max` | 후보 궤적 속도 범위 스케일 |
| `minL` / `maxL` | Pure Pursuit lookahead 거리 범위 |

### 선두 차량 추종(Follow) 모드 — 구간별 P 제어

head-to-head 모드에서 **지정한 raceline 인덱스 구간(zone) 안에서만** lattice 추월을 멈추고, 선두차(car2)를 따라가며 차간거리를 P 제어(ACC)로 유지하는 기능입니다. 추월이 위험한 코너 구간 등에서 안전하게 대기하다가, 구간을 벗어나거나 선두차와 멀어지면 자동으로 기존 lattice 추월 모드로 복귀합니다.

**동작 조건** — 아래 두 조건을 모두 만족하는 동안 following이 활성화됩니다.

1. car1(ego)의 현재 raceline 인덱스가 `zones` 중 한 구간 안에 있음
2. car2가 car1 **앞**(트랙 진행 방향, arc-length 기준)으로 `lateral_align_m` 이내에 있음

활성화되면 lattice 후보 생성을 건너뛰고 raceline을 횡 오프셋 없이 그대로 추종(=추월 불가)하며, 속도는 아래 P 제어식으로 결정됩니다.

```
v_cmd = v_leader + kp_gap × (gap − desired_gap_m)
v_cmd = clip(v_cmd, 0, min(raceline 속도, max_follow_speed))
```

*   `gap`: raceline arc-length(s) 기준 전방 차간거리 (트랙 wrap-around 처리됨)
*   `v_leader`: 직전 plan() 호출 대비 선두차 이동량으로 추정한 속도 (sim 시간 기준 실측 dt 사용)
*   차간거리가 목표보다 크면 가속, 작으면 감속하여 `desired_gap_m`으로 수렴

**설정** — `config/lattice_config.yaml`의 `car1:` 아래 `follow:` 블록으로 제어합니다. `follow` 키를 제거하거나 `zones`를 비우면 완전히 비활성(기존 동작 유지)됩니다.

```yaml
car1:
  # ... 기존 파라미터 ...
  follow:
    zones:                        # following 활성 구간 (raceline 행 인덱스 기준)
      - { min: 90,  max: 140 }
      - { min: 210, max: 235 }
      - { min: 0,   max: 15 }
    lateral_align_m: 3.0          # car2가 이 거리(m) 이내로 앞에 있으면 following 진입
    desired_gap_m: 1.5            # P 제어 목표 차간거리 (m)
    kp_gap: 0.8                   # 차간거리 오차 → 속도 보정 P 게인
    max_follow_speed: 3.0         # following 중 속도 상한 (m/s)
    horizon_m: 3.0                # 추종 경로로 잘라 쓸 전방 raceline 길이 (m)
```

| 파라미터 | 설명 |
|---|---|
| `zones` | following을 허용할 raceline 행 인덱스 구간 목록. 인덱스 확인은 `tools/inspect_raceline.py` 활용 |
| `lateral_align_m` | 진입/해제 판정 거리. lattice 회피 반응거리(~3m)보다 크게 잡아야 follow가 추월을 선점함 |
| `desired_gap_m` | ACC 목표 차간거리. 차량 길이(0.58m)를 고려하고 `lateral_align_m`보다 작게 설정 |
| `kp_gap` | P 게인. 클수록 목표 거리 수렴이 빠르지만 속도 변화가 급격해짐 |
| `max_follow_speed` | following 중 서행 상한. raceline 속도와 비교해 더 작은 값이 적용됨 |
| `horizon_m` | 고정 arc-length로 경로를 잘라 Pure Pursuit lookahead보다 항상 길게 유지 (코너 곡률 추종 보장) |

> car1이 car2보다 항상 빠른 세팅에서는 gap이 단조 감소하므로 별도 히스테리시스 없이 단일 임계값으로 진입/해제가 판정됩니다. car2가 뒤로 처지면(추월 완료) gap이 wrap되어 커지면서 자동 해제됩니다. 진입/해제 시 `[follow] ENTER / EXIT` 로그가 출력됩니다.

### Raceline 인덱스 확인 도구

`zones`, `spawn_idx_ranges` 등 raceline 행 인덱스 기반 설정을 잡을 때 사용하는 인터랙티브 뷰어입니다. 맵 위에 raceline을 겹쳐 그리고, 마우스를 올리면 가장 가까운 점의 **행 인덱스와 (x, y) 좌표**를 표시합니다.

```bash
python3 src/f1tenth_lattice_ros2/tools/inspect_raceline.py --map_name Simple --raceline raceline1
```

---

# 📦 데이터 생성 (다중 에피소드 자동 수집)

`episode_manager`를 이용해 head-to-head 주행 데이터를 **완전 자동**으로 대량 수집하는 방법입니다.  
에피소드마다 두 차량을 레이스라인 위 무작위 위치에 텔레포트하고, 일정 시간 주행 후 rosbag을 저장합니다.  
충돌 여부가 자동으로 판단되어 `clean` / `collision` 폴더로 분류됩니다.

## 1. 동작 흐름

```
런처 실행
  └─ Gazebo + car1/car2 스폰 (headless)
       └─ [6초 후] episode_manager 시작
            └─ 에피소드 반복 (num_episodes 횟수만큼)
                 ├─ car1/car2 → raceline 위 랜덤 위치로 텔레포트
                 ├─ settle_time_sec 동안 물리 안정화 대기
                 ├─ rosbag 녹화 시작 (staging/ 에 임시 저장)
                 ├─ sequence_duration_sec 후 녹화 중단
                 └─ 충돌 없음 → clean/ 이동 | 충돌 발생 → collision/ 이동
```

## 2. episodes.yaml 설정

`src/f1tenth_lattice_ros2/config/episodes.yaml` 에서 수집 조건을 조정합니다.

| 항목 | 기본값 | 설명 |
|---|---|---|
| `num_episodes` | `500` | 수집할 에피소드 수 |
| `raceline_path` | `maps/Simple/raceline1.csv` | 스폰 기준 레이스라인 (플래너 추종 raceline에도 자동 적용) |
| `spawn_idx_ranges` | (미지정) | car1 스폰 위치를 raceline 행 인덱스 구간으로 제한 (아래 참고) |
| `opponent_offset.min/max` | `5` / `50` | car2가 car1 앞에 놓이는 행 인덱스 범위 |
| `traj_v_scale.car1/car2` | `1.0~1.0` / `0.3~0.7` | 에피소드별 속도 스케일 범위 (균등 랜덤) |
| `spawn_perturbation` | `±1.2m` / `±10°` | 스폰 시 중심라인 수직 오프셋·heading 섭동 (균등 분포) |
| `random_seed` | `42` | 재현성 시드 (`null`이면 매번 다름) |
| `sequence_duration_sec` | `8.0` | 에피소드 1개당 녹화 길이 (sim 초) |
| `settle_time_sec` | `3.0` | 텔레포트 후 안정화 대기 시간 (sim 초) |
| `output.base_dir` | `/root/f1tenth_ws/data/0630_single` | 데이터 저장 루트 경로 |

```yaml
# episodes.yaml 핵심 항목 예시
num_episodes: 500
sequence_duration_sec: 8.0
traj_v_scale:
  car1: { min: 1.0, max: 1.0 }
  car2: { min: 0.3, max: 0.7 }
output:
  base_dir: /root/f1tenth_ws/data/0630_single
```

### 스폰 구간 제한 (`spawn_idx_ranges`)

car1(ego)의 스폰 위치를 특정 raceline 구간으로 제한할 수 있습니다. 여러 구간을 지정하면 전체 후보 인덱스 풀에서 균등 샘플링(구간 길이에 비례)하며, 키를 생략하거나 `null`이면 전체 raceline에서 샘플링합니다. 특정 코너/직선 구간의 데이터만 집중 수집할 때 유용합니다.

```yaml
# 예: 코너 구간(0~80)과 직선 구간(200~350)에서만 스폰
spawn_idx_ranges:
  - { min: 0,   max: 80  }
  - { min: 200, max: 350 }
```

*   구간이 겹치면 중복 인덱스는 제거되어 균등성이 유지됩니다.
*   `max`가 raceline 길이를 넘거나 `min > max`이면 실행 시 검증 오류로 즉시 종료됩니다.
*   구간별 행 인덱스는 `tools/inspect_raceline.py`로 확인할 수 있습니다.

## 3. 실행

컨테이너 내부(`/root/f1tenth_ws`)에서 실행합니다.

```bash
ros2 launch f1tenth_lattice_ros2 f1tenth_lattice_gazebo.launch.py \
  head2head:=true \
  record:=true \
  headless:=true \
  episodes:=src/f1tenth_lattice_ros2/config/episodes.yaml
```

| 파라미터 | 값 | 설명 |
|---|---|---|
| `head2head` | `true` | car1(ego) + car2(opponent) 동시 스폰 |
| `record` | `true` | episode_manager가 rosbag 녹화 담당 |
| `episodes` | yaml 파일 경로 | 지정 시 episode_manager 활성화 (단일 bag 모드 비활성) |
| `headless` | `true` | Gazebo GUI·RViz2 미실행 → RTF 향상 |

> **Tip:** `headless:=false`로 변경하면 Gazebo 창을 보며 디버깅할 수 있습니다.

## 4. 출력 구조

```
output.base_dir/          (예: /root/f1tenth_ws/data/0630_single/)
├── staging/              # 녹화 중 임시 저장 위치
├── clean/                # 충돌 없는 에피소드 bag
│   ├── ep_001_<timestamp>/
│   ├── ep_002_<timestamp>/
│   └── ...
└── collision/            # 충돌 발생 에피소드 bag
    └── ...
```

## 5. rosbag → CSV 변환

수집 완료 후 `extract` alias로 bag 파일에서 CSV를 추출합니다.

```bash
# 단일 bag 변환
extract --bag data/0630_single/clean/ep_001_<timestamp> --output data/0630_single/csv/ep_001.csv

# 전체 clean 에피소드 일괄 변환 (예시)
for bag in data/0630_single/clean/*/; do
  name=$(basename "$bag")
  extract --bag "$bag" --output "data/0630_single/csv/${name}.csv"
done
```

## 6. 모델 학습 (train.py)

수집한 CSV 데이터로 End2Race 모델(GRU 기반)을 학습하는 스크립트입니다.  
LiDAR 360빔 + 직전 속도를 입력으로 받아 조향(`steer`)·목표 속도(`desired_speed`)를 출력하도록 지도학습합니다.

```bash
# 기본 학습 (origin 모드)
python3 train.py --data_path data/0630_single/csv --mode origin

# 기존 체크포인트에서 이어서 학습 (fine-tuning)
python3 train.py --data_path data/0630_single/csv \
  --checkpoint_path src/f1tenth_end2race_ros2/models/origin_20260629.pth
```

> **Note:** 모델 정의(`End2Race`, `End2RaceWithDelay`)를 `model.py`에서 import하므로,
> `src/f1tenth_end2race_ros2/f1tenth_end2race_ros2/`를 `PYTHONPATH`에 추가하고 실행합니다.
> ```bash
> PYTHONPATH=src/f1tenth_end2race_ros2/f1tenth_end2race_ros2 python3 train.py ...
> ```

### 학습 방식

*   **입력 데이터:** `lidar_0`~`lidar_359`, `steer`, `desired_speed` 컬럼을 가진 에피소드별 CSV (`lidar_delay` 모드는 `lidar_delay` 컬럼 추가 필요). 필수 컬럼이 없거나 길이가 부족한 CSV는 자동으로 건너뜁니다.
*   **시퀀스 구성:** 각 에피소드를 `sequence_length` 길이의 슬라이딩 윈도우(`stride` 간격)로 잘라 시퀀스 단위로 학습합니다. 속도 입력은 한 스텝 이전 값(`speed_prev`)을 사용합니다.
*   **손실 함수:** `MSE(steer) + 0.05 × MSE(speed)` — 조향 학습에 가중치를 둔 구성입니다.
*   **저장:** epoch 평균 손실이 최저를 갱신할 때마다 `model_path`에 저장됩니다. `--model_path` 미지정 시 `{mode}_{YYYYMMDD}.pth`로 자동 생성되며, 같은 이름의 파일이 이미 있으면 해당 가중치를 불러와 이어서 학습합니다.
*   **학습 안정화:** gradient clipping(max_norm=1.0) + `ReduceLROnPlateau` 스케줄러(patience=10, factor=0.5)를 사용합니다.

### 주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--data_path` | `Dataset_Austin/success` | 에피소드 CSV들이 있는 디렉토리 |
| `--mode` | `origin` | `origin`: 기본 / `lidar_delay`: LiDAR 지연 입력 포함 학습 |
| `--model_path` | 자동 생성 | 모델 저장 경로 (`{mode}_{날짜}.pth`) |
| `--checkpoint_path` | `None` | 학습 시작 전 불러올 가중치 (fine-tuning) |
| `--sequence_length` | `100` | 시퀀스당 타임스텝 수 |
| `--stride` | `50` | 슬라이딩 윈도우 간격 |
| `--hidden_scale` | `4` | GRU 히든 사이즈 스케일 (**추론 시 `end2race_config.yaml`의 값과 일치 필수**) |
| `--mask_prob` | `0.1` | 학습 중 입력 마스킹 확률 (regularization) |
| `--batch_size` | `16` | 배치 크기 |
| `--learning_rate` | `0.001` | 초기 학습률 |
| `--num_epochs` | `500` | 학습 epoch 수 |

학습이 끝나면 생성된 `.pth`를 `src/f1tenth_end2race_ros2/models/`로 옮기고, End2Race 런처의 `models:=` 인자로 지정해 주행을 확인합니다.

---

## 🛠 추가적인 기능 구현

### joy_teleop.py
현재 조이스틱(Joystick)으로 가제보 내부 차량을 조종할 수 있도록 런치를 구성해 놓았습니다. 터미널 1의 명령어(차량 스폰)를 실행한 후 다른 터미널에서 아래 명령어를 실행합니다.

```bash
ros2 launch f1tenth_fgm_ros2 example.launch.py
```

### data_logger.py
차량의 라이다 센서 데이터와 주행 명령을 동기화하여 CSV 파일로 실시간 기록하는 노드입니다.
* **구독 토픽:** `/{robot_name}/scan`, `/{robot_name}/drive_stamped`, `/{robot_name}/odom`, `/{robot_name}/map_reset`
* **특징:** `AckermannDriveStamped` 메시지를 활용하여 라이다 데이터와 조향/속도 명령의 **타임스탬프를 정확히 매칭**하여 저장합니다. 여러 로봇의 데이터를 수집할 때 덮어씌워지지 않도록 **파일명에 로봇 이름이 포함**됩니다.
* **데이터 헤더:** `lap`, `time`, `steer`, `desired_speed`, `lidar_0` ... `lidar_N`
* 다운스케일링 Hz 조정 옵션 지원 (5.0, 10.0, 20.0, 40.0)

```bash
# 기본 실행 (car1)
ros2 run f1tenth_fgm_ros2 data_logger.py --ros-args -p target_hz:=10.0 -p scan_downsample_factor:=360

# 특정 로봇 지정 실행 예시
ros2 run f1tenth_fgm_ros2 data_logger.py --ros-args -p robot_name:=car2 -p target_hz:=20.0
```

### extract_bag_csv.py (Data Extraction)
기록된 `rosbag` 파일에서 CSV 데이터를 추출하는 도구입니다. Docker 내부에 설정된 `extract` alias를 통해 간편하게 실행할 수 있습니다.
* **특징:** Zero-Order Hold 방식을 사용하여 제어 명령과 센서 데이터를 타임스탬프 기반으로 동기화합니다.
* **사용법:**
  ```bash
  # 컨테이너 내부에서 실행
  extract --bag <rosbag_path> --output <output_path>
  ```

### random_reset.py
안전 구역 내에서 차량을 무작위 위치 및 방향으로 재스폰(Teleport) 시키는 노드입니다.
해당 노드가 실행되면 대상 로봇의 데이터 로거에 리셋 신호(`/map_reset`)가 전달되어 **새로운 랩(Lap) 측정**이 시작됩니다.

```bash
# 기본 실행 (car1 리셋)
ros2 run f1tenth_fgm_ros2 random_reset.py 

# 특정 로봇 리셋 예시
ros2 run f1tenth_fgm_ros2 random_reset.py --ros-args -p robot_name:=car2
```

---

## ⚙️ 주요 파라미터 수정 (Tip)

*   **`carWidth_tolerance`:** 차량 폭에 대한 안전 마진 값입니다. 코너 안쪽 벽을 긁는다면 이 값을 높이세요. (현재 추천: **0.30**)
*   **`max_speed`:** 최대 주행 속도 (안정적인 테스트를 위해 1.0 이하 추천)
*   **Chassis Collision Detection:** `racecar.gazebo`에 범퍼 센서가 추가되어 차량 섀시의 충돌이 물리적으로 감지됩니다. 충돌 발생 시 시뮬레이션 상에서 즉각적인 피드백을 확인할 수 있습니다.

## 메모
[로컬에서 Xlaunch 실행 -> SSH 통한 서버 접속 -> 서버 내에서 도커 빌드 -> 도커 내에서 GUI 사용하기] 과정을 위한 도커 실행 파라미터
```bash
XAUTH=/tmp/.docker.xauth
rm -f "$XAUTH"
touch "$XAUTH"
xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge -
chmod 644 "$XAUTH"

docker run -dit \
  --name f1tenth_foxy \
  --network host \
  --ipc host \
  --privileged \
  --gpus all \
  -e DISPLAY="$DISPLAY" \
  -e XAUTHORITY=/tmp/.docker.xauth \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.docker.xauth:/tmp/.docker.xauth:ro \
  -v /dev/dri:/dev/dri \
  -v "$PWD/src:/root/f1tenth_ws/src" \
  -v /dev/input:/dev/input \
  -v "$PWD/bags:/root/f1tenth_ws/bags" \
  -v "$PWD/data:/root/f1tenth_ws/data" \
  f1tenth_ws_foxy-ros2_humble \
  bash

export DISPLAY=localhost:10.0
export XAUTHORITY=/root/.Xauthority
```

## 📄 License

This project is licensed under the **MIT License**.  
자세한 내용은 [LICENSE](./LICENSE) 파일을 확인하세요.

---
**Credits:**
- Base simulator: [F1TENTH Official](https://github.com/f1tenth/f1tenth_simulator)
- Environment & Algorithm Porting: Joshua Ha
- World models: Originally developed by CIRL@seoultech