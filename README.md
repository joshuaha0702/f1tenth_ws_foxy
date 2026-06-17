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

설치된 Docker 버전에 따라 명령어가 다를 수 있습니다. 아래 중 작동하는 명령어를 사용하세요.

```bash
# Docker Compose V2 (최신):
docker compose build
docker compose up -d

# Docker Compose V1 (구버전):
docker-compose build
docker-compose up -d

# Tip: 만약 명령어가 둘 다 안 된다면 docker --version으로 도커 설치 여부를 먼저 확인하세요.
```

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

### 주행 시작

런처 실행 후 시뮬레이터와 RViz2가 열리면, **RViz2 상단의 `2D Nav Goal` 버튼을 클릭**하여 맵 위 임의의 지점을 지정합니다. 해당 신호를 수신하는 순간 차량이 주행을 시작합니다.

### 레이스라인 변경

`maps/` 폴더에 3개의 레인(`raceline0.csv` ~ `raceline2.csv`)이 포함되어 있습니다.  
런처 파일(`f1tenth_lattice_gazebo.launch.py`) 내 `raceline_path` 값을 수정하여 변경합니다.

```
inner  : maps/raceline0.csv
center : maps/raceline1.csv  ← 기본값
outer  : maps/raceline2.csv
```

### 주요 파라미터 수정

`config/lattice_config.yaml` 에서 플래너 동작을 조정할 수 있습니다.

| 파라미터 | 설명 |
|---|---|
| `lh_grid_lb` / `lh_grid_ub` | lookahead 거리 범위 (m) |
| `cost_weights` | 비용 함수 가중치 `[raceline추종, 속도보상, 곡률페널티, 충돌]` |
| `traj_v_span_min/max` | 후보 궤적 속도 범위 스케일 |
| `minL` / `maxL` | Pure Pursuit lookahead 거리 범위 |

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
| `num_episodes` | `50` | 수집할 에피소드 수 |
| `raceline_path` | `maps/raceline1.csv` | 스폰 기준 레이스라인 |
| `opponent_offset.min/max` | `40` / `100` | car2가 car1 앞에 놓이는 행 인덱스 범위 |
| `traj_v_scale.car1/car2` | `0.9~1.0` / `0.3~0.7` | 에피소드별 속도 스케일 범위 (균등 랜덤) |
| `random_seed` | `42` | 재현성 시드 (`null`이면 매번 다름) |
| `sequence_duration_sec` | `15.0` | 에피소드 1개당 녹화 길이 (sim 초) |
| `settle_time_sec` | `3.0` | 텔레포트 후 안정화 대기 시간 (sim 초) |
| `output.base_dir` | `/root/f1tenth_ws/data/0529` | 데이터 저장 루트 경로 |

```yaml
# episodes.yaml 핵심 항목 예시
num_episodes: 50
sequence_duration_sec: 15.0
traj_v_scale:
  car1: { min: 0.9, max: 1.0 }
  car2: { min: 0.3, max: 0.7 }
output:
  base_dir: /root/f1tenth_ws/data/0529
```

## 3. 실행

컨테이너 내부(`/root/f1tenth_ws`)에서 실행합니다.

```bash
ros2 launch f1tenth_lattice_ros2 f1tenth_lattice_gazebo.launch.py \
  head2head:=true \
  record:=true \
  episodes:=src/f1tenth_lattice_ros2/config/episodes.yaml \
  headless:=true
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
output.base_dir/          (예: /root/f1tenth_ws/data/0529/)
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
extract --bag data/0529/clean/ep_001_<timestamp> --output data/0529/csv/ep_001.csv

# 전체 clean 에피소드 일괄 변환 (예시)
for bag in data/0529/clean/*/; do
  name=$(basename "$bag")
  extract --bag "$bag" --output "data/0529/csv/${name}.csv"
done
```

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

## 📄 License

This project is licensed under the **MIT License**.  
자세한 내용은 [LICENSE](./LICENSE) 파일을 확인하세요.

---
**Credits:**
- Base simulator: [F1TENTH Official](https://github.com/f1tenth/f1tenth_simulator)
- Environment & Algorithm Porting: Joshua Ha
- World models: Originally developed by CIRL@seoultech