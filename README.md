# F1TENTH ROS 2 Foxy Workspace

ROS 2 Foxy와 Gazebo 기반의 F1TENTH 시뮬레이션 워크스페이스입니다. Follow the Gap(FGM), Lattice Planner + Pure Pursuit, End2Race 신경망 추론, 다중 에피소드 rosbag 수집·재생·CSV 변환 도구를 포함합니다.

Docker 환경에서 실행하는 것을 기준으로 하며 기본 로봇 네임스페이스는 `car1`, 상대 차량은 `car2`입니다.

## 구성

| 경로 | 역할 |
|---|---|
| `src/racecar_description` | 차량 URDF, Gazebo world/model, RViz 설정 |
| `src/f1tenth_fgm_ros2` | Follow the Gap 주행, 조이스틱, 리셋·로깅 유틸리티 |
| `src/f1tenth_lattice_ros2` | Lattice Planner, Pure Pursuit, 에피소드 수집, rosbag 재생·변환 |
| `src/f1tenth_end2race_ros2` | 학습 모델 기반 End2Race 추론 |
| `src/gazebo_ros2_2Dmap_plugin` | Gazebo world의 2D occupancy map 생성 플러그인 |
| `train.py`, `model.py` | End2Race 학습 스크립트와 모델 정의 |

현재 Lattice용 맵과 raceline은 다음 디렉토리에 있습니다.

- `Simple`
- `monza_track`
- `interlagos_track`
- `silverstone_track`

## 설치 및 빌드

### 저장소 받기

```bash
git clone -b foxy https://github.com/joshuaha0702/f1tenth_ws_foxy.git
cd f1tenth_ws_foxy
```

### GUI 권한 설정

Linux 호스트에서 Gazebo와 RViz를 표시하려면 다음 명령을 먼저 실행합니다.

```bash
xhost +local:docker
```

### 컨테이너 실행

NVIDIA GPU 환경:

```bash
docker compose build
docker compose up -d
docker exec -it f1tenth_foxy bash
```

CPU 전용 환경:

```bash
docker compose build
docker compose --profile cpu up -d ros2_cpu
docker exec -it f1tenth_foxy_cpu bash
```

종료:

```bash
docker compose down
```

구버전 Docker에서는 `docker compose` 대신 `docker-compose`를 사용할 수 있습니다.

### ROS 패키지 빌드

컨테이너 내부의 `/root/f1tenth_ws`에서 실행합니다.

```bash
colcon build --symlink-install
source install/setup.bash
```

Dockerfile에 다음 alias가 등록되어 있습니다.

| alias | 동작 |
|---|---|
| `cb` | 워크스페이스로 이동한 뒤 빌드하고 `install/setup.bash` 적용 |
| `cs` | 기존 빌드의 `install/setup.bash` 적용 |
| `extract <폴더명>` | `data/<폴더명>/clean/*`의 rosbag을 CSV로 일괄 변환 |

소스나 launch 파일을 변경한 뒤에는 `cb`를 다시 실행하십시오.

## FGM 데모

FGM, 차량 스폰, Ackermann→Twist 브리지, RViz를 한 번에 실행합니다.

```bash
ros2 launch f1tenth_fgm_ros2 f1tenth_foxy_gazebo.launch.py
```

주요 launch 인자:

| 인자 | 기본값 | 설명 |
|---|---:|---|
| `namespace` | `car1` | 로봇 네임스페이스 |
| `map` | `Simple` | Gazebo map 이름 |
| `x`, `y` | `6.4`, `16.0` | 시작 위치(m) |
| `yaw` | `-1.570796` | 시작 방향(rad) |

직접 나누어 실행하려면 차량 스폰 후 FGM 노드를 실행합니다.

```bash
# 터미널 1
ros2 launch racecar_description spawn_car.launch.py \
  namespace:=car1 map:=Simple x:=6.4 y:=16.0 yaw:=-1.570796

# 터미널 2
ros2 run f1tenth_fgm_ros2 fgm_node --ros-args \
  --params-file src/f1tenth_fgm_ros2/config/fgm_config.yaml \
  -r __ns:=/car1 -p robot_name:=car1

# 터미널 3: Gazebo용 제어 메시지 브리지
ros2 run f1tenth_fgm_ros2 ackermann_to_twist.py --ros-args -r __ns:=/car1
```

FGM 설정은 `src/f1tenth_fgm_ros2/config/fgm_config.yaml`에서 변경합니다. 주요 항목은 `max_speed`, `min_speed`, `max_steering_angle`, `carWidth_tolerance`입니다.

## Lattice Planner

Lattice 경로 생성과 별도 Pure Pursuit 제어 노드를 함께 실행합니다. 기본값은 `car1`과 `car2`가 모두 생성되는 head-to-head 모드입니다. odometry와 scan이 들어오면 자동으로 계획과 제어를 시작하므로 RViz의 `2D Nav Goal` 입력은 필요하지 않습니다.

```bash
ros2 launch f1tenth_lattice_ros2 f1tenth_lattice_gazebo.launch.py
```

단일 차량 또는 다른 트랙으로 실행하는 예:

```bash
ros2 launch f1tenth_lattice_ros2 f1tenth_lattice_gazebo.launch.py \
  map:=monza_track \
  head2head:=false \
  raceline:=src/f1tenth_lattice_ros2/maps/monza_track/raceline1.csv
```

주요 launch 인자:

| 인자 | 기본값 | 설명 |
|---|---|---|
| `map` | `Simple` | `maps/<map>/` 및 Gazebo world 선택 |
| `head2head` | `true` | `car2` 생성·주행 여부 |
| `headless` | `false` | `true`이면 gzclient와 RViz를 생략 |
| `config` | 패키지의 `config/sim_lattice_config.yaml` | 차량별 플래너 및 스폰 설정 (시뮬 수집 전용; `lattice_config.yaml`은 실차 튜닝본) |
| `raceline` | `maps/<map>/raceline1.csv` | 추종할 raceline |
| `x`, `y`, `yaw_deg` | config의 `car1.spawn` | `car1` 시작 pose |
| `x2`, `y2`, `yaw_deg2` | config의 `car2.spawn` | `car2` 시작 pose |
| `cleanup_gazebo` | `false` | 시작 전에 기존 Gazebo 프로세스를 종료할지 여부 |
| `record` | `false` | rosbag 녹화 여부 |
| `bag_output` | `./bags/output` | 단일 녹화의 출력 prefix |
| `episodes` | 빈 값 | 지정 시 다중 에피소드 매니저 활성화 |

`namespace` 인자도 선언되어 있지만 현재 Lattice 시뮬레이션 launch 내부 차량 이름은 `car1`과 `car2`로 고정되어 있습니다.

### Raceline 선택

각 맵 폴더에는 `raceline0.csv`부터 `raceline2.csv`까지의 세 레인과 반대 방향용 `*_ccw.csv`가 있습니다.

```text
raceline0.csv  inner
raceline1.csv  center (기본값)
raceline2.csv  outer
```

경로 선택 우선순위는 다음과 같습니다.

1. launch의 `raceline:=...`
2. `episodes:=...` YAML의 `raceline_path`
3. `maps/<map>/raceline1.csv`

Raceline 행 인덱스는 인터랙티브 도구로 확인할 수 있습니다.

```bash
python3 src/f1tenth_lattice_ros2/tools/inspect_raceline.py \
  --map_name Simple --raceline raceline1
```

`--lanes lane0 lane1 lane2`를 추가하면 lane 중심선도 함께 표시됩니다.

### Follow 모드

`lattice_config.yaml`의 `car1.follow`은 지정된 raceline 인덱스 구간에서 추월 대신 선두차와의 간격을 P 제어로 유지합니다. `zones`를 비우거나 `follow` 블록을 제거하면 비활성화됩니다.

```yaml
car1:
  follow:
    zones:
      - { min: 90, max: 140 }
      - { min: 210, max: 235 }
    lateral_align_m: 3.0
    desired_gap_m: 1.5
    kp_gap: 0.8
    max_follow_speed: 3.0
    horizon_m: 3.0
```

## End2Race

### Gazebo 추론

학습된 GRU 모델로 `car1`을 주행시킵니다. 현재 저장소를 새로 빌드한 환경에서는 아래처럼 단독 주행으로 먼저 확인하십시오. 상대차 실행에 관한 경로 문제는 [현재 제한사항](#현재-제한사항)에 정리되어 있습니다.

```bash
ros2 launch f1tenth_end2race_ros2 f1tenth_end2race_gazebo.launch.py \
  model_path:=src/f1tenth_end2race_ros2/models/origin_20260616.pth \
  head2head:=false
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `namespace` | `car1` | End2Race 차량 네임스페이스 |
| `head2head` | `true` | Lattice 기반 `car2` 생성 여부 |
| `model_path` | 패키지의 `models/end2race.pth` | 추론 모델 경로 |
| `x`, `y` | `6.4`, `16.0` | `car1` 시작 위치 |
| `x2`, `y2` | `6.4`, `12.0` | `car2` 시작 위치 |

추론 설정은 `src/f1tenth_end2race_ros2/config/end2race_config.yaml`에 있습니다. 학습 때 사용한 `hidden_scale`과 추론 설정의 값을 반드시 일치시켜야 합니다. `control.speed_scale`은 모델 출력 속도의 배율입니다.

### 실차 추론

LiDAR와 VESC 드라이버가 먼저 실행되어 `/scan`과 `/drive`를 제공한다고 가정합니다.

```bash
ros2 launch f1tenth_end2race_ros2 f1tenth_end2race_real.launch.py \
  model_path:=src/f1tenth_end2race_ros2/models/end2race.pth
```

원격 모니터링용 RViz를 함께 실행하려면 `use_rviz:=true`를 추가합니다. `namespace`의 기본값은 `car1`이지만 launch에서 `/{namespace}/scan`, `/{namespace}/drive`를 각각 `/scan`, `/drive`로 remap합니다.

## rosbag 녹화

### 단일 녹화

`episodes` 없이 `record:=true`를 지정하면 첫 `/car1/drive` 명령을 기다린 뒤 전체 토픽 녹화를 시작합니다.

```bash
ros2 launch f1tenth_lattice_ros2 f1tenth_lattice_gazebo.launch.py \
  head2head:=true \
  record:=true \
  bag_output:=/root/f1tenth_ws/bags/run
```

출력 디렉토리 이름에는 KST timestamp가 붙습니다.

### 다중 에피소드 자동 수집

Simple 맵의 폭·길이 증강 버전은 `Simple_augmented`입니다. 원래 Simple의 주행 폭
약 3.2 m, 직선 16 m, 중앙 레이스 라인 약 47.1 m를 각각 4.0 m, 20.5 m,
59.85 m로 늘렸습니다. 점유 지도, Gazebo 충돌 벽, 레이스 라인 세 개
(`raceline0~2.csv`, 반폭의 30/50/70% 위치)와 반대 방향용 `*_ccw.csv`를 모두 같은
파라미터로 생성합니다. `simple_augmented_lattice_config.yaml`은 `0812_head2head_100hz`를 수집한 시뮬레이션용
플래너 설정(상대차 회피 가중치 300, lh_grid 1.0~3.0)을 쓰고, 스폰 좌표와 follow zone
(raceline1의 코너 구간 99~151, 250~301)만 증강 트랙에 맞춥니다. `lattice_config.yaml`은
실차용 튜닝(회피 가중치 0.0, lookahead 확대, adaptive lookahead·heading 보정 ON)이라 head-to-head
수집에 쓰면 충돌률이 약 65%까지 올라가고 추종 정책이 달라져 라벨 분포가 바뀝니다. 그래서 Gazebo
런치의 기본 config는 0480f88 이전 값을 그대로 둔 `sim_lattice_config.yaml`이며, 실차 전용 추종
보강(`adaptive_lookahead`, `k_heading`)은 config에서 명시적으로 켜지 않는 한 꺼져 있습니다.
`simple_augmented_episodes.yaml`은 원본과 같은 분포(선두차 0.3~0.7배속, offset 5~50행)에
횡방향 섭동만 트랙 폭에 비례해 ±1.5 m로 키웠습니다.

```bash
python3 src/f1tenth_lattice_ros2/tools/generate_simple_augmented.py \
  --track-width 4.0 --straight-length 20.5
colcon build --packages-select racecar_description f1tenth_lattice_ros2 f1tenth_fgm_ros2 --symlink-install
source install/setup.bash
ros2 launch f1tenth_lattice_ros2 f1tenth_lattice_gazebo.launch.py \
  map:=Simple_augmented \
  config:=src/f1tenth_lattice_ros2/config/simple_augmented_lattice_config.yaml \
  head2head:=true headless:=true record:=true \
  episodes:=src/f1tenth_lattice_ros2/config/simple_augmented_episodes.yaml
```

에피소드 bag은 car1과 선두 차량 car2의 scan·drive 토픽을 모두 기록합니다.
완료 후 두 차량의 학습 CSV를 함께 추출합니다. 단일 차량 bag에 car2 토픽이
없으면 car2 CSV는 만들지 않습니다.

```bash
python3 src/f1tenth_lattice_ros2/tools/extract_bag_csv.py simple_augmented_h2h_20260928
# 한 차량만 추출: --robots car1
```

#### Simple 수치 변형 맵 20종 (`Simple_v01`~`Simple_v20`)

`config/simple_variants.yaml`에 적힌 폭(3.2~4.8 m), 직선 길이(16~28 m), 벽 굴곡
(진폭 0~0.4 m, 파장 4~12 m, 시드)으로 20개 맵을 생성합니다. 안쪽/바깥쪽 벽은 독립적으로
흔들려서 국소 트랙 폭이 구간마다 달라지고, 레이스 라인은 명목 반경을 유지합니다.
생성기는 국소 폭 2.8 m 미만이나 레이스 라인 벽 여유 0.6 m 미만이면 실패합니다.

맵마다 `maps/<name>/`에 지도, 레이스 라인(`raceline0~2`, `*_ccw`), `lattice_config.yaml`,
`episodes.yaml`, 렌더 이미지 `<name>_render.png`(3D 벽 메시, 상단 뷰, 국소 폭 그래프)가 생기고,
`racecar_description`에 `meshes/<name 소문자>.stl`과 `worlds/<name 소문자>.world`가 생깁니다.
전체 비교 이미지는 `maps/Simple_variants_overview.png`입니다.

```bash
cd src/f1tenth_lattice_ros2/tools
python3 generate_simple_variants.py              # 전체 20개 (--only Simple_v05 로 일부만)
cd -
colcon build --packages-select racecar_description f1tenth_lattice_ros2 f1tenth_fgm_ros2 --symlink-install
source install/setup.bash
ros2 launch f1tenth_lattice_ros2 f1tenth_lattice_gazebo.launch.py \
  map:=Simple_v05 \
  config:=src/f1tenth_lattice_ros2/maps/Simple_v05/lattice_config.yaml \
  head2head:=true headless:=true record:=true \
  episodes:=src/f1tenth_lattice_ros2/maps/Simple_v05/episodes.yaml
```

단일 맵은 `generate_simple_augmented.py --name <이름> --track-width ... --straight-length ...
--wall-amplitude ... --seed ... --configs`로도 만들 수 있습니다. `spawn_car.launch.py`는
`worlds/<map 소문자>.world`가 있으면 그 월드를 사용합니다.

```bash
ros2 launch f1tenth_lattice_ros2 f1tenth_lattice_gazebo.launch.py \
  head2head:=true \
  headless:=true \
  record:=true \
  episodes:=src/f1tenth_lattice_ros2/config/episodes.yaml
```

launch 시작 약 15초 후 에피소드 매니저가 동작합니다. 각 에피소드마다 차량을 raceline 위에 배치하고, 안정화 후 rosbag을 녹화한 뒤 충돌 여부에 따라 분류합니다.

주요 `episodes.yaml` 항목:

| 항목 | 현재 기본값 | 설명 |
|---|---|---|
| `num_episodes` | `500` | 에피소드 수 |
| `raceline_path` | `maps/Simple/raceline1.csv`의 절대경로 | 스폰과 주행에 사용할 raceline |
| `spawn_idx_ranges` | 미지정 | `car1` 스폰 인덱스 범위 제한 |
| `opponent_offset.min/max` | `5` / `50` | `car2`의 전방 인덱스 offset 범위 |
| `traj_v_scale.car1/car2` | `1.0` / `0.3~0.7` | 에피소드별 속도 scale 범위 |
| `spawn_perturbation` | `±1.2 m`, `±10°` | 횡방향·heading 무작위 섭동 |
| `random_seed` | `42` | `null`이면 실행마다 다른 시퀀스 |
| `sequence_duration_sec` | `8.0` | 녹화 길이(sim time) |
| `settle_time_sec` | `3.0` | 텔레포트 후 안정화 시간(sim time) |
| `output.base_dir` | `/root/f1tenth_ws/data/0812_head2head_100hz` | 출력 루트 |

단일 차량 반복 수집 예시는 `config/single_episodes.yaml`에 있습니다.

```bash
ros2 launch f1tenth_lattice_ros2 f1tenth_lattice_gazebo.launch.py \
  map:=Simple \
  head2head:=false \
  headless:=true \
  record:=true \
  episodes:=src/f1tenth_lattice_ros2/config/single_episodes.yaml
```

출력 구조:

```text
output.base_dir/
├── staging/                       # 녹화 중 임시 디렉토리
├── clean/                         # 충돌 없는 rosbag
│   └── ep0000_<timestamp>/
└── collision/                     # 충돌이 감지된 rosbag
    └── ep0001_<timestamp>/
```

`record:=false`로 실행하면 에피소드 배치와 충돌 판정은 수행하지만 bag은 저장하지 않습니다.

## rosbag 재생

`gazebo_replay.launch.py`가 bag 재생, RViz, 필요한 TF를 함께 실행합니다. 기본값은 Gazebo를 띄우지 않는 RViz 전용 모드입니다.

```bash
# RViz 전용 재생
ros2 launch f1tenth_lattice_ros2 gazebo_replay.launch.py \
  bag:=/root/f1tenth_ws/data/<실험명>/clean/<에피소드>/<bag_file>.db3

# Gazebo 차량 위치까지 재현
ros2 launch f1tenth_lattice_ros2 gazebo_replay.launch.py \
  bag:=/root/f1tenth_ws/data/<실험명>/clean/<에피소드>/<bag_file>.db3 \
  gazebo:=true

# 0.5배속
ros2 launch f1tenth_lattice_ros2 gazebo_replay.launch.py \
  bag:=<db3 파일 경로> rate:=0.5
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `bag` | 오래된 `data/0529` 예시 경로 | 재생할 SQLite3 `.db3` 파일 경로. 항상 직접 지정 권장 |
| `rate` | `1.0` | 재생 속도 배율 |
| `gazebo` | `false` | `false`: RViz만, `true`: Gazebo + RViz |

Gazebo 모드에서는 `bag2gazebo_node`가 `/car1/odom`과 `/car2/odom`을 받아 물리가 정지된 Gazebo 차량 pose를 갱신합니다.

## rosbag을 CSV로 변환

`extract`는 `/root/f1tenth_ws/data/<폴더명>/clean/` 바로 아래의 모든 rosbag 디렉토리를 이름순으로 변환합니다.

```bash
extract 0812_head2head_100hz
```

직접 실행할 수도 있습니다.

```bash
python3 src/f1tenth_lattice_ros2/tools/extract_bag_csv.py 0812_head2head_100hz
```

입력과 출력:

```text
/root/f1tenth_ws/data/<폴더명>/clean/<rosbag>/
  -> /root/f1tenth_ws/data/<폴더명>/csv/car1_extracted_<rosbag>.csv
```

추출기는 `/car1/scan`, `/car1/drive`, `/clock`을 읽으며 360개 LiDAR 값, `steer`, `desired_speed`, `lidar_delay`를 기록합니다. `/clock`이 없는 bag은 변환하지 않습니다.

## End2Race 모델 학습

`train.py`와 같은 디렉토리의 `model.py`를 사용하므로 저장소 루트에서 실행합니다.

```bash
python3 train.py \
  --data_path data/0812_head2head_100hz/csv \
  --mode origin \
  --model_path src/f1tenth_end2race_ros2/models/origin_custom.pth
```

기존 모델에서 이어서 학습:

```bash
python3 train.py \
  --data_path data/0812_head2head_100hz/csv \
  --mode origin \
  --checkpoint_path src/f1tenth_end2race_ros2/models/origin_20260629.pth \
  --model_path src/f1tenth_end2race_ros2/models/origin_finetuned.pth
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--data_path` | `Dataset_Austin/success` | CSV 디렉토리 |
| `--mode` | `origin` | `origin` 또는 `lidar_delay` |
| `--model_path` | `{mode}_{YYYYMMDD}.pth` | 최적 loss 모델 저장 경로 |
| `--checkpoint_path` | 없음 | 초기 가중치 경로 |
| `--sequence_length` | `400` | 시퀀스 길이 |
| `--stride` | `50` | 슬라이딩 윈도우 간격 |
| `--hidden_scale` | `4` | GRU hidden scale |
| `--mask_prob` | `0.1` | 입력 마스킹 확률 |
| `--batch_size` | `16` | 배치 크기 |
| `--learning_rate` | `0.001` | 초기 학습률 |
| `--num_epochs` | `1000` | epoch 수 |
| `--disable_lidar_noise` | 꺼짐 | 거리 비례 LiDAR 가우시안 노이즈 증강 끄기 |
| `--start_from_rest` | 꺼짐 | 에피소드 첫 행을 이전 속도 0으로 포함. 에피소드마다 정지 상태에서 시작하는 End2Race 추론과 맞춤 |

`origin` 모드는 `lidar_0`~`lidar_359`, `steer`, `desired_speed`가 필요하며 `lidar_delay` 모드는 `lidar_delay` 컬럼도 필요합니다. 학습과 추론의 `hidden_scale` 값이 다르면 모델을 불러올 수 없습니다.

현재 `docker-compose.yml`은 `src/`, `data/`, `bags/`만 bind mount하므로 저장소 루트의 `train.py`와 `model.py`는 컨테이너에 자동으로 나타나지 않습니다. 컨테이너에서 학습하려면 두 파일을 `/root/f1tenth_ws`에 별도로 mount 또는 복사하거나, 필요한 Python 패키지가 설치된 호스트 환경에서 실행하십시오.

### Simple 변형 맵 파이프라인: 수집 → 검증 → 학습 → 추월 평가

아래 명령은 모두 컨테이너 안에서 실행합니다(`tools` = `src/f1tenth_lattice_ros2/tools`).
시뮬레이션은 한 번에 하나만 실행하십시오. 두 개를 동시에 돌리면 CPU 경합으로
drive 누락(최대 300 ms)이 생겨 사용 가능한 에피소드가 절반 이하로 줄었습니다.

1. 수집: 맵을 차례로 실행하고 맵마다 launch를 종료합니다. seed는 `<seed base> + 맵 번호`입니다.

   ```bash
   ROS_DOMAIN_ID=91 GAZEBO_MASTER_URI=http://127.0.0.1:11491 \
     bash tools/run_episodes.sh /root/f1tenth_ws/data/simple_variants_h2h 50 20260929 \
     Simple_v01 Simple_v02   # ... Simple_v20
   ```

2. 제어 주기 검증: `CONTROL_RATE_100HZ_ISSUE.md`의 기준(drive `header.stamp` 간격 10 ms,
   중복·역행·누락 없음, drive stamp = odom stamp)을 차량별로 검사합니다. `strict`는 문서 기준
   그대로, `usable`은 한 스텝(20 ms) 이하 누락만 허용하고 컨트롤러가 odom을 놓친 적이 없는 경우입니다.

   ```bash
   python3 tools/validate_episodes.py "/root/f1tenth_ws/data/simple_variants_h2h/*/clean/ep*" \
     --summary /root/f1tenth_ws/data/simple_variants_h2h/validation.csv
   ```

3. 학습셋: `usable` 에피소드만 차량별 폴더로 추출합니다. 평가용 맵은 `--exclude-maps`로 뺍니다.

   ```bash
   cd tools && python3 build_training_set.py \
     /root/f1tenth_ws/data/simple_variants_h2h/validation.csv \
     /root/f1tenth_ws/data/simple_variants_h2h/train --exclude-maps Simple_v07 Simple_v15
   ```

   `extract_bag_csv.py`는 drive `header.stamp`가 유효하면(엄격 증가, 중앙 간격 약 10 ms) 그 시각으로
   모든 drive를 한 행씩 기록하고, 각 행에 sim time상 그 이전의 최신 scan을 붙입니다. 에피소드 끝의
   STOP 정지 명령(steer 0, speed 0)은 뺍니다. 예전 bag처럼 stamp가 유효하지 않으면 기존 수신 시각
   방식(`--time-source receipt`)으로 돌아갑니다.

   학습은 저장소 루트에서 `--start_from_rest`를 켜고 실행합니다. 이 옵션이 없으면 모델이 "이전 속도 0"
   입력을 본 적이 없어, 추론 시 출발하지 못하거나 속도가 점점 떨어졌습니다.

   ```bash
   python3 train.py --data_path data/simple_variants_h2h/train/car1 --start_from_rest --num_epochs 300 \
     --model_path src/f1tenth_end2race_ros2/models/ego_variants.pth      # 선두 차량은 train/car2
   ```

4. 추월 평가: `EGO=end2race`이면 car1을 End2Race 모델이 몰고(odom마다 추론, sim time 100 Hz,
   `speed_scale` 1.0), `LEADER=end2race`이면 car2를 모델이 몹니다. 같은 seed로 `EGO=lattice`를
   돌리면 전문가 기준선이 됩니다. `CONFIG_DIR`에는 `<map>/{lattice_config,episodes}.yaml`을 둡니다.

   ```bash
   EGO=end2race EGO_MODEL=/path/to/ego.pth DURATION=15 \
     bash tools/run_episodes.sh /root/f1tenth_ws/data/eval_ego 20 100 Simple_v07 Simple_v15
   python3 tools/evaluate_overtakes.py /root/f1tenth_ws/data/eval_ego
   ```

   추월은 car1이 선두 차량보다 차 한 대 길이(0.6 m) 이상 앞선 경우이고, 충돌은 접촉 상대에
   따라 차량/벽(ego, 선두 차량)으로 나눕니다. 같은 인자는 `f1tenth_lattice_gazebo.launch.py`의
   `ego:=end2race ego_model:=...`, `leader:=end2race leader_model:=...`로도 쓸 수 있습니다.

#### 2026-09-29 결과 (Simple_v01~v20, 맵당 50 에피소드)

- 수집 1,000 에피소드. 제어 주기 `usable` 판정 car1 878, car2 885. usable bag은 99.72~100 Hz,
  bag별 dt 표준편차 최대 0.53 ms, 중복·역행 0. 학습 CSV(v07·v15 제외) car1 643 에피소드/51.3만 행,
  car2 648/51.6만 행, 행 간격 10 ms(최대 20 ms).
- 평가: 5개 맵(v07·v15 학습 제외, v12 학습, Simple_augmented·Simple 처음 보는 맵) × 20 에피소드, 15초.

| car1 (ego) | 추월 | 충돌 없는 추월 | 차량 충돌 | 벽 충돌 | 출발 실패 | 평균 속도 |
|---|---:|---:|---:|---:|---:|---:|
| lattice 전문가 | 72% | 68% | 22% | 1% | 1% | 1.81 m/s |
| `origin_0820.pth` (Simple만 학습) | 82% | 70% | 26% | 0% | 0% | 1.87 m/s |
| `ego_variants_20260929.pth` (`--start_from_rest` 없음) | 57% | 48% | 20% | 2% | 9% | 1.48 m/s |
| `ego_variants_20260929_ft.pth` (위 모델 + `--start_from_rest` 80 epoch) | 81% | 62% | 29% | 0% | 0% | 1.85 m/s |

| car2 (선두, car1은 lattice) | 선두 벽 충돌 | 선두 멈춤 | 선두 속도 (에피소드별 표준편차) | ego 추월 | 차량 충돌 |
|---|---:|---:|---:|---:|---:|
| lattice (`traj_v_scale` 0.3~0.7) | 0% | 0% | 1.03 m/s (0.24) | 72% | 22% |
| `leader_variants_20260929.pth` | 0% | 0% | 1.09 m/s (0.14) | 75% | 24% |

End2Race 추론 노드(odom 트리거)는 평가 중 평균 99.5 Hz였고, 누락은 대부분 에피소드 시작 직후
0.1~0.3초에 몰려 있었습니다.

## 기타 도구

### 조이스틱 주행

```bash
ros2 launch f1tenth_fgm_ros2 example.launch.py
```

설정 파일은 `src/f1tenth_fgm_ros2/config/joy_teleop.yaml`입니다.

### 차량 무작위 리셋

```bash
ros2 run f1tenth_fgm_ros2 random_reset.py
ros2 run f1tenth_fgm_ros2 random_reset.py --ros-args -p robot_name:=car2
```

`/{robot_name}/map_reset`을 발행하며 Gazebo의 차량 pose를 무작위로 변경합니다.

### CSV 로그 영상 생성

LiDAR, 조향각, 속도 로그를 MP4로 렌더링합니다.

```bash
python3 src/f1tenth_fgm_ros2/tools/visualize_log.py \
  data/<실험명>/csv/<파일>.csv \
  --output replay.mp4 \
  --fps 20
```

MP4 저장에는 시스템의 `ffmpeg`가 필요합니다.

### Raceline 생성

기존 occupancy map에서 lane과 raceline을 생성합니다.

```bash
python3 src/f1tenth_lattice_ros2/tools/generate_raceline.py \
  --map_dir src/f1tenth_lattice_ros2/maps/Simple \
  --map_name Simple \
  --num_lanes 3
```

반대 방향 raceline은 `--counter_clockwise`를 추가합니다. 생성 전에 원본 map과 기존 CSV를 백업하는 것을 권장합니다.

### Gazebo world에서 2D map 생성

`tools/generate_all.sh`는 `monza_track`, `interlagos_track`, `silverstone_track`을 순서대로 처리합니다. Gazebo 프로세스를 실행하고 종료하며 map 파일을 덮어쓸 수 있으므로 내용을 확인한 뒤 컨테이너의 워크스페이스 루트에서 실행하십시오.

```bash
bash src/f1tenth_lattice_ros2/tools/generate_all.sh
```

## 설정 파일

| 파일 | 용도 |
|---|---|
| `src/f1tenth_fgm_ros2/config/fgm_config.yaml` | FGM 차량·속도·안전 여유 설정 |
| `src/f1tenth_lattice_ros2/config/lattice_config.yaml` | 실차(`f1tenth_lattice_real.launch.py`) Lattice/Pure Pursuit 설정 |
| `src/f1tenth_lattice_ros2/config/sim_lattice_config.yaml` | 시뮬 수집(`f1tenth_lattice_gazebo.launch.py` 기본) Lattice/Pure Pursuit 설정 |
| `src/f1tenth_lattice_ros2/config/episodes.yaml` | head-to-head 반복 수집 설정 |
| `src/f1tenth_lattice_ros2/config/single_episodes.yaml` | 단일 차량 반복 수집 예시 |
| `src/f1tenth_end2race_ros2/config/end2race_config.yaml` | End2Race 모델·제어 설정 |

100 Hz 수집 구조와 측정 방법은 `CONTROL_RATE_100HZ_ISSUE.md`를 참고하십시오.

## 현재 제한사항

- `f1tenth_lattice_real.launch.py`의 기본 map/raceline 경로는 현재 `maps/<맵 이름>/...` 구조로 아직 갱신되지 않았습니다. 실차 Lattice 실행 전에 launch의 리소스 경로를 수정해야 합니다.
- `f1tenth_end2race_gazebo.launch.py`의 `head2head:=true` 상대차도 오래된 Lattice map/raceline 경로를 사용합니다. 현재 저장소만 새로 빌드한 환경에서는 `head2head:=false`로 End2Race 단독 주행을 먼저 확인하십시오.
- `data_logger.py`는 `/drive_stamped`를 `TwistStamped`로 구독하면서 Ackermann의 `drive` 필드를 읽는 구형 구현이 남아 있습니다. 자동 수집 데이터는 rosbag과 `extract_bag_csv.py` 경로를 사용하십시오.
- `gazebo_replay.launch.py`의 `bag` 기본값은 과거 `data/0529` 경로이므로 재생할 `.db3` 경로를 항상 명시하십시오.

## 문제 해결

- `could not select device driver "nvidia"`: CPU profile로 실행합니다.
- Gazebo/RViz 창이 열리지 않음: 호스트의 `DISPLAY`와 `xhost +local:docker` 실행 여부를 확인합니다.
- 이전 Gazebo가 포트를 점유함: Lattice launch에 `cleanup_gazebo:=true`를 한 번만 명시합니다. 이 옵션은 호스트의 `gzserver`/`gzclient` 프로세스를 종료하므로 공유 환경에서는 주의하십시오.
- `Package ... not found`: 컨테이너에서 `cb` 또는 `source install/setup.bash`를 실행합니다.
- CSV 변환 시 `/clock` 경고: 해당 bag에 `/clock` 토픽이 기록됐는지 `ros2 bag info <bag>`로 확인합니다.

## License

이 프로젝트는 [MIT License](LICENSE)를 따릅니다.

Credits:

- Base simulator: [F1TENTH Official](https://github.com/f1tenth/f1tenth_simulator)
- Environment & Algorithm Porting: Joshua Ha
- World models: CIRL@seoultech
