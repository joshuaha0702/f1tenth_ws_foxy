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
cb  # 빌드 + 환경 소싱까지 한 번에
cs  # 빌드 없이 환경 소싱만
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

모든 노드는 기본적으로 `car1`이라는 네임스페이스(robot_name)를 사용하도록 설정되어 있습니다.

### 터미널 1: 가제보 월드 및 차량 스폰

가제보 시뮬레이터를 실행하고 racecar_walker 트랙에 차량을 배치합니다.

```bash
ros2 launch racecar_description spawn_car.launch.py
```

### 터미널 2: FGM 알고리즘 노드 실행

차량의 라이다 데이터를 분석해 조향 명령을 내리는 메인 알고리즘 노드입니다. 멀티 로봇 구동 시 `robot_name` 파라미터를 변경하여 실행할 수 있습니다.

```bash
# 기본 실행 (car1)
ros2 run f1tenth_fgm_ros2 fgm_node

# 특정 네임스페이스 지정 실행 예시
ros2 run f1tenth_fgm_ros2 fgm_node --ros-args -p robot_name:=car2
```

### 터미널 3: 시각화 (RViz2)
시뮬레이션이 실행 중인 상태에서 새로운 터미널을 열고 접속하여 실행합니다.

```bash
rviz2 -d src/racecar_description/rviz/f1tenth_default.rviz
```

### 통합 런쳐 실행 
현재 통합 런처에는 위 3가지 기능을 한 번에 실행할 수 있도록 만들었습니다.

```bash
ros2 launch f1tenth_fgm_ros2 f1tenth_foxy_gazebo.launch.py
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
* **특징:** `TwistStamped` 메시지를 활용하여 라이다 데이터와 조향/속도 명령의 **타임스탬프를 정확히 매칭**하여 저장합니다. 여러 로봇의 데이터를 수집할 때 덮어씌워지지 않도록 **파일명에 로봇 이름이 포함**됩니다.
* **데이터 헤더:** `lap`, `time`, `steer`, `desired_speed`, `lidar_0` ... `lidar_N`
* 다운스케일링 Hz 조정 옵션 지원 (5.0, 10.0, 20.0, 40.0)

```bash
# 기본 실행 (car1)
ros2 run f1tenth_fgm_ros2 data_logger.py --ros-args -p target_hz:=10.0 -p scan_downsample_factor:=360

# 특정 로봇 지정 실행 예시
ros2 run f1tenth_fgm_ros2 data_logger.py --ros-args -p robot_name:=car2 -p target_hz:=20.0
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

만약 차량이 코너 안쪽 벽을 긁는다면, `config` 폴더의 `f1tenth_fgm_ros2.yaml` 에서 다음 값을 조정하세요.

* **`carWidth_tolerance`:** 차량 폭에 대한 안전 마진 값 (현재 추천: 0.40)
* **`max_speed`:** 최대 주행 속도 (안정적인 테스트를 위해 1.0 이하 추천)

## 📄 License

This project is licensed under the **MIT License**.  
자세한 내용은 [LICENSE](./LICENSE) 파일을 확인하세요.

---
**Credits:**
- Base simulator: [F1TENTH Official](https://github.com/f1tenth/f1tenth_simulator)
- Environment & Algorithm Porting: Joshua Ha
- World models: Originally developed by CIRL@seoultech