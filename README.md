# 🏎️ F1TENTH ROS 2 Foxy Simulation

본 패키지는 ROS 2 Foxy 기반의 F1TENTH 시뮬레이션 환경과 FGM(Follow the Gap Method) 자율주행 알고리즘을 포함하고 있습니다. 제공된 Docker 환경을 통해 실행하는 것을 원칙으로 합니다.

    # -b 옵션으로 foxy 브랜치를 바로 가져옵니다.
    git clone -b foxy https://github.com/joshuaha0702/f1tenth_ws_foxy
## 1. 환경 구축 및 빌드

### 1. host connection
    xhost +local:docker

### 2. 컨테이너 실행

설치된 Docker 버전에 따라 명령어가 다를 수 있습니다. 아래 중 작동하는 명령어를 사용하세요.

    # Docker Compose V2 (최신):
    docker compose build
    docker compose up -d

    # Docker Compose V1 (구버전):
    docker-compose build
    docker-compose up -d

    Tip: 만약 명령어가 둘 다 안 된다면 docker --version으로 도커 설치 여부를 먼저 확인하세요.

### 3. 컨테이너 접속

    docker exec -it f1tenth_foxy bash

### 4. alias

    # 컨테이너 내부 전용 단축어 (Dockerfile에 기입됨)
    cb  # 빌드 + 환경 소싱까지 한 번에
    cs  # 빌드 없이 환경 소싱만

## 2. 패키지 빌드

컨테이너 내부(/root/f1tenth_ws)에서 빌드를 진행합니다. (alias cb 사용 가능)

Bash

### 기본 명령어
    colcon build --symlink-install
    source install/setup.bash

### 또는 설정된 단축어 사용
    cb

# 🚀 실행 방법 (데모 주행)

### 터미널 1: 가제보 월드 및 차량 스폰

가제보 시뮬레이터를 실행하고 선배님의 racecar_walker 트랙에 차량을 배치합니다.

    ros2 launch racecar_description spawn_car.launch.py

### 터미널 2: FGM 알고리즘 노드 실행

차량의 라이다 데이터를 분석해 조향 명령을 내리는 메인 알고리즘 노드입니다.

    ros2 run f1tenth_fgm_ros2 fgm_node

### 터미널 3: 시각화 (RViz2)
시뮬레이션이 실행 중인 상태에서 새로운 터미널을 열고 접속하여 실행합니다.

    rviz2 -d src/racecar_description/rviz/f1tenth_default.rviz

### 통합 런쳐 실행 
현재 통합런처에는 위 3가지 기능을 한번에 실행 할 수 있도록 만들었습니다.

    ros2 launch f1tenth_fgm_ros2 f1tenth_fgm_ros2.launch.py

## 추가적인 기능 구현

### joy_teleop.py
현재 joystic 으로 가제보 내부 차를 돌릴 수 있도록 런치를 구성해놓았습니다.

조이스틱으로 조종을 위해서 위 방법 터미널 1의 명령어를 실행후 다른 터미널에서 아래 명령어를 실행합니다.

    ros2 launch f1tenth_fgm_ros2 example.launch.py

### data_logger.py
이 노드는 /scan /drive 토픽을 구독하여 실시간으로 csv파일 형식으로 작성하는 코드입니다.

현재 헤더는 lab, timestamp, speed, steering, scan 으로 되어있습니다.

    ros2 run f1tenth_fgm_ros2 data_logger.py



### ⚙️ 주요 파라미터 수정 (Tip)

만약 차량이 코너 안쪽 벽을 긁는다면, config의 f1tenth_fgm_ros2.yaml 에서 다음 값을 조정하세요.

    carWidth_tolerance: 안전 마진 값 (현재 추천: 0.40)

    max_speed: 주행 속도 (안정적인 테스트를 위해 1.0 이하 추천)

## 📄 License

This project is licensed under the **MIT License**.  
자세한 내용은 [LICENSE](./LICENSE) 파일을 확인하세요.

---
**Credits:**
- Base simulator: [F1TENTH Official](https://github.com/f1tenth/f1tenth_simulator)
- Environment & Algorithm Porting: Joshua Ha
- World models: Originally developed by CIRL@seoultech