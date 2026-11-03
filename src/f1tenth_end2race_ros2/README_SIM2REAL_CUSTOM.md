# F1TENTH Sim2Real 현실성 보완 가제보 가이드

기존 Gazebo 환경은 **센서 노이즈가 없는 완벽한 환경, 지나치게 높은 모터 토크와 비현실적인 타이어 마찰계수**로 인해 실차 환경(Sim2Real Gap)과 차이가 컸습니다. 

이를 보완하여 학습 모델을 현실적으로 검증할 수 있도록 전용 커스텀 파일들을 분리 생성했습니다.

---

## 1. 생성된 커스텀 파일 목록

1. **URDF / Gazebo 물리 모델**:
   - `racecar_description/urdf/racecar_realistic.gazebo`
   - `racecar_description/urdf/racecar_realistic.xacro`
2. **차량 스폰 Launch 파일**:
   - `racecar_description/launch/spawn_car_realistic.launch.py`
3. **End2Race 모델 검증 Launch 파일**:
   - `f1tenth_end2race_ros2/launch/f1tenth_end2race_gazebo_custom.launch.py`

---

## 2. 주요 개선 사항 (Sim2Real Gap 축소)

| 구분 | 기존 가제보 환경 | 보완된 커스텀 환경 (`racecar_realistic`) |
| :--- | :--- | :--- |
| **라이다 센서** | Hokuyo 30Hz, 30m, **노이즈 0 (이상치)** | **Lakibeam-1 (TOF) 30Hz, 25m, TOF 노이즈(1.0cm)** |
| **타이어 마찰 (mu)** | 1.0 (지나치게 접지력이 높음) | **0.85 (현실적인 슬립/드리프트 허용)** |
| **모터 토크** | 5000 Nm (순간 가속 비현실적) | **25.0 Nm (실차 가속도 특성 반영)** |
| **최고 속도/조향각** | 10.0 m/s, 0.52 rad (~30도) | **7.0 m/s, 0.34 rad (~19.5도, 실차 한계)** |
| **조향 속도 한계** | 5.0 rad/s | **3.2 rad/s (실차 서보 모터 반응 속도와 일치)** |

---

## 3. 실행 방법

```bash
# 기본 실행 (Simple 맵에서 car1에 학습 모델 로드)
ros2 launch f1tenth_end2race_ros2 f1tenth_end2race_gazebo_custom.launch.py

# 특정 모델 가중치(.pth)와 맵 지정 실행
ros2 launch f1tenth_end2race_ros2 f1tenth_end2race_gazebo_custom.launch.py \
  map:=silverstone_track \
  model_path:=/home/capstone1/f1tenth_ws_foxy/src/f1tenth_end2race_ros2/models/end2race.pth

# 상대 차량(car2)과 함께 대결 주행 검증 시
ros2 launch f1tenth_end2race_ros2 f1tenth_end2race_gazebo_custom.launch.py head2head:=true
```

---

## 4. 추가 튜닝 포인트 (`racecar_realistic.gazebo`)

현실 데이터와 더 일치시키려면 `racecar_description/urdf/racecar_realistic.gazebo` 파일에서 아래 파라미터들을 조정할 수 있습니다:
- **라이다 노이즈**: `<noise><stddev>0.015</stddev></noise>` 값을 키우면 센서 잡음이 증가합니다.
- **바퀴 슬립/미끄러짐**: `<mu1 value="0.85"/>`를 0.7~0.8로 낮추면 코너링 시 언더스티어/미끄러짐이 더 발생합니다.
- **서보 지연/PID**: `<left_steering_pid_gain>2.5 0.01 0.05</left_steering_pid_gain>` 값을 조절하여 조향 반응성을 미세 조정할 수 있습니다.
