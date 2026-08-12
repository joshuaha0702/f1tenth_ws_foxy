# Head-to-head 제어 주기 100 Hz 이슈 기록

## 개요

Head-to-head 데이터 수집에서 Gazebo의 `/car1/odom`, `/car2/odom`(차량의 위치·속도 정보)은 시뮬레이션 시간 기준 100 Hz였지만, 기존 `/drive`(조향·속도 명령) 제어 출력은 일부 데이터에서 약 34~52 Hz까지 낮아졌다.

이 문제는 callback(메시지 수신 같은 사건에 반응해 실행되는 함수) group과 executor thread(콜백을 배정하고 실행하는 작업 흐름)를 나누는 것만으로 해결되지 않았다. 최종적으로 lattice planning(여러 후보 경로를 생성·평가하는 작업)과 Pure Pursuit controller(경로를 따라갈 조향·속도를 계산하는 제어기)를 서로 다른 ROS 프로세스(독립적으로 실행되는 프로그램 단위)로 분리해 두 차량의 `/drive`를 시뮬레이션 시간 기준 100 Hz로 유지했다.

## Callback group/thread 분리와 프로세스 분리의 차이

핵심은 **동시 실행(concurrency, 여러 작업을 겹쳐 진행하되 같은 순간 실행은 보장하지 않음)을 허용하는 것**과 **실제 병렬 실행(parallelism, 여러 작업을 같은 순간에 실행)이 가능한 것**의 차이다.

- Callback group과 `MultiThreadedExecutor`의 thread 분리는 여러 콜백이 겹쳐 진행될 수 있도록 동시 실행을 허용한다.
- 그러나 콜백들이 같은 Python 프로세스와 GIL(Global Interpreter Lock, 한 프로세스에서 Python 코드의 동시 실행을 제한하는 잠금)을 공유하므로 Python 코드가 서로 다른 CPU core에서 실제로 병렬 실행된다고 보장하지 않는다.
- ROS 프로세스 분리는 각 프로세스에 독립된 Python interpreter(Python 코드를 실행하는 프로그램)와 GIL을 제공하므로 OS가 planner와 controller를 서로 다른 CPU core(연산을 수행하는 CPU의 개별 처리 장치)에서 실제로 병렬 실행할 수 있다.

Callback group과 `MultiThreadedExecutor`를 사용하는 경우 콜백들은 다음처럼 같은 Python 프로세스 안에서 실행된다.

```text
하나의 Python 프로세스
├── Thread A: scan → lattice planning
└── Thread B: odom → Pure Pursuit → drive
```

이 구조에서는 다음 자원을 공유한다.

- CPython 인터프리터(C 언어로 구현된 일반적인 Python 실행기)와 GIL
- ROS executor(실행할 콜백을 선택하고 호출하는 구성 요소)와 callback scheduling(콜백 실행 순서를 정하는 작업)
- 프로세스의 CPU 시간
- Python 객체와 메모리 상태

Callback group 분리는 두 콜백의 동시 실행을 허용할 뿐이다. 실행 시점, 실제 병렬 실행, 제어 주기를 보장하지 않는다. 일반적인 Python 코드에서는 한 thread가 GIL을 가지고 실행되는 동안 다른 thread는 기다리거나 번갈아 실행되므로, 무거운 planning 콜백이 별도 executor thread의 odom 콜백을 지연시킬 수 있다.

NumPy 또는 Numba처럼 내부 native code(Python이 아닌 컴파일된 코드)가 GIL을 해제하는 구간은 thread 구조에서도 일부 병렬 실행될 수 있다. 하지만 모든 연산과 ROS callback scheduling이 GIL 밖에서 수행되는 것은 아니므로, 이 예외만으로 100 Hz 제어 주기를 보장할 수는 없다.

ROS 프로세스를 분리하면 planner와 controller가 각각 독립적인 Python 인터프리터, GIL, executor, callback queue를 갖는다.

```text
Planner 프로세스
  scan → lattice planning → planned_trajectory
                                  │
                                  ▼
Controller 프로세스
  odom + planned_trajectory → Pure Pursuit → drive
```

따라서 planner의 계산 시간이 길어져도 controller의 GIL이나 executor를 직접 점유하지 않는다. OS scheduler(운영체제가 실행할 작업과 CPU를 배정하는 기능)는 두 프로세스를 서로 다른 CPU core에 배치해 실제 병렬로 실행할 수 있다. 두 프로세스는 `planned_trajectory` 토픽(ROS 메시지가 전달되는 이름 있는 통로)으로 필요한 trajectory만 전달한다.

프로세스 분리가 하드 실시간성(정해진 시간 안의 실행을 항상 보장하는 성질)을 보장하는 것은 아니다. 시스템 CPU가 완전히 포화되면 OS scheduling에 의한 지연은 여전히 발생할 수 있다. 다만 이번 문제의 주요 원인이었던 동일 Python 프로세스 내 GIL/executor 간섭은 제거된다.

## 원인

1. 기존에는 scan 기반 lattice planning과 odom 기반 Pure Pursuit 제어가 같은 Python 프로세스에 있었다.
2. 일반 planning 계산은 약 30~100 ms가 걸렸고, 최초 Numba/JIT(실행 중 코드를 기계어로 컴파일하는 방식) 실행은 수 초까지 걸렸다.
3. planning 중 odom callback 처리가 밀렸다.
4. odom QoS queue(Quality of Service, ROS 메시지 보관·전달 정책)가 작아 밀린 중간 메시지가 유실됐다.
5. 결과적으로 odom은 100 Hz인데도 실제 `/drive` 출력 빈도는 낮아졌다.

추가로 다음 문제가 발견됐다.

- 상위 데이터 수집 launch와 포함된 Gazebo launch가 모두 `record` 인자를 사용했다. Foxy launch configuration에서 값이 충돌해 사용자가 `record:=true`를 지정해도 episode manager에는 `false`가 전달될 수 있었다.
- episode manager가 너무 일찍 시작하면 최초 JIT 워밍업 부하가 첫 번째 bag에 포함됐다.
- `ros2 topic hz`는 벽시계 기준이므로 Gazebo의 real-time factor(RTF, 실제 시간 대비 시뮬레이션 진행 속도)가 1보다 낮거나 높으면 시뮬레이션 시간 기준 제어 주기와 다르게 표시된다.

## 적용한 수정

### Planner와 controller 프로세스 분리

- `planner_node.py`에서는 lattice trajectory 계산만 수행한다.
- 선택한 trajectory를 `std_msgs/msg/Float64MultiArray` 형식의 `planned_trajectory` 토픽으로 발행한다.
- `pure_pursuit_controller_node.py`를 별도 ROS 프로세스로 실행한다.
- controller는 `odom`, `planned_trajectory`, `/episode/control`을 구독한다.
- trajectory가 준비된 상태에서는 odom 메시지마다 Pure Pursuit를 계산하고 `/drive`를 발행한다.
- odom QoS depth를 20으로 늘렸다.
- Pure Pursuit의 Numba 경로를 노드 시작 시 사전 워밍업한다.

### Timestamp(메시지에 기록된 시각) 처리

새 controller는 다음과 같이 triggering odom의 timestamp를 drive에 그대로 사용한다.

```python
drive_msg.header.stamp = odom_msg.header.stamp
```

이를 통해 다음 항목을 bag에서 직접 검사할 수 있다.

- 연속 drive stamp 간격
- odom과 drive의 stamp 대응 여부
- 중복 timestamp
- 역행 timestamp
- 10 ms보다 긴 누락 구간

### Episode 시작 지연

episode manager 시작을 launch 후 6초에서 15초로 늦췄다. 두 planner의 최초 JIT 경로가 충분히 워밍업된 뒤 첫 에피소드 녹화를 시작하기 위한 조치다.

### `record` 인자 충돌 방지

Gazebo include가 동일한 `record` launch configuration을 변경하기 전에 사용자의 데이터 녹화 설정을 `episode_record`라는 별도 configuration으로 보존한다. episode manager에는 보존된 값을 전달한다.

## 주기 측정 기준

최종 판정은 `ros2 topic hz`가 아니라 메시지의 `header.stamp`를 기준으로 한다.

연속 drive message의 timestamp를 `t[i]`라고 할 때 다음과 같이 계산한다.

```text
dt[i] = t[i + 1] - t[i]
rate = (N - 1) / (t[N - 1] - t[0])
```

100 Hz 합격 기준은 다음과 같다.

- 평균 `dt`가 약 0.010초
- 시뮬레이션 시간 기준 rate가 약 100 Hz
- 중복 또는 역행 stamp가 없음
- 0.010초보다 긴 누락 구간이 없음
- drive stamp가 대응하는 odom stamp에 존재함

벽시계 주기는 RTF의 영향을 받는다. 예를 들어 RTF가 0.78이면 시뮬레이션 기준 100 Hz 토픽이 벽시계 기준 약 78 Hz로 관측될 수 있다.

기존 데이터는 `/drive/header.stamp`가 `get_clock().now()`로 생성되어 같은 값이 반복되거나 양자화될 수 있으므로 주의해야 한다. 기존 bag 분석 시에는 drive header의 유효성을 먼저 검사하고, 유효하지 않으면 `/clock`에 매핑한 시뮬레이션 시각과 bag 수신시각을 보조 지표로 사용한다.

## 검증 결과

15초 워밍업을 적용한 2개 episode bag(ROS 메시지를 저장한 기록 파일)의 결과는 다음과 같다.

| Bag | 차량 | Odom 수 | Drive 수 | Sim-time rate | Drive dt | Gap/중복/역행 | Odom stamp 대응 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ep0000 | car1 | 836 | 808 | 100.000000 Hz | 모두 10 ms | 0 / 0 / 0 | 808 / 808 |
| ep0000 | car2 | 836 | 808 | 100.000000 Hz | 모두 10 ms | 0 / 0 / 0 | 808 / 808 |
| ep0001 | car1 | 832 | 808 | 100.000000 Hz | 모두 10 ms | 0 / 0 / 0 | 808 / 808 |
| ep0001 | car2 | 833 | 808 | 100.000000 Hz | 모두 10 ms | 0 / 0 / 0 | 808 / 808 |

실제 500-episode 수집의 완료 bag에서도 두 차량 모두 `header.stamp` 기준 100.000000 Hz, 모든 간격 10 ms, gap/중복 0을 확인했다.

## 실행 안전 조건

다른 사용자의 ROS/Gazebo 작업을 종료하거나 충돌시키지 않기 위해 수집은 다음 조건으로 격리했다.

```text
ROS_DOMAIN_ID=91
GAZEBO_MASTER_URI=http://127.0.0.1:11491
cleanup_gazebo:=false
```

종료할 때는 광범위한 `pkill`을 사용하지 않고, 수집 시작 시 기록한 PID(프로세스 식별 번호)/PGID(프로세스 그룹 식별 번호)의 프로세스 그룹만 대상으로 해야 한다.
