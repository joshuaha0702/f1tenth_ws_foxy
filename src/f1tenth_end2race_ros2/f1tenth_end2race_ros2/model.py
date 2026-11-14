import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np


def speed_dims_from_state_dict(state_dict) -> int:
    """체크포인트(plain state_dict)의 속도 입력 폭: 'command'/'odom' 모델은 1,
    train.py --speed_input hybrid ([odom, 직전 명령]) 모델은 2,
    --speed_input none (LiDAR 전용, speed_mlp 없음) 모델은 0."""
    if "speed_mlp.0.weight" not in state_dict:
        return 0
    return int(state_dict["speed_mlp.0.weight"].shape[1])


def hidden_scale_from_state_dict(state_dict, processed_features: int = 420):
    """체크포인트(plain state_dict)의 GRU hidden 배수.
    gru.weight_hh_l0 은 (3*H, H), H = int(round(processed_features * hidden_scale)),
    processed_features = 360 + 360 // 6 = 420 (End2RaceWithDelay 는 480)."""
    hidden = int(state_dict["gru.weight_hh_l0"].shape[1])
    scale = hidden / processed_features
    if abs(scale - round(scale)) < 1e-4:
        return int(round(scale))
    return scale


class _CUDAGraphRunner:
    """고정 shape CUDA Graph 재실행 래퍼."""
    def __init__(self, graph, h_buf: torch.Tensor, out_buf: torch.Tensor, h_out_buf: torch.Tensor):
        self.graph = graph
        self.h_buf = h_buf
        self.out_buf = out_buf
        self.h_out_buf = h_out_buf

    def __call__(self, prev_hidden: Optional[torch.Tensor]):
        if prev_hidden is None:
            self.h_buf.zero_()
        else:
            self.h_buf.copy_(prev_hidden)
        self.graph.replay()
        return self.out_buf, self.h_out_buf


class _TracedRunner:
    """TorchScript trace+freeze 실행 래퍼 (CPU 및 CUDA Graph 미지원 시 폴백)."""
    def __init__(self, traced, x_buf: torch.Tensor, s_buf: torch.Tensor, h0: torch.Tensor):
        self.traced = traced
        self.x_buf = x_buf
        self.s_buf = s_buf
        self.h0 = h0

    def __call__(self, prev_hidden: Optional[torch.Tensor]):
        h_in = self.h0 if prev_hidden is None else prev_hidden
        return self.traced(self.x_buf, self.s_buf, h_in)


def _compile_fast_forward(model: nn.Module, x_buf: torch.Tensor, s_buf: torch.Tensor):
    """GRU 실시간 추론용 CUDA Graph / TorchScript 컴파일 및 워밍업."""
    if not hasattr(model, "gru"):
        return None
    h_buf = torch.zeros(model.gru.num_layers, 1, model.gru.hidden_size,
                        dtype=torch.float32, device=x_buf.device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            traced = torch.jit.freeze(torch.jit.trace(model, (x_buf, s_buf, h_buf)))
            for _ in range(2):
                traced(x_buf, s_buf, h_buf)
            if x_buf.is_cuda and hasattr(torch.cuda, "CUDAGraph"):
                try:
                    stream = torch.cuda.Stream(device=x_buf.device)
                    stream.wait_stream(torch.cuda.current_stream(x_buf.device))
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            traced(x_buf, s_buf, h_buf)
                    torch.cuda.current_stream(x_buf.device).wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        out_buf, h_out_buf = traced(x_buf, s_buf, h_buf)
                    graph.replay()
                    torch.cuda.synchronize(x_buf.device)
                    return _CUDAGraphRunner(graph, h_buf, out_buf, h_out_buf)
                except Exception:
                    pass
            if x_buf.is_cuda:
                torch.cuda.synchronize(x_buf.device)
            return _TracedRunner(traced, x_buf, s_buf, h_buf)
    except Exception:
        return None
    finally:
        if was_training:
            model.train()


class End2Race(nn.Module):
    def __init__(self, mask_prob=0.0, hidden_scale=4, speed_dims=1):
        super(End2Race, self).__init__()
        
        # 모델 설정 변수
        self.num_features = 360 # LiDAR 데이터 개수
        self.num_actions = 2   # [Steering, Speed]
        self.mask_prob = mask_prob
        self.hidden_scale = hidden_scale
        self.speed_dims = speed_dims
        
        # 1. 센서 전처리 파라미터 (Learnable Parameter k)
        k_init = (-1 / 10.0) * torch.log(torch.tensor(0.01) / (2 - torch.tensor(0.01)))
        self.k = nn.Parameter(torch.full((self.num_features,), k_init.item()))
        
        # 2. Speed 처리용 MLP. speed_dims=0 (train.py --speed_input none) 이면 속도 입력 없이
        #    모든 스텝에 학습형 dummy_embedding 을 넣는다 (LiDAR 전용: 속도 입력을 복사하는
        #    지름길을 구조적으로 차단, 속도 상태는 GRU 메모리가 추정).
        self.speed_mlp = (nn.Sequential(nn.Linear(speed_dims, self.num_features // 6), nn.ReLU())
                          if speed_dims > 0 else None)
        self.dummy_embedding = nn.Parameter(torch.randn(1, self.num_features // 6))
        
        # 3. GRU 입력 사이즈 계산
        processed_features = self.num_features + self.num_features // 6
        
        # 4. GRU 아키텍처
        hidden_dim = int(round(processed_features * hidden_scale))
        self.gru = nn.GRU(
            input_size=processed_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # 5. Output Layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, processed_features),
            nn.ReLU(),
            nn.Linear(processed_features, self.num_actions)
        )
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """네트워크 가중치 Xavier 및 Zero 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        
        nn.init.xavier_normal_(self.dummy_embedding)
    
    def forward(self, x: torch.Tensor, speed_input: Optional[torch.Tensor] = None, 
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # LiDAR 전처리 (Sigmoid 변환)
        processed_lidar = (-1 / (1 + torch.exp(-self.k * x)) + 1) * 2
        
        batch_size, seq_len, _ = x.shape
        if self.speed_mlp is None:
            # LiDAR 전용: 속도 입력은 무시(있어도 쓰지 않음)
            speed_embedding = self.dummy_embedding.expand(batch_size, seq_len, -1)
        else:
            speed_embedding = self.speed_mlp(speed_input)
        
        # 훈련 중 마스킹 전략 유지
        if self.speed_mlp is not None and self.training and self.mask_prob > 0:
            mask = torch.rand(batch_size, seq_len, 1, device=speed_input.device) < self.mask_prob
            mask_batch = self.dummy_embedding.expand(batch_size, seq_len, -1)
            speed_embedding = torch.where(mask, mask_batch, speed_embedding)
        
        # 특징 결합 (LiDAR + Speed)
        features = torch.cat([processed_lidar, speed_embedding], dim=2)
        
        # GRU 및 출력층 통과
        gru_out, last_hidden = self.gru(features, hidden)
        actions = self.output_layer(gru_out)
        
        return actions, last_hidden

    def set_inference_device(self, device: torch.device):
        """추론 시 사용할 device를 캐싱 (init 후 1회 호출)"""
        self._infer_device = device
        self._x_buf = torch.zeros(1, 1, self.num_features, dtype=torch.float32, device=device)
        self._s_buf = torch.zeros(1, 1, max(self.speed_dims, 1), dtype=torch.float32, device=device)
        self._fast_forward = _compile_fast_forward(self, self._x_buf, self._s_buf)

    @torch.inference_mode()
    def inference_step(self, lidar_data: np.ndarray, current_speed,
                       prev_hidden: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, torch.Tensor]:
        """실시간 추론 함수"""
        self._x_buf.copy_(torch.as_tensor(lidar_data, dtype=torch.float32).view(1, 1, -1))
        if self.speed_dims == 0:
            pass  # LiDAR 전용 모델: current_speed 무시
        elif np.ndim(current_speed) == 0:
            if self.speed_dims != 1:
                raise ValueError(f"speed_dims={self.speed_dims} 모델에는 길이 {self.speed_dims} 속도 벡터가 필요합니다")
            self._s_buf[0, 0, 0] = float(current_speed)
        else:
            self._s_buf.copy_(torch.as_tensor(current_speed, dtype=torch.float32).view(1, 1, self.speed_dims))

        if getattr(self, "_fast_forward", None) is not None:
            actions_tensor, next_hidden = self._fast_forward(prev_hidden)
        else:
            actions_tensor, next_hidden = self.forward(self._x_buf, self._s_buf, prev_hidden)
        actions = actions_tensor.squeeze().cpu().numpy()
        return actions, next_hidden


def is_mlp_state_dict(state_dict) -> bool:
    """End2RaceMLP 체크포인트 여부 (GRU 대신 mlp.* 가중치)."""
    return "mlp.0.weight" in state_dict


class End2RaceMLP(End2Race):
    """어블레이션용: End2Race 의 GRU 를 MLP 3층으로 바꾼 모델 (나머지 전처리·속도 임베딩·출력층 동일).

    stack_frames K > 1 이면 현재 scan 과 stack_stride 행 간격의 과거 scan K-1 장을 이어 붙여 입력한다
    (100 Hz 행 기준 stride 10 = 0.1 s). 과거 scan 버퍼를 hidden 자리로 주고받으므로 train.py 와
    agent_node 의 hidden 처리를 그대로 쓴다. 버퍼가 없을 때(에피소드 시작)는 첫 scan 을 복제해 채운다.
    K = 1 이면 메모리 없는 단일 프레임 MLP (hidden 은 항상 None).
    """

    def __init__(self, mask_prob=0.0, speed_dims=1, mlp_width=2176, stack_frames=1, stack_stride=10):
        nn.Module.__init__(self)
        self.num_features = 360
        self.num_actions = 2
        self.mask_prob = mask_prob
        self.speed_dims = speed_dims
        self.stack_frames = int(stack_frames)
        self.stack_stride = int(stack_stride)
        self.hist_len = (self.stack_frames - 1) * self.stack_stride
        # 체크포인트에 스택 설정을 남겨 agent_node 가 읽게 한다
        self.register_buffer("stack_cfg", torch.tensor([self.stack_frames, self.stack_stride]))

        k_init = (-1 / 10.0) * torch.log(torch.tensor(0.01) / (2 - torch.tensor(0.01)))
        self.k = nn.Parameter(torch.full((self.num_features,), k_init.item()))
        self.speed_mlp = (nn.Sequential(nn.Linear(speed_dims, self.num_features // 6), nn.ReLU())
                          if speed_dims > 0 else None)
        self.dummy_embedding = nn.Parameter(torch.randn(1, self.num_features // 6))

        in_features = self.num_features * self.stack_frames + self.num_features // 6
        self.mlp = nn.Sequential(
            nn.Linear(in_features, mlp_width), nn.ReLU(),
            nn.Linear(mlp_width, mlp_width), nn.ReLU(),
            nn.Linear(mlp_width, mlp_width), nn.ReLU(),
        )
        processed_features = self.num_features + self.num_features // 6
        self.output_layer = nn.Sequential(
            nn.Linear(mlp_width, processed_features),
            nn.ReLU(),
            nn.Linear(processed_features, self.num_actions)
        )
        self._initialize_parameters()

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        frames, stride = (int(v) for v in state_dict["stack_cfg"].tolist())
        return cls(speed_dims=speed_dims_from_state_dict(state_dict),
                   mlp_width=int(state_dict["mlp.0.weight"].shape[0]),
                   stack_frames=frames, stack_stride=stride, **kwargs)

    def forward(self, x: torch.Tensor, speed_input: Optional[torch.Tensor] = None,
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        processed_lidar = (-1 / (1 + torch.exp(-self.k * x)) + 1) * 2

        batch_size, seq_len, _ = x.shape
        if self.speed_mlp is None:
            speed_embedding = self.dummy_embedding.expand(batch_size, seq_len, -1)
        else:
            speed_embedding = self.speed_mlp(speed_input)
        if self.speed_mlp is not None and self.training and self.mask_prob > 0:
            mask = torch.rand(batch_size, seq_len, 1, device=speed_input.device) < self.mask_prob
            mask_batch = self.dummy_embedding.expand(batch_size, seq_len, -1)
            speed_embedding = torch.where(mask, mask_batch, speed_embedding)

        # 프레임 스택: hidden = 직전 hist_len 행의 전처리 scan [B, hist_len, 360]
        P = self.hist_len
        if P > 0:
            hist = processed_lidar[:, :1].expand(-1, P, -1) if hidden is None else hidden
            full = torch.cat([hist, processed_lidar], dim=1)
            frames = [full[:, P - j * self.stack_stride:P - j * self.stack_stride + seq_len]
                      for j in range(self.stack_frames)]
            lidar_features = torch.cat(frames, dim=2)
            next_hidden = full[:, -P:]
        else:
            lidar_features = processed_lidar
            next_hidden = None

        features = torch.cat([lidar_features, speed_embedding], dim=2)
        actions = self.output_layer(self.mlp(features))
        return actions, next_hidden


class End2RaceWithDelay(nn.Module):
    def __init__(self, mask_prob=0.0, hidden_scale=4):
        super(End2RaceWithDelay, self).__init__()

        self.num_features = 360
        self.num_actions = 2
        self.mask_prob = mask_prob
        self.hidden_scale = hidden_scale

        k_init = (-1 / 10.0) * torch.log(torch.tensor(0.01) / (2 - torch.tensor(0.01)))
        self.k = nn.Parameter(torch.full((self.num_features,), k_init.item()))

        embed_dim = self.num_features // 6

        self.speed_mlp = nn.Sequential(nn.Linear(1, embed_dim), nn.ReLU())
        self.delay_mlp = nn.Sequential(nn.Linear(1, embed_dim), nn.ReLU())
        self.dummy_embedding = nn.Parameter(torch.randn(1, embed_dim))

        processed_features = self.num_features + embed_dim * 2

        hidden_dim = int(round(processed_features * hidden_scale))
        self.gru = nn.GRU(
            input_size=processed_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, processed_features),
            nn.ReLU(),
            nn.Linear(processed_features, self.num_actions)
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        nn.init.xavier_normal_(self.dummy_embedding)

    def forward(self, x: torch.Tensor, speed_input: Optional[torch.Tensor] = None,
                delay_input: Optional[torch.Tensor] = None,
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        processed_lidar = (-1 / (1 + torch.exp(-self.k * x)) + 1) * 2

        batch_size, seq_len, _ = x.shape
        speed_embedding = self.speed_mlp(speed_input)
        delay_embedding = self.delay_mlp(delay_input)

        if self.training and self.mask_prob > 0:
            mask = torch.rand(batch_size, seq_len, 1, device=speed_input.device) < self.mask_prob
            mask_batch = self.dummy_embedding.expand(batch_size, seq_len, -1)
            speed_embedding = torch.where(mask, mask_batch, speed_embedding)

        features = torch.cat([processed_lidar, speed_embedding, delay_embedding], dim=2)

        gru_out, last_hidden = self.gru(features, hidden)
        actions = self.output_layer(gru_out)

        return actions, last_hidden


class End2RaceNorm(nn.Module):
    """
    정규화(Normalization)가 적용된 End2Race 모델:
    1. LiDAR: 학습 가능한 Inverse-Sigmoid 비선형 정규화 ([0, 1] 범위)
    2. Speed Input: max_speed(2.0 m/s) 기준 [0, 1] 스케일링 정규화
    3. Feature Fusion: LiDAR + Speed 임베딩 결합 후 nn.LayerNorm 적용
    4. Output: nn.Tanh()를 통해 [Steering, Speed] 모두 [-1.0, 1.0] 범위로 정규화 출력
    """

    def __init__(self, mask_prob=0.0, hidden_scale=4, speed_dims=1,
                 max_steer=0.28, max_speed=2.0, min_speed=0.0):
        super(End2RaceNorm, self).__init__()

        # 물리 제약 상수 (정규화/역정규화용)
        self.max_steer = max_steer
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.speed_mid = 0.5 * (max_speed + min_speed)   # 1.25 m/s
        self.speed_half = 0.5 * (max_speed - min_speed)  # 0.75 m/s

        self.num_features = 360  # LiDAR 빔 개수
        self.num_actions = 2     # [Steering, Speed]
        self.mask_prob = mask_prob
        self.hidden_scale = hidden_scale
        self.speed_dims = speed_dims

        # 1. LiDAR 센서 전처리 파라미터 (Learnable Parameter k)
        k_init = (-1 / 10.0) * torch.log(torch.tensor(0.01) / (2 - torch.tensor(0.01)))
        self.k = nn.Parameter(torch.full((self.num_features,), k_init.item()))

        # 2. Speed 처리용 MLP (입력은 max_speed로 나누어진 [0, 1] 정규화 값)
        embed_dim = self.num_features // 6
        self.speed_mlp = nn.Sequential(
            nn.Linear(speed_dims, embed_dim),
            nn.ReLU()
        )
        self.dummy_embedding = nn.Parameter(torch.randn(1, embed_dim))

        # 3. Feature Fusion & Layer Normalization
        processed_features = self.num_features + embed_dim
        self.feature_norm = nn.LayerNorm(processed_features)

        # 4. GRU Architecture
        hidden_dim = int(round(processed_features * hidden_scale))
        self.gru = nn.GRU(
            input_size=processed_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 5. Output Layer with Tanh (출력 범위 [-1.0, 1.0])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, processed_features),
            nn.ReLU(),
            nn.Linear(processed_features, self.num_actions),
            nn.Tanh()
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        """네트워크 가중치 Xavier 및 Zero 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        nn.init.xavier_normal_(self.dummy_embedding)

    def forward(self, x: torch.Tensor, speed_input: Optional[torch.Tensor] = None,
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. LiDAR Inverse-Sigmoid 비선형 정규화 ([0, 1] 범위)
        processed_lidar = (-1 / (1 + torch.exp(-self.k * x)) + 1) * 2

        # 2. Speed 정규화 (m/s -> [0, 1] 스케일링) 후 임베딩
        batch_size, seq_len, _ = x.shape
        speed_norm = speed_input / self.max_speed
        speed_embedding = self.speed_mlp(speed_norm)

        # 마스킹 전략 (학습 시)
        if self.training and self.mask_prob > 0:
            mask = torch.rand(batch_size, seq_len, 1, device=speed_input.device) < self.mask_prob
            mask_batch = self.dummy_embedding.expand(batch_size, seq_len, -1)
            speed_embedding = torch.where(mask, mask_batch, speed_embedding)

        # 3. 결합 및 LayerNorm 적용 (LiDAR와 Speed의 스케일 균형 및 안정화)
        raw_features = torch.cat([processed_lidar, speed_embedding], dim=2)
        features = self.feature_norm(raw_features)

        # 4. GRU & Output (Tanh를 통해 [-1, 1] 출력)
        gru_out, last_hidden = self.gru(features, hidden)
        actions = self.output_layer(gru_out)

        return actions, last_hidden

    def set_inference_device(self, device: torch.device):
        """추론 시 사용할 device 캐싱 및 버퍼 사전 할당"""
        self._infer_device = device
        self._x_buf = torch.zeros(1, 1, self.num_features, dtype=torch.float32, device=device)
        self._s_buf = torch.zeros(1, 1, self.speed_dims, dtype=torch.float32, device=device)
        self._fast_forward = _compile_fast_forward(self, self._x_buf, self._s_buf)

    @torch.inference_mode()
    def inference_step(self, lidar_data: np.ndarray, current_speed,
                       prev_hidden: Optional[torch.Tensor] = None,
                       unnormalize: bool = True) -> Tuple[np.ndarray, torch.Tensor]:
        """
        실시간 추론 함수:
        unnormalize=True인 경우, [-1, 1] 출력을 물리 단위 [rad, m/s]로 복원하여 반환
        unnormalize=False인 경우, 정규화된 [-1, 1] 값을 그대로 반환
        """
        self._x_buf.copy_(torch.as_tensor(lidar_data, dtype=torch.float32).view(1, 1, -1))
        if np.ndim(current_speed) == 0:
            if self.speed_dims != 1:
                raise ValueError(f"speed_dims={self.speed_dims} 모델에는 길이 {self.speed_dims} 속도 벡터가 필요합니다")
            self._s_buf[0, 0, 0] = float(current_speed)
        else:
            self._s_buf.copy_(torch.as_tensor(current_speed, dtype=torch.float32).view(1, 1, self.speed_dims))

        if getattr(self, "_fast_forward", None) is not None:
            actions_tensor, next_hidden = self._fast_forward(prev_hidden)
        else:
            actions_tensor, next_hidden = self.forward(self._x_buf, self._s_buf, prev_hidden)
        actions = actions_tensor.squeeze().cpu().numpy()

        if unnormalize:
            # [-1, 1] -> 물리 단위 [rad, m/s] 복원
            # 조향: [-1, 1] -> [-max_steer, +max_steer] ([-0.28, +0.28] rad)
            steer = actions[0] * self.max_steer
            # 속도: [-1, 1] -> [min_speed, max_speed] ([0.5, 2.0] m/s, 0.75 * a + 1.25)
            speed = actions[1] * self.speed_half + self.speed_mid
            speed = np.clip(speed, self.min_speed, self.max_speed)
            return np.array([steer, speed], dtype=np.float32), next_hidden

        return actions, next_hidden
