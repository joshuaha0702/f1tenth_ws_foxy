import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np


def speed_dims_from_state_dict(state_dict) -> int:
    """체크포인트(plain state_dict)의 속도 입력 폭: 'command'/'odom' 모델은 1,
    train.py --speed_input hybrid ([odom, 직전 명령]) 모델은 2."""
    return int(state_dict["speed_mlp.0.weight"].shape[1])


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
        
        # 2. Speed 처리용 MLP
        self.speed_mlp = nn.Sequential(
            nn.Linear(speed_dims, self.num_features // 6),
            nn.ReLU()
        )
        self.dummy_embedding = nn.Parameter(torch.randn(1, self.num_features // 6))
        
        # 3. GRU 입력 사이즈 계산
        processed_features = self.num_features + self.num_features // 6
        
        # 4. GRU 아키텍처
        self.gru = nn.GRU(
            input_size=processed_features,
            hidden_size=processed_features * hidden_scale,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # 5. Output Layer
        self.output_layer = nn.Sequential(
            nn.Linear(processed_features * hidden_scale, processed_features),
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
        speed_embedding = self.speed_mlp(speed_input)
        
        # 훈련 중 마스킹 전략 유지
        if self.training and self.mask_prob > 0:
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
        self._s_buf = torch.zeros(1, 1, self.speed_dims, dtype=torch.float32, device=device)

    @torch.inference_mode()
    def inference_step(self, lidar_data: np.ndarray, current_speed,
                       prev_hidden: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, torch.Tensor]:
        """실시간 추론 함수"""
        self._x_buf.copy_(torch.as_tensor(lidar_data, dtype=torch.float32).view(1, 1, -1))
        if np.ndim(current_speed) == 0:
            if self.speed_dims != 1:
                raise ValueError(f"speed_dims={self.speed_dims} 모델에는 길이 {self.speed_dims} 속도 벡터가 필요합니다")
            self._s_buf[0, 0, 0] = float(current_speed)
        else:
            self._s_buf.copy_(torch.as_tensor(current_speed, dtype=torch.float32).view(1, 1, self.speed_dims))

        actions_tensor, next_hidden = self.forward(self._x_buf, self._s_buf, prev_hidden)
        actions = actions_tensor.squeeze().cpu().numpy()
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

        self.gru = nn.GRU(
            input_size=processed_features,
            hidden_size=processed_features * hidden_scale,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.output_layer = nn.Sequential(
            nn.Linear(processed_features * hidden_scale, processed_features),
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
                 max_steer=0.52, max_speed=2.0):
        super(End2RaceNorm, self).__init__()

        # 물리 제약 상수 (정규화/역정규화용)
        self.max_steer = max_steer
        self.max_speed = max_speed

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
        self.gru = nn.GRU(
            input_size=processed_features,
            hidden_size=processed_features * hidden_scale,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 5. Output Layer with Tanh (출력 범위 [-1.0, 1.0])
        self.output_layer = nn.Sequential(
            nn.Linear(processed_features * hidden_scale, processed_features),
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

        actions_tensor, next_hidden = self.forward(self._x_buf, self._s_buf, prev_hidden)
        actions = actions_tensor.squeeze().cpu().numpy()

        if unnormalize:
            # [-1, 1] -> 물리 단위 [rad, m/s] 복원
            steer = actions[0] * self.max_steer
            # 속도는 양수(전진)이므로 clamp(0.0) 후 max_speed 곱
            speed = np.clip(actions[1], 0.0, 1.0) * self.max_speed
            return np.array([steer, speed], dtype=np.float32), next_hidden

        return actions, next_hidden
