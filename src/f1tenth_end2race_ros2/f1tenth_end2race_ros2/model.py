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


def hidden_scale_from_state_dict(state_dict, processed_features: int = 420) -> int:
    """체크포인트(plain state_dict)의 GRU hidden 배수.
    gru.weight_hh_l0 은 (3*H, H), H = processed_features * hidden_scale,
    processed_features = 360 + 360 // 6 = 420."""
    hidden = int(state_dict["gru.weight_hh_l0"].shape[1])
    if hidden % processed_features:
        raise ValueError(f"GRU hidden 크기 {hidden}가 {processed_features}의 배수가 아닙니다")
    return hidden // processed_features


class End2Race(nn.Module):
    def __init__(self, mask_prob=0.0, hidden_scale=4, speed_dims=1):
        super(End2Race, self).__init__()
        
        # 모델 설정 변수 (기본 구조 유지)
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
        
        # 가중치 초기화 실행
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """네트워크의 가중치를 Xavier 및 Zero 초기화합니다."""
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
        """
        훈련 및 시퀀스 처리를 위한 기본 Forward pass
        """
        # LiDAR 전처리 (Sigmoid 변환) - 아키텍처 로직 유지
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
        # 입력 텐서를 미리 할당해 매 스텝 allocation 제거
        self._x_buf = torch.zeros(1, 1, self.num_features, dtype=torch.float32, device=device)
        self._s_buf = torch.zeros(1, 1, max(self.speed_dims, 1), dtype=torch.float32, device=device)

    @torch.inference_mode()
    def inference_step(self, lidar_data: np.ndarray, current_speed,
                       prev_hidden: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, torch.Tensor]:
        """
        ROS 2 Gazebo 환경에서 매 프레임마다 호출하는 실시간 추론 함수
        current_speed: float (speed_dims=1) 또는 길이 speed_dims 벡터 (hybrid: [odom, 직전 명령])
        """
        self._x_buf.copy_(torch.as_tensor(lidar_data, dtype=torch.float32).view(1, 1, -1))
        if self.speed_dims == 0:
            pass  # LiDAR 전용 모델: current_speed 무시
        elif np.ndim(current_speed) == 0:
            if self.speed_dims != 1:
                raise ValueError(f"speed_dims={self.speed_dims} 모델에는 길이 {self.speed_dims} 속도 벡터가 필요합니다")
            self._s_buf[0, 0, 0] = float(current_speed)
        else:
            self._s_buf.copy_(torch.as_tensor(current_speed, dtype=torch.float32).view(1, 1, self.speed_dims))

        actions_tensor, next_hidden = self.forward(self._x_buf, self._s_buf, prev_hidden)

        actions = actions_tensor.squeeze().cpu().numpy()
        return actions, next_hidden
