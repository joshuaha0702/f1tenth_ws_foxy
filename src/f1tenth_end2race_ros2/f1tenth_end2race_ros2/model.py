import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np

class End2Race(nn.Module):
    def __init__(self, mask_prob=0.0, hidden_scale=4):
        super(End2Race, self).__init__()
        
        # 모델 설정 변수 (기본 구조 유지)
        self.num_features = 360 # LiDAR 데이터 개수
        self.num_actions = 2   # [Steering, Speed]
        self.mask_prob = mask_prob
        self.hidden_scale = hidden_scale
        
        # 1. 센서 전처리 파라미터 (Learnable Parameter k)
        k_init = (-1 / 10.0) * torch.log(torch.tensor(0.01) / (2 - torch.tensor(0.01)))
        self.k = nn.Parameter(torch.full((self.num_features,), k_init.item()))
        
        # 2. Speed 처리용 MLP
        self.speed_mlp = nn.Sequential(
            nn.Linear(1, self.num_features // 6),
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

    @torch.no_grad()
    def inference_step(self, lidar_data: np.ndarray, current_speed: float, 
                       prev_hidden: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, torch.Tensor]:
        """
        ROS 2 Gazebo 환경에서 매 프레임마다 호출하는 실시간 추론 함수
        """
        self.eval() # 평가 모드 강제
        
        # 1. 모델이 현재 위치한 장치(CPU/GPU) 확인
        device = next(self.parameters()).device
        
        # 2. 입력 데이터를 Tensor로 변환 및 차원 확장 [Batch=1, Seq=1, Feature]
        # Gazebo의 numpy 데이터를 모델 장치에 맞게 전송
        x_tensor = torch.as_tensor(lidar_data, dtype=torch.float32, device=device).view(1, 1, -1)
        speed_tensor = torch.as_tensor([[current_speed]], dtype=torch.float32, device=device).view(1, 1, 1)
        
        # 3. 모델 추론 (기존 forward 호출)
        actions_tensor, next_hidden = self.forward(x_tensor, speed_tensor, prev_hidden)
        
        # 4. 결과를 numpy 배열로 변환하여 반환
        actions = actions_tensor.squeeze().cpu().numpy()
        
        return actions, next_hidden
