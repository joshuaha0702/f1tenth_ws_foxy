import torch
import torch.nn as nn
from typing import Optional, Tuple

def speed_dims_from_state_dict(state_dict) -> int:
    """Width of the speed input of an End2Race checkpoint (plain state_dict): 1 for the
    'command'/'odom' models, 2 for train.py --speed_input hybrid ([odom, previous command]),
    0 for --speed_input none (LiDAR-only, no speed_mlp)."""
    if "speed_mlp.0.weight" not in state_dict:
        return 0
    return int(state_dict["speed_mlp.0.weight"].shape[1])


def hidden_scale_from_state_dict(state_dict, processed_features: int = 420) -> int:
    """GRU hidden-size multiplier of an End2Race checkpoint (plain state_dict).
    gru.weight_hh_l0 is (3*H, H) with H = processed_features * hidden_scale, where
    processed_features = 360 + 360 // 6 = 420 for End2Race (480 for End2RaceWithDelay)."""
    hidden = int(state_dict["gru.weight_hh_l0"].shape[1])
    if hidden % processed_features:
        raise ValueError(f"GRU hidden size {hidden} is not a multiple of {processed_features}")
    return hidden // processed_features


class End2Race(nn.Module):

    def __init__(self, mask_prob=0.0, hidden_scale=4, speed_dims=1):
        super(End2Race, self).__init__()
        
        # Store configuration
        num_features = 360
        num_actions=2
        self.mask_prob = mask_prob
        self.hidden_scale = hidden_scale
        self.speed_dims = speed_dims
        
        # Common: Learnable sensor preprocessing parameter
        k_init = (-1 / 10.0) * torch.log(torch.tensor(0.01) / (2 - torch.tensor(0.01)))
        self.k = nn.Parameter(torch.full((num_features,), k_init.item()))
        
        # Speed-specific modules. speed_dims == 0 (train.py --speed_input none): LiDAR-only,
        # the learnable dummy_embedding is fed at every step instead of a speed input, which
        # removes the "copy the speed input" shortcut; the GRU state has to carry the speed.
        self.speed_mlp = (nn.Sequential(nn.Linear(speed_dims, num_features // 6), nn.ReLU())
                          if speed_dims > 0 else None)
        self.dummy_embedding = nn.Parameter(torch.randn(1, num_features // 6))
        
        # Calculate processed feature size
        processed_features = num_features + num_features // 6
        
        # Common GRU architecture with mode-dependent dimensions
        self.gru = nn.GRU(
            input_size=processed_features,
            hidden_size=processed_features * hidden_scale,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Common output layer with mode-dependent dimensions
        self.output_layer = nn.Sequential(
            nn.Linear(processed_features * hidden_scale, processed_features),
            nn.ReLU(),
            nn.Linear(processed_features, num_actions)
        )
        
        # Initialize all parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize all parameters."""
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
        
        # Initialize dummy embedding
        nn.init.xavier_normal_(self.dummy_embedding)
    
    def forward(self, x: torch.Tensor, speed_input: Optional[torch.Tensor] = None, 
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with speed conditioning.
        
        Args:
            x: LiDAR input tensor [batch, seq_len, num_features]
            speed_input: Speed tensor [batch, seq_len, speed_dims]
            hidden: Hidden state from previous timestep
            
        Returns:
            actions: Predicted actions [steering, speed]
            last_hidden: Updated hidden state
        """
        # Process LiDAR with learnable sigmoid transformation
        processed_lidar = (-1 / (1 + torch.exp(-self.k * x)) + 1) * 2
        
        # Process speed input
        batch_size, seq_len, _ = x.shape
        if self.speed_mlp is None:
            speed_embedding = self.dummy_embedding.expand(batch_size, seq_len, -1)  # LiDAR-only
        else:
            speed_embedding = self.speed_mlp(speed_input)
        
        # Apply dummy embedding during training
        if self.speed_mlp is not None and self.training and self.mask_prob > 0:
            mask = torch.rand(batch_size, seq_len, 1, device=speed_input.device) < self.mask_prob
            mask_batch = self.dummy_embedding.expand(batch_size, seq_len, -1)
            speed_embedding = torch.where(mask, mask_batch, speed_embedding)
        
        # Concatenate features
        features = torch.cat([processed_lidar, speed_embedding], dim=2)
        
        # Forward pass through GRU and output layer
        gru_out, last_hidden = self.gru(features, hidden)
        actions = self.output_layer(gru_out)
        
        return actions, last_hidden


class End2RaceWithDelay(nn.Module):

    def __init__(self, mask_prob=0.0, hidden_scale=4):
        super(End2RaceWithDelay, self).__init__()

        num_features = 360
        num_actions = 2
        self.mask_prob = mask_prob
        self.hidden_scale = hidden_scale

        k_init = (-1 / 10.0) * torch.log(torch.tensor(0.01) / (2 - torch.tensor(0.01)))
        self.k = nn.Parameter(torch.full((num_features,), k_init.item()))

        embed_dim = num_features // 6

        self.speed_mlp = nn.Sequential(nn.Linear(1, embed_dim), nn.ReLU())
        self.delay_mlp = nn.Sequential(nn.Linear(1, embed_dim), nn.ReLU())
        self.dummy_embedding = nn.Parameter(torch.randn(1, embed_dim))

        processed_features = num_features + embed_dim * 2

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
            nn.Linear(processed_features, num_actions)
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