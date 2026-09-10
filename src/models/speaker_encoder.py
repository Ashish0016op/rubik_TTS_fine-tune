"""Speaker conditioning module for few-shot voice adaptation with minimal reference audio."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ReferenceAudioEncoder(nn.Module):
    """Convolutional audio encoder with attentive statistical pooling for extracting speaker embeddings."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 256,
        embedding_dim: int = 256,
        kernel_size: int = 5
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim // 2, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.conv4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
        self.bn4 = nn.BatchNorm1d(hidden_dim)

        # Attentive statistical pooling
        self.attention = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=1),
            nn.Softmax(dim=-1)
        )

        # Statistical pooling doubles dimension (mean + std)
        self.fc_out = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extracts speaker embedding.
        
        Args:
            waveform: [batch, 1, num_samples] or [batch, num_samples]
        Returns:
            embedding: [batch, embedding_dim] L2-normalized speaker vector.
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
            
        x = F.relu(self.bn1(self.conv1(waveform)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Attentive weights
        alpha = self.attention(x)
        # Mean
        mu = torch.sum(alpha * x, dim=-1)
        # Std
        sigma = torch.sqrt(torch.sum(alpha * (x ** 2), dim=-1) - (mu ** 2) + 1e-6)

        stats = torch.cat([mu, sigma], dim=-1)
        emb = self.fc_out(stats)
        return F.normalize(emb, p=2, dim=-1)


class SpeakerPrefixProjector(nn.Module):
    """Projects fixed-length speaker embedding into prefix token embeddings for the transformer backbone."""

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 2048,
        num_prefix_tokens: int = 4
    ):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_prefix_tokens * hidden_dim)
        )

    def forward(self, speaker_embedding: torch.Tensor) -> torch.Tensor:
        """Projects [batch, embedding_dim] to [batch, num_prefix_tokens, hidden_dim]."""
        batch_size = speaker_embedding.shape[0]
        projected = self.mlp(speaker_embedding)
        return projected.view(batch_size, self.num_prefix_tokens, self.hidden_dim)


class SpeakerConditioningModule(nn.Module):
    """Combined speaker audio encoder and prefix projector for zero/few-shot voice conditioning."""

    def __init__(
        self,
        embedding_dim: int = 256,
        transformer_hidden_dim: int = 2048,
        num_prefix_tokens: int = 4
    ):
        super().__init__()
        self.encoder = ReferenceAudioEncoder(embedding_dim=embedding_dim)
        self.projector = SpeakerPrefixProjector(
            embedding_dim=embedding_dim,
            hidden_dim=transformer_hidden_dim,
            num_prefix_tokens=num_prefix_tokens
        )

    def get_prefix_tokens(self, reference_audio: torch.Tensor) -> torch.Tensor:
        """Returns prefix tensor [batch, num_prefix_tokens, hidden_dim]."""
        emb = self.encoder(reference_audio)
        return self.projector(emb)
