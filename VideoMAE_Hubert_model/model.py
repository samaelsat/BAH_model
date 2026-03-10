"""
BAH A/H Recognition - Multimodal Model
ABAW10 @ CVPR 2026 Challenge

Architecture:
  1. VideoMAE   → encode 2×16-frame clips → [B, T_v, D]
  2. HuBERT     → encode raw waveform     → [B, T_a, D]
  3. Projection layers align both to d_model=768
  4. Cross-Modal Transformer (2 layers)
       - Video attends to Audio (cross-attn)
       - Audio attends to Video (cross-attn)
  5. Temporal Aggregation
       - Audio: attention-weighted pooling → [B, D]
       - Video: fed to LSTM as temporal sequence
  6. BiLSTM   → [B, T_v, 2*H]  → mean pool → [B, 2*H]
  7. Fusion   → concat LSTM + audio_global → Linear → [B, D]
  8. Classifier → Linear → [B, 2]
"""

import torch
import torch.nn as nn
from transformers import VideoMAEModel, HubertModel


# ────────────────────────────────────────────────────────────────────────────────
# Cross-Modal Transformer
# ────────────────────────────────────────────────────────────────────────────────

class CrossModalLayer(nn.Module):
    """
    One cross-modal attention layer:
      - video tokens attend to audio tokens (V→A cross-attention)
      - audio tokens attend to video tokens (A→V cross-attention)
    Each branch has its own FFN and LayerNorm.
    """

    def __init__(self, d_model: int = 768, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        # Video ← Audio cross-attention
        self.v2a = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_v1 = nn.LayerNorm(d_model)
        self.ffn_v = self._ffn(d_model, dropout)
        self.norm_v2 = nn.LayerNorm(d_model)

        # Audio ← Video cross-attention
        self.a2v = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_a1 = nn.LayerNorm(d_model)
        self.ffn_a = self._ffn(d_model, dropout)
        self.norm_a2 = nn.LayerNorm(d_model)

    @staticmethod
    def _ffn(d: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 4, d),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        v: torch.Tensor,                  # [B, T_v, D]
        a: torch.Tensor,                  # [B, T_a, D]
        a_key_pad_mask: torch.Tensor | None = None,  # [B, T_a] True=ignore
    ):
        # ── Video branch ─────────────────────────────────────────────────────
        v_cross, _ = self.v2a(
            query=v, key=a, value=a,
            key_padding_mask=a_key_pad_mask,
        )
        v = self.norm_v1(v + v_cross)
        v = self.norm_v2(v + self.ffn_v(v))

        # ── Audio branch ─────────────────────────────────────────────────────
        a_cross, _ = self.a2v(query=a, key=v, value=v)
        a = self.norm_a1(a + a_cross)
        a = self.norm_a2(a + self.ffn_a(a))

        return v, a


class CrossModalTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])

    def forward(self, v, a, a_key_pad_mask=None):
        for layer in self.layers:
            v, a = layer(v, a, a_key_pad_mask)
        return v, a


# ────────────────────────────────────────────────────────────────────────────────
# Main Model
# ────────────────────────────────────────────────────────────────────────────────

class AHVideoClassifier(nn.Module):
    """
    Multimodal A/H Recognition:
    VideoMAE + HuBERT + Cross-Modal Transformer + BiLSTM + Classifier
    """

    def __init__(
        self,
        video_model_name: str = "MCG-NJU/videomae-base",
        audio_model_name: str = "facebook/hubert-base-ls960",
        d_model: int = 768,
        nhead: int = 8,
        num_cross_layers: int = 2,
        dropout: float = 0.3,
        lstm_hidden: int = 512,
        lstm_layers: int = 2,
        lstm_bidirectional: bool = True,
        num_classes: int = 2,
        num_frames_per_clip: int = 16,   # VideoMAE is pretrained on 16 frames
        freeze_video_backbone: bool = False,
        freeze_audio_backbone: bool = False,
    ):
        super().__init__()

        self.num_frames_per_clip = num_frames_per_clip
        self.d_model = d_model

        # ── Backbones ─────────────────────────────────────────────────────────
        self.video_backbone = VideoMAEModel.from_pretrained(video_model_name)
        video_dim = self.video_backbone.config.hidden_size        # 768
        tubelet   = self.video_backbone.config.tubelet_size       # 2
        patch_sz  = self.video_backbone.config.patch_size         # 16
        self._tubelet  = tubelet
        self._patch_sz = patch_sz

        self.audio_backbone = HubertModel.from_pretrained(audio_model_name)
        audio_dim = self.audio_backbone.config.hidden_size        # 768

        if freeze_video_backbone:
            for p in self.video_backbone.parameters():
                p.requires_grad = False
        if freeze_audio_backbone:
            for p in self.audio_backbone.parameters():
                p.requires_grad = False

        # ── Projection ────────────────────────────────────────────────────────
        # Project backbones to shared d_model (no-op if already 768)
        self.video_proj = (nn.Linear(video_dim, d_model)
                           if video_dim != d_model else nn.Identity())
        self.audio_proj = (nn.Linear(audio_dim, d_model)
                           if audio_dim != d_model else nn.Identity())

        # ── Cross-Modal Transformer ───────────────────────────────────────────
        self.cross_modal = CrossModalTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_cross_layers,
            dropout=0.1,
        )

        # ── Audio temporal attention pooling ──────────────────────────────────
        self.audio_attn_pool = nn.Linear(d_model, 1)

        # ── BiLSTM temporal transformation + classification ───────────────────
        lstm_out_dim = lstm_hidden * (2 if lstm_bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=lstm_bidirectional,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Fuse LSTM output + audio global context
        self.fusion_head = nn.Sequential(
            nn.Linear(lstm_out_dim + d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_new_params()

    def _init_new_params(self):
        """Xavier init for all non-backbone layers."""
        for module in [self.cross_modal, self.fusion_head, self.classifier,
                       self.audio_attn_pool]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: [B, T_total, C, H, W]   T_total = 2 × num_frames_per_clip (32)
        Returns: [B, T_v, d_model]       T_v  = 2 × (clip_frames / tubelet)
        """
        B, T_total, C, H, W = frames.shape
        num_clips = T_total // self.num_frames_per_clip   # = 2

        clip_feats = []
        for i in range(num_clips):
            clip = frames[:, i * self.num_frames_per_clip:
                             (i + 1) * self.num_frames_per_clip]   # [B, 16, C, H, W]

            # VideoMAEModel expects pixel_values: [B, T, C, H, W]
            out  = self.video_backbone(pixel_values=clip)
            feat = out.last_hidden_state                 # [B, N_patches, video_dim]

            # Reshape: N_patches = T_temp * spatial_patches
            T_temp = self.num_frames_per_clip // self._tubelet   # 16/2 = 8
            spatial = feat.shape[1] // T_temp                    # 196 for 224^2 / 16^2

            feat = feat.view(B, T_temp, spatial, -1)
            feat = feat.mean(dim=2)                              # [B, 8, video_dim]

            clip_feats.append(feat)

        video_feat = torch.cat(clip_feats, dim=1)               # [B, 16, video_dim]
        return self.video_proj(video_feat)                       # [B, 16, d_model]

    def _encode_audio(
        self,
        waveforms: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        waveforms: [B, T_audio]
        Returns: [B, T_a, d_model]
        """
        out = self.audio_backbone(
            input_values=waveforms.float(),
            attention_mask=attn_mask,
        )
        return self.audio_proj(out.last_hidden_state)            # [B, T_a, d_model]

    def _attention_pool_audio(self, audio_feat: torch.Tensor) -> torch.Tensor:
        """Soft attention pooling over audio time axis → [B, d_model]."""
        w = torch.softmax(self.audio_attn_pool(audio_feat), dim=1)  # [B, T_a, 1]
        return (audio_feat * w).sum(dim=1)                           # [B, d_model]

    def _build_audio_key_pad_mask(
        self,
        attn_mask: torch.Tensor | None,
        T_a: int,
    ) -> torch.Tensor | None:
        """
        HuBERT downsamples audio ~50× (20ms stride).
        Approximates the feature-level padding mask from the input mask.
        True positions are ignored in MultiheadAttention key_padding_mask.
        """
        if attn_mask is None:
            return None
        B = attn_mask.shape[0]
        # Downsample mask to match audio feature length
        mask_ds = attn_mask.float().unsqueeze(1)             # [B, 1, T_in]
        mask_ds = torch.nn.functional.adaptive_avg_pool1d(mask_ds, T_a)  # [B, 1, T_a]
        mask_ds = mask_ds.squeeze(1)                          # [B, T_a]
        # True → ignore (padding positions where mean≈0)
        return mask_ds < 0.5

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        frames: torch.Tensor,            # [B, T, C, H, W]
        waveforms: torch.Tensor,         # [B, T_audio]
        attn_masks: torch.Tensor | None = None,  # [B, T_audio]
    ) -> torch.Tensor:                   # [B, num_classes]

        # 1. Backbone encoding
        v = self._encode_video(frames)                           # [B, T_v, D]
        a = self._encode_audio(waveforms, attn_masks)            # [B, T_a, D]

        # 2. Audio key-padding mask for cross-attention
        a_kpm = self._build_audio_key_pad_mask(attn_masks, T_a=a.shape[1])

        # 3. Cross-modal transformer (bidirectional fusion)
        v, a = self.cross_modal(v, a, a_key_pad_mask=a_kpm)     # same shapes

        # 4. Global audio context via attention pooling
        audio_global = self._attention_pool_audio(a)             # [B, D]

        # 5. BiLSTM over video temporal sequence
        lstm_out, _ = self.lstm(v)                               # [B, T_v, 2H]
        lstm_pooled = lstm_out.mean(dim=1)                       # [B, 2H]

        # 6. Fuse video + audio representations
        fused = torch.cat([lstm_pooled, audio_global], dim=-1)   # [B, 2H+D]
        fused = self.fusion_head(fused)                          # [B, D]

        # 7. Classify
        return self.classifier(fused)                            # [B, 2]
