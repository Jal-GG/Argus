"""OSNet Re-ID embedder via torchreid (optional dependency).

Requires ``pip install torchreid`` and network access on first run to fetch
pretrained weights. Raises ImportError at construction otherwise so callers
can fall back gracefully.
"""

from __future__ import annotations

import numpy as np

from ...utils.logger import get_logger
from .embedder import l2_normalize

logger = get_logger(__name__)


class OSNetEmbedder:
    """OSNet appearance-feature extractor (torchreid)."""

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        device: str | None = None,
        input_size: tuple[int, int] = (256, 128),
    ) -> None:
        import torch
        import torchvision.transforms as T

        try:
            import torchreid
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError("torchreid is required for OSNetEmbedder") from exc

        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = resolved
        self.input_size = input_size

        self.model = torchreid.models.build_model(
            name=model_name, num_classes=1000, pretrained=True, loss="softmax"
        )
        self.model.eval()
        self.model.to(resolved)
        self.feature_dim = int(getattr(self.model, "feature_dim", 512))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # BGR->RGB handled manually below; transforms assume RGB tensors.
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(input_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        import torch

        if len(crops) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        batch = torch.stack(
            [self.transform(crop[:, :, ::-1]) for crop in crops]  # BGR→RGB
        ).to(self.device)

        with torch.no_grad():
            out = self.model(batch)
            if out.dim() > 2:  # some heads return (N, D, 1, 1)
                out = out.view(out.size(0), -1)
        return l2_normalize(out.cpu().numpy().astype(np.float32))
