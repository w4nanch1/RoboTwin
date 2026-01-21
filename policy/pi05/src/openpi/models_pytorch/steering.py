"""
Activation steering (占位框架).

这里提供一个可插拔的 steering function：
- 输入：某一层的 activation (torch.Tensor)
- 输出：被修改后的 activation (torch.Tensor)

当前版本先做 identity（不修改），后续你可以在这里实现具体的 steering 逻辑。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pathlib
from typing import Any, Final

import torch


@dataclass(frozen=True)
class SteeringConfig:
    """Per-request steering configuration.

    - enabled: 是否启用 steering
    - layer: 触发的 transformer layer index（0-based）
    - type: steering 类型（用于选择不同的 steering 实现）
    - params: steering 超参数（不同类型可以定义不同字段，统一用 dict 承载）
    """

    enabled: bool = False
    layer: int = -1
    type: str = "collect_grad"
    params: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Any) -> "SteeringConfig":
        if not isinstance(d, dict):
            return SteeringConfig()
        params = d.get("params", {})
        if not isinstance(params, dict):
            params = {}
        return SteeringConfig(
            enabled=bool(d.get("enabled", False)),
            layer=int(d.get("layer", -1)),
            type=str(d.get("type", "collect_grad")),
            params=params,
        )


class BaseSteering(ABC):
    """Steering 基类：不同 steering 算法继承它并实现 `steer_activation`。"""

    @abstractmethod
    def steer_activation(
        self,
        activation: torch.Tensor,
        *,
        cfg: SteeringConfig,
        layer_idx: int,
        sample_time: float | None = None,
    ) -> torch.Tensor:  # noqa: D401
        """输入该层的 activation，返回修改后的 activation。"""

    def on_activation_grad(
        self, activation: torch.Tensor, grad: torch.Tensor, *, cfg: SteeringConfig, layer_idx: int
    ) -> None:
        """当 activation 的梯度可用（backward 时）会回调到这里。

        默认什么都不做。需要梯度的 steering 算法可以 override 这个方法。
        """


class CollectGradSteering(BaseSteering):
    """只收集梯度：不修改 activation。"""

    def steer_activation(
        self,
        activation: torch.Tensor,
        *,
        cfg: SteeringConfig,
        layer_idx: int,
        sample_time: float | None = None,
    ) -> torch.Tensor:
        _ = (cfg, layer_idx, sample_time)
        return activation

    def on_activation_grad(
        self, activation: torch.Tensor, grad: torch.Tensor, *, cfg: SteeringConfig, layer_idx: int
    ) -> None:
        # NOTE: do not keep references to graph tensors.
        grad_cpu = grad.detach().to("cpu")

        out_dir = cfg.params.get("out_dir")
        sample_id = cfg.params.get("sample_id")
        ep_id = cfg.params.get("ep_id", -1)
        sample_t = cfg.params.get("sample_t", None)

        if out_dir is None or sample_id is None:
            return

        out_dir = pathlib.Path(str(out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        # filename: layer{idx}_ep{ep_id}_t{sample_t}_sample{sample_id}.pt
        if sample_t is None:
            sample_t_str = "nan"
        else:
            try:
                sample_t_str = f"{float(sample_t):.6f}"
            except Exception:
                sample_t_str = str(sample_t)
        out_path = out_dir / f"layer{int(layer_idx)}_ep{int(ep_id)}_t{sample_t_str}_sample{int(sample_id)}.pt"
        payload = {
            "layer_idx": int(layer_idx),
            "sample_id": int(sample_id),
            "ep_id": int(ep_id),
            "sample_t": float(sample_t) if sample_t is not None else None,
            "grad": grad_cpu,
            "activation_shape": tuple(activation.shape),
            "grad_shape": tuple(grad.shape),
            "dtype": str(grad_cpu.dtype),
        }
        torch.save(payload, out_path)


def _bin_time(sample_time: float, *, num_bins: int = 10) -> int:
    """Map a continuous time in [0,1] to one of `num_bins` discrete bins.

    We mirror the inference denoise loop which starts at t=1.0 and decreases by 1/num_bins each step:
    bin 0 ~ t in (0.9, 1.0], bin 1 ~ (0.8, 0.9], ..., bin 9 ~ (0.0, 0.1].
    """
    t = float(sample_time)
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    idx = int((1.0 - t) * num_bins)
    if idx >= num_bins:
        idx = num_bins - 1
    if idx < 0:
        idx = 0
    return idx


_MEAN_GRADS_CACHE: dict[str, dict[str, Any]] = {}


def _load_mean_grads(path: str) -> dict[str, Any]:
    """Load and cache mean gradients file."""
    if path in _MEAN_GRADS_CACHE:
        return _MEAN_GRADS_CACHE[path]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "mean_grads" not in payload:
        raise ValueError(f"Invalid mean grads file: {path}")
    _MEAN_GRADS_CACHE[path] = payload
    return payload


class GradientSteering(BaseSteering):
    """Use (negative) mean gradients as a steering direction to modify activations.

    Expected cfg.params:
      - mean_grads_path: str, path to a torch file produced by compute_mean_grads.py
      - scale: float, step size (activation <- activation - scale * grad_mean)
      - num_bins: int (optional, default 10)
    """

    def steer_activation(
        self,
        activation: torch.Tensor,
        *,
        cfg: SteeringConfig,
        layer_idx: int,
        sample_time: float | None = None,
    ) -> torch.Tensor:
        _ = layer_idx
        if sample_time is None:
            return activation

        mean_grads_path = cfg.params.get("mean_grads_path")
        if mean_grads_path is None:
            return activation

        scale = float(cfg.params.get("scale", 1.0))
        num_bins = int(cfg.params.get("num_bins", 10))
        bin_idx = _bin_time(sample_time, num_bins=num_bins)

        payload = _load_mean_grads(str(mean_grads_path))
        mean_grads = payload["mean_grads"]  # torch.Tensor [num_bins, ...]
        if not isinstance(mean_grads, torch.Tensor):
            raise TypeError("mean_grads must be a torch.Tensor")
        g = mean_grads[bin_idx].to(device=activation.device, dtype=activation.dtype)

        # Broadcast to activation shape
        while g.ndim < activation.ndim:
            g = g.unsqueeze(0)

        # Use negative gradient direction: activation <- activation - scale * grad_mean
        return activation - scale * g


_STEERING_REGISTRY: Final[dict[str, BaseSteering]] = {
    "collect_grad": CollectGradSteering(),
    "gradient": GradientSteering(),
}


def steer_activation(
    activation: torch.Tensor, *, cfg: SteeringConfig, layer_idx: int, sample_time: float | None = None
) -> torch.Tensor:
    """框架入口：根据 cfg.type 选择具体 steering 实现并执行。

    为了保持模型侧 hook 调用方式不变，我们保留同名函数作为 dispatch wrapper。
    """
    impl = _STEERING_REGISTRY.get(cfg.type, _STEERING_REGISTRY["collect_grad"])
    return impl.steer_activation(activation, cfg=cfg, layer_idx=layer_idx, sample_time=sample_time)


def on_activation_grad(activation: torch.Tensor, grad: torch.Tensor, *, cfg: SteeringConfig, layer_idx: int) -> None:
    """框架入口：当 backward 计算出 dL/d(activation) 时分发给具体 steering 实现。"""
    impl = _STEERING_REGISTRY.get(cfg.type, _STEERING_REGISTRY["collect_grad"])
    impl.on_activation_grad(activation, grad, cfg=cfg, layer_idx=layer_idx)


