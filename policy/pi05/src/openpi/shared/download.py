from collections.abc import Sequence
import logging
import os
import pathlib
import time
from typing import Any, TYPE_CHECKING, TypeAlias

import flax
import flax.traverse_util
import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

if TYPE_CHECKING:
    from openpi import transforms as _transforms
else:
    # Delay import to avoid circular dependency
    _transforms = None

BasePolicy: TypeAlias = _base_policy.BasePolicy


def maybe_download(path: str, gs: dict[str, Any] | None = None) -> pathlib.Path:
    """Download a file or directory if it's a remote path, otherwise return the local path.
    
    Args:
        path: Local file path or remote path (e.g., gs://bucket/path)
        gs: Optional dict with Google Storage credentials (e.g., {"token": "anon"})
    
    Returns:
        Path to the local file or directory
    """
    path_str = str(path)
    
    # If it's a local path, check if it exists
    if not path_str.startswith("gs://"):
        local_path = pathlib.Path(path_str)
        if not local_path.exists():
            raise FileNotFoundError(f"Local path does not exist: {local_path}")
        return local_path
    
    # Remote path - download it
    data_home = os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.openpi_data"))
    cache_dir = pathlib.Path(data_home)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a cache key from the remote path
    cache_key = path_str.replace("gs://", "").replace("/", "_")
    local_path = cache_dir / cache_key
    
    # If already cached, return it
    if local_path.exists():
        return local_path
    
    # Download the file
    fs_kwargs = {}
    if gs:
        fs_kwargs.update(gs)
    
    try:
        with fsspec.open(path_str, **fs_kwargs) as remote_file:
            if local_path.is_dir() or path_str.endswith("/"):
                # Directory download
                local_path.mkdir(parents=True, exist_ok=True)
                # For directories, we'd need to list and download recursively
                # For now, treat as single file
                pass
            else:
                # File download
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as local_file:
                    local_file.write(remote_file.read())
    except Exception as e:
        raise FileNotFoundError(f"Failed to download {path_str}: {e}") from e
    
    return local_path


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence["_transforms.DataTransformFn"] = (),
        output_transforms: Sequence["_transforms.DataTransformFn"] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        # Import transforms at runtime to avoid circular import
        if _transforms is None:
            from openpi import transforms as _transforms
        
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Extract optional non-tensor metadata before transformations / tensor conversion.
        # We use this for per-request features like activation steering.
        steer_cfg = None
        if isinstance(obs, dict) and "steer" in obs:
            steer_cfg = obs.get("steer")

        # Make a copy since transformations may modify the inputs in place.
        # IMPORTANT: Remove non-array keys (e.g. nested dicts) before the tensor conversion below.
        inputs = jax.tree.map(lambda x: x, obs)
        if isinstance(inputs, dict) and "steer" in inputs:
            inputs = dict(inputs)
            inputs.pop("steer", None)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise
        if self._is_pytorch_model and steer_cfg is not None:
            # Per-request activation steering config (handled by the PyTorch model if supported).
            sample_kwargs["steering"] = steer_cfg

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
