"""Deterministic model-uplink codecs; downlinks remain in the model's dtype.

INT8 sends one array per state tensor followed by a float64 scale vector.
The model state is the trusted decoding schema; integer/bool buffers are exact.
Scales are serialized as Parameters, so payload accounting includes them.
"""

import numpy as np


UPLINK_CODECS = ("none", "fp16", "int8")


def validate_codec(codec: str) -> str:
    if codec not in UPLINK_CODECS:
        raise ValueError("communication.uplink_codec must be none, fp16 or int8")
    return codec


def _validate_array(array: np.ndarray) -> None:
    if not isinstance(array, np.ndarray) or array.dtype.kind not in "fiub":
        raise ValueError("Model tensors must be real numeric or boolean ndarrays")
    if not np.all(np.isfinite(array)):
        raise ValueError("Model tensors must contain only finite values")


def encode_uplink(weights: list[np.ndarray], codec: str = "none") -> list[np.ndarray]:
    """Encode full model state without modifying inputs or integer buffers."""
    validate_codec(codec)
    encoded = []
    scales = []
    for array in weights:
        _validate_array(array)
        if array.dtype.kind != "f" or codec == "none":
            encoded.append(array.copy())
        elif codec == "fp16":
            if array.size and np.max(np.abs(array)) > np.finfo(np.float16).max:
                raise ValueError("Model tensor exceeds finite FP16 range")
            encoded.append(array.astype(np.float16))
        else:
            values = array.astype(np.float64)
            maximum = float(np.max(np.abs(values))) if values.size else 0.0
            scale = maximum / 127.0 if maximum else 1.0
            # Subnormal FP64 tensors may underflow their nominal scale.
            scale = max(scale, np.nextafter(0.0, 1.0))
            encoded.append(np.asarray(np.clip(np.rint(values / scale), -127, 127), dtype=np.int8))
            scales.append(scale)
    if codec == "int8":
        encoded.append(np.asarray(scales, dtype=np.float64))
    return encoded


def decode_uplink(encoded: list[np.ndarray], template: list[np.ndarray],
                  codec: str = "none") -> list[np.ndarray]:
    """Validate against model shape/dtype and restore before aggregation."""
    validate_codec(codec)
    expected_count = len(template) + (codec == "int8")
    if len(encoded) != expected_count:
        raise ValueError("Incorrect encoded model tensor count")
    scales = None
    if codec == "int8":
        scales = encoded[-1]
        _validate_array(scales)
        count = sum(array.dtype.kind == "f" for array in template)
        if scales.dtype != np.float64 or scales.shape != (count,) or np.any(scales <= 0):
            raise ValueError("Invalid INT8 scale vector")
    decoded = []
    scale_index = 0
    for array, reference in zip(encoded, template):
        _validate_array(array)
        _validate_array(reference)
        compressed = reference.dtype.kind == "f" and codec != "none"
        expected_dtype = (np.dtype("float16" if codec == "fp16" else "int8")
                          if compressed else reference.dtype)
        if array.shape != reference.shape or array.dtype != expected_dtype:
            raise ValueError("Encoded tensor shape/dtype does not match model schema")
        if compressed and codec == "int8":
            if np.any(array == -128):
                raise ValueError("Symmetric INT8 tensor must be in [-127, 127]")
            with np.errstate(over="ignore", invalid="ignore"):
                restored = np.asarray(array.astype(np.float64) * scales[scale_index], dtype=reference.dtype)
            scale_index += 1
        else:
            restored = array.astype(reference.dtype, copy=True)
        _validate_array(restored)
        decoded.append(restored)
    return decoded
