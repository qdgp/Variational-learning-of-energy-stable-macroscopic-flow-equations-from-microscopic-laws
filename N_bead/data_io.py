"""MAT output policy: store floating-point data in single precision."""
import numpy as np
from scipy.io import savemat as scipy_savemat


def as_single(value):
    """Convert saved floats recursively without changing working arrays."""
    if isinstance(value, dict):
        return {key: as_single(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        if value.dtype.kind == 'f':
            return value.astype(np.float32, copy=False)
        if value.dtype.kind == 'c':
            return value.astype(np.complex64, copy=False)
        if value.dtype.kind == 'O':
            result = np.empty(value.shape, dtype=object)
            for index in np.ndindex(value.shape):
                result[index] = as_single(value[index])
            return result
        return value
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except ValueError:
            array = None
        if array is not None and array.dtype.kind in 'fc':
            return as_single(array)
        return type(value)(as_single(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return np.float32(value)
    if isinstance(value, (complex, np.complexfloating)):
        return np.complex64(value)
    return value


def savemat(filename, data, **kwargs):
    """Write a MAT file with float32/complex64 arrays and integer indices."""
    scipy_savemat(filename, as_single(data), **kwargs)
