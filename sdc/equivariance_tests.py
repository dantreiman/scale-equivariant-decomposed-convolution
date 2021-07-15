import numpy as np
import random


def test_equivariance(x_in, f, tx, ty, t_param_ranges=[(0,1)], samples=100, edge_crop_y=16):
    """Tests f:X->Y for equivariance with respect to transformation t by comparing
       f(tx(x)) with ty(f(x)) to test experimentally if t commutes with f.  Returns mean absolute error.
    
    Args:
      x_in (np.array) The input to f.
      f (function) The function to test.
      tx (function) The transformation acting on X.
      ty (function) The transformation acting on Y (may be the same as tx if X and Y are the same size).
      t_param_ranges (list) Transformation parameters will be randomly drawn from the uniform distributions listed here.
      samples (int) The number of random transformations to test.
      edge_crop_y (int) The number of samples to crop from y edges when computing mean error, or None.
    Returns: (float) The absolute error, ty(f(x)), f(tx(x))
    """
    x_samples = [x_in]
    t_params = [[random.uniform(a,b) for a,b in t_param_ranges] for _ in range(samples)]
    # Compute x batch by transforming x
    for p in t_params:
        x = tx(x_in, *p)
        x_samples.append(x)
    y_samples = f(np.stack(x_samples, axis=0))
    y_in = y_samples[0]
    y_expected_samples = [y_in]
    # Prepare expectations by applying transformation to y
    for p in t_params:
        y_expected = ty(y_in, *p)
        y_expected_samples.append(y_expected)
    y_expected_samples = np.stack(y_expected_samples, axis=0)
    if edge_crop_y is not None:
        y_samples_cropped = y_samples[:,edge_crop_y:-edge_crop_y]
        y_expected_samples_cropped = y_expected_samples[:,edge_crop_y:-edge_crop_y]
    else:
        y_samples_cropped = y_samples
        y_expected_samples_cropped = y_expected_samples
    abs_error = np.abs(y_samples_cropped - y_expected_samples_cropped).sum(axis=tuple(range(1, y_samples.ndim - 1)))
    return abs_error, y_samples, y_expected_samples


# -------------------- Common Transformations --------------------


def translate_1d(x, *d):
    """Translates x (with periodic boundary conditions)"""
    return np.roll(x, np.array(d, dtype=np.int32), axis=range(0, len(d)))


def scale_and_crop_1d(x, f):
    """Scale x by scale factor f.
       Returns an array of the same size as x, center-cropping or padding as needed.
    """
    new_length = int(len(x) * f)
    if new_length == len(x):
        return x
    resampled = scipy.signal.resample(x, new_length)
    if new_length > len(x):
        crop = new_length - len(x)
        return resampled[int(np.floor(crop / 2)) : -int(np.ceil(crop / 2))]  # crop
    else:  # new_length < len(x)
        padding = len(x) - new_length
        # Pads first dimension
        padding = [(int(np.floor(padding / 2.0)), int(np.ceil(padding / 2.0)))] + [(0,0) for _ in range(1,resampled.ndim)]
        padded = np.pad(resampled, padding)
        return padded

