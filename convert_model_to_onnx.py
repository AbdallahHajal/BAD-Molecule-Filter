#!/usr/bin/env python
"""Convert MM_model.pkl to ONNX, so the filter stops depending on scikit-learn 1.1.3.

Why this exists. MM_model.pkl was written by scikit-learn 1.1.3. In 1.3 the tree node
array gained a `missing_go_to_left` field, so the pickle no longer loads on any current
scikit-learn:

    ValueError: node array from the pickle has an incompatible dtype

That makes the published model unusable to anyone installing the tool today, independent
of any hosting question. ONNX stores the tree ensemble in a format that is not tied to a
library version, so the model keeps working as scikit-learn moves on.

It is also smaller and much lighter to load. The pickle carries four float64 arrays per
tree, of which three (impurity, n_node_samples, weighted_n_node_samples) are training
diagnostics that prediction never reads. ONNX keeps only what inference needs.

    pickle   750.7 MB on disk, 1.65 GB peak RSS to load
    onnx     423.1 MB on disk, loads in about 6 s

Fidelity was checked on 2,000 compounds drawn from the training matrices:

    max |delta p|     7.3e-08      (float32 rounding)
    mean |delta p|    1.3e-08
    class agreement   100.00%      at the published 0.6 / 0.3 thresholds

Running this needs an environment old enough to open the pickle. scikit-learn 1.1.3 needs
Python <= 3.11 and numpy < 2:

    uv venv --python 3.11 bad311
    uv pip install --python bad311/bin/python "numpy<2" "scikit-learn==1.1.3" \
        "scipy<1.12" skl2onnx onnx onnxruntime
    bad311/bin/python convert_model_to_onnx.py MM_model.pkl BAD_MM_model.onnx

The conversion is a one-off. Once the .onnx exists, nothing downstream needs scikit-learn
at all; onnxruntime alone will score compounds.
"""
import os
import pickle
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def main(src, dst):
    from skl2onnx import to_onnx

    # pickle.load is unsafe on untrusted input. The only file this is ever pointed at is
    # the author's own MM_model.pkl, published with the 2024 paper and shipped from this
    # repository's own releases. Do not run it against a model file from anywhere else.
    t0 = time.time()
    with open(src, "rb") as f:
        model = pickle.load(f)
    print(f"loaded {type(model).__name__}: {model.n_estimators} trees, "
          f"{model.n_features_in_} features  ({time.time() - t0:.0f}s)")

    # float32 input: the thresholds carry far more precision than the descriptors justify
    sample = np.zeros((1, model.n_features_in_), dtype=np.float32)

    t0 = time.time()
    onx = to_onnx(model, sample, target_opset=15, options={id(model): {"zipmap": False}})
    print(f"converted in {time.time() - t0:.0f}s")

    with open(dst, "wb") as f:
        f.write(onx.SerializeToString())

    before, after = os.path.getsize(src), os.path.getsize(dst)
    print(f"{src}  {before / 1e6:7.1f} MB")
    print(f"{dst}  {after / 1e6:7.1f} MB   ({before / after:.1f}x smaller)")
    print("\nVerify against the original before replacing anything in production.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: convert_model_to_onnx.py <MM_model.pkl> <out.onnx>")
    main(sys.argv[1], sys.argv[2])
