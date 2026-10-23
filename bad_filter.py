"""Feature pipeline and scoring for the BAD Molecule Filter.

This reproduces the pipeline published with the 2024 paper, with two changes that do not
affect the numbers:

1. The model is read from ONNX instead of MM_model.pkl. The pickle was written by
   scikit-learn 1.1.3 and no longer loads on 1.3 or later, so the published file is
   unusable on a current install. The ONNX conversion was checked on 2,000 compounds:
   max |delta p| 7.3e-08, class agreement 100.00%. See convert_model_to_onnx.py.
2. The fitted StandardScaler is read from scaler_MM.npz (its mean_ and scale_ arrays)
   instead of scaler_MM.pkl, so scikit-learn is not needed at run time either.

Feature order is the one the model was trained on, taken from columns_MM.json:
255 Mordred descriptors, standardised, followed by 602 Morgan bits, unscaled.
"""
import json
import os
import threading
import time
import urllib.request

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from mordred import Calculator, descriptors

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_URL = ("https://github.com/AbdallahHajal/BAD-Molecule-Filter/releases/"
             "download/model-onnx-v1/BAD_MM_model.onnx")
MODEL_PATH = os.environ.get("BAD_MODEL_PATH", os.path.join(HERE, "BAD_MM_model.onnx"))

# Published decision thresholds, as in the 2024 paper and the original app.
AGGREGATOR_ABOVE = 0.6
NON_AGGREGATOR_BELOW = 0.3

# Descriptor time is the expensive part, so a cap on the number of structures does not
# cap the work a request can ask for. Measured on one machine with the 255 descriptor
# calculator below:
#
#     aspirin             13 atoms      0.06 s
#     clotrimazole        25 atoms      0.11 s
#     nonacene            38 atoms      0.09 s
#     C50 chain           50 atoms      1.38 s
#     adamantane x10     100 atoms      1.64 s
#     20 biphenyls       120 atoms      1.62 s
#     C150 chain         150 atoms      0.40 s
#
# Cost tracks atom count only loosely, so the estimate below is not a prediction. Its job
# is to refuse obviously enormous requests before any work starts. The real bound is the
# deadline in featurize_chunked, which is tight because the worst single molecule inside
# MAX_ATOMS measured 1.7 s.
#
# (Before the calculator was restricted to the 255 descriptors actually used, the same
# C150 chain took 15.1 s and 500 of them would have occupied the server for two hours.)
MAX_SMILES_CHARS = 600
MAX_ATOMS = 150
COST_BUDGET_SECONDS = float(os.environ.get("BAD_COST_BUDGET", "60"))


def atom_count(mol):
    """The graph size Mordred actually works on.

    Not GetNumHeavyAtoms: hydrogens written explicitly in the SMILES, including isotopic
    ones such as [2H], are real nodes in the graph and cost real time, but do not count as
    heavy atoms. Counting only heavy atoms would let a caller past the limits with a
    molecule that is expensive to describe.
    """
    return mol.GetNumAtoms()


def estimate_seconds(n_atoms):
    """A deliberately generous upper bound on one molecule's descriptor time.

    Generous on purpose: it is only used to refuse a request outright, and refusing a
    legitimate batch is worse than letting the deadline stop a slow one. 500 drug-like
    molecules estimate at about 39 seconds and actually take about 22.
    """
    return 0.05 + 2.0 * (n_atoms / 150.0) ** 2

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

_cols = json.load(open(os.path.join(HERE, "columns_MM.json")))
CONTINUOUS_COLS = _cols["continuous"]   # 255, in training order
BINARY_COLS = _cols["binary"]           # 602, in training order

_sc = np.load(os.path.join(HERE, "scaler_MM.npz"))
SCALER_MEAN = _sc["mean_"].astype(np.float64)
SCALER_SCALE = _sc["scale_"].astype(np.float64)

_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)

# The published app built the calculator from every Mordred descriptor, about 1,600 of
# them, and then threw away all but these 255. Computing only the 255 gives the same
# numbers for a fraction of the work, and it also means no descriptor outside this list,
# including the expensive ones such as the detour matrix, can ever be reached by input.
_wanted = set(CONTINUOUS_COLS)
_selected = [d for d in Calculator(descriptors, ignore_3D=True).descriptors
             if str(d) in _wanted]
_missing = _wanted - {str(d) for d in _selected}
if _missing:
    raise RuntimeError("Mordred does not provide %d of the required descriptors: %s"
                       % (len(_missing), sorted(_missing)[:5]))
_calc = Calculator(_selected)

_session = None
_session_lock = threading.Lock()


class TooMuchWork(Exception):
    """The request would cost more descriptor time than the budget allows."""


def call_for(p):
    if p > AGGREGATOR_ABOVE:
        return "Aggregator"
    if p < NON_AGGREGATOR_BELOW:
        return "Non-aggregator"
    return "Ambiguous"


def download_model(path=MODEL_PATH, url=MODEL_URL, log=print):
    """Fetch the ONNX model once. It is 423 MB, so this takes a while on first run."""
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    tmp = path + ".part"
    log("downloading the model from %s" % url)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and done % (50 << 20) < (1 << 20):
                log("  %d / %d MB" % (done >> 20, total >> 20))
    os.replace(tmp, path)
    log("model saved to %s (%.1f MB)" % (path, os.path.getsize(path) / 1e6))
    return path


def get_session(log=print):
    """Load the ONNX model once per process."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                import onnxruntime as ort
                download_model(log=log)
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                log("loading the model")
                _session = ort.InferenceSession(MODEL_PATH, opts,
                                                providers=["CPUExecutionProvider"])
                log("model ready")
    return _session


def model_is_loaded():
    return _session is not None


def featurize(mols):
    """Build the 857-column matrix for a list of valid RDKit molecules.

    Returns (X, bad_rows) where bad_rows holds the indices whose Mordred descriptors
    could not all be computed. Those rows are reported as failures rather than imputed,
    because a substituted value would produce a prediction that looks real.
    """
    bits = np.array([_morgan.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8)
    morgan = pd.DataFrame(bits, columns=["Col_%d" % i for i in range(MORGAN_BITS)])

    mordred = _calc.pandas(mols, nproc=1, quiet=True)
    mordred = mordred.apply(pd.to_numeric, errors="coerce")
    mordred.index = morgan.index

    cont = mordred.reindex(columns=CONTINUOUS_COLS).to_numpy(dtype=np.float64)
    binary = morgan[BINARY_COLS].to_numpy(dtype=np.float64)

    bad_rows = np.where(~np.isfinite(cont).all(axis=1))[0]
    cont_scaled = (cont - SCALER_MEAN) / SCALER_SCALE
    return np.hstack([cont_scaled, binary]).astype(np.float32), set(bad_rows.tolist())


CHUNK_SECONDS = 5.0


def featurize_chunked(mols, deadline=None):
    """featurize in pieces, checking the clock between them.

    The cost estimate decides whether a request is accepted, but an estimate can be wrong.
    This is the backstop: work is cut into pieces of about CHUNK_SECONDS and the deadline
    is checked after each, so a request that turns out to be far more expensive than it
    looked stops instead of running for as long as it likes. The overshoot is bounded by
    one molecule's cost, which the MAX_ATOMS cap holds to about 15 seconds.
    """
    parts, bad = [], set()
    i = 0
    while i < len(mols):
        j, cost = i, 0.0
        while j < len(mols):
            c = estimate_seconds(atom_count(mols[j]))
            if j > i and cost + c > CHUNK_SECONDS:
                break
            cost += c
            j += 1
        X, chunk_bad = featurize(mols[i:j])
        parts.append(X)
        bad |= {i + k for k in chunk_bad}
        i = j
        if deadline is not None and i < len(mols) and time.monotonic() > deadline:
            raise TooMuchWork(
                "That request is taking longer than its size suggested and was stopped "
                "after %d seconds. Send fewer structures, or run it on your own machine."
                % round(COST_BUDGET_SECONDS))
    return np.vstack(parts), bad


def predict(smiles_list, log=print):
    """Score a list of SMILES. Returns one dict per input, in the order given."""
    results = [{"smiles": s, "probability": None, "call": None, "error": None}
               for s in smiles_list]

    mols, rows = [], []
    for i, s in enumerate(smiles_list):
        if s and len(s) > MAX_SMILES_CHARS:
            results[i]["error"] = "Too long (limit %d characters)" % MAX_SMILES_CHARS
            continue
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            results[i]["error"] = "Not a readable SMILES"
        elif atom_count(m) > MAX_ATOMS:
            results[i]["error"] = ("Too large (%d atoms, limit %d)"
                                   % (atom_count(m), MAX_ATOMS))
        else:
            mols.append(m)
            rows.append(i)

    if not mols:
        return results

    cost = sum(estimate_seconds(atom_count(m)) for m in mols)
    if cost > COST_BUDGET_SECONDS:
        raise TooMuchWork(
            "That request needs about %d seconds of descriptor calculation, and the limit "
            "is %d. Send fewer structures, or smaller ones, or run it on your own machine."
            % (round(cost), round(COST_BUDGET_SECONDS)))

    sess = get_session(log=log)
    X, bad_rows = featurize_chunked(mols, deadline=time.monotonic() + COST_BUDGET_SECONDS)

    keep = [k for k in range(len(mols)) if k not in bad_rows]
    for k in bad_rows:
        results[rows[k]]["error"] = "Descriptors could not be computed"
    if not keep:
        return results

    probs = sess.run(None, {sess.get_inputs()[0].name: X[keep]})[1][:, 1]
    for k, p in zip(keep, probs):
        r = results[rows[k]]
        r["probability"] = float(p)
        r["call"] = call_for(float(p))
    return results
