# BAD Molecule Filter

Predicts whether a small molecule forms colloidal aggregates in a biochemical assay.

**Use it here: https://abdallahhajal.github.io/BAD-Molecule-Filter/**

Draw a structure, paste SMILES, or upload a CSV. You get a probability per compound.

The page is on GitHub Pages and the scoring runs on a Hugging Face Space,
[Abdallahabouhajal/BAD-Molecule-Filter](https://huggingface.co/spaces/Abdallahabouhajal/BAD-Molecule-Filter),
which also has a plain page of its own. The Space sleeps when nobody is using it, so the
first run after a quiet spell takes a minute or two to wake it.

## Reading the output

| Probability | Call |
|---|---|
| > 0.6 | Aggregator |
| 0.3 to 0.6 | Ambiguous. Treat as untested rather than clean. |
| < 0.3 | Non-aggregator |

A negative call means only that the model saw nothing resembling an aggregator. It does not
mean the compound was tested and found inert. The training labels come from assay behaviour,
so a compound simply inactive against the screening enzyme is indistinguishable from one
measured and found not to form colloids.

## How it works

Each structure becomes 602 Morgan fingerprint bits (radius 2, 2048 bits) and 255 Mordred
descriptors. The descriptors are standardised with the scaler fitted on the 56,754 training
compounds. An extremely randomised trees model of 1000 trees returns the probability.

This is the merged model, the one the web tool has always used.

## Run it on your own machine

```
git clone https://github.com/AbdallahHajal/BAD-Molecule-Filter
cd BAD-Molecule-Filter
pip install -r requirements.txt
python server.py
```

Open http://localhost:7860. The 423 MB model is downloaded once on first use and cached next
to `server.py`. Set `BAD_MODEL_PATH` to keep it elsewhere.

Python 3.11 is needed, because mordred 1.2.0 does not install on 3.12 or later.

There is also a `Dockerfile`:

```
docker build -t bad-filter .
docker run -p 7860:7860 bad-filter
```

## Scoring from your own code

```python
import bad_filter
for r in bad_filter.predict(["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"]):
    print(r["smiles"], r["probability"], r["call"])
```

Or over HTTP:

```
curl -X POST http://localhost:7860/predict \
  -H 'Content-Type: application/json' \
  -d '{"smiles": ["CC(=O)Oc1ccccc1C(=O)O"]}'
```

## The model file

`MM_model.pkl`, released with the 2024 paper, was written by scikit-learn 1.1.3. Version 1.3
changed the tree node array, so the pickle no longer loads:

```
ValueError: node array from the pickle has an incompatible dtype
```

It has been converted to ONNX, which is not tied to a library version. The conversion was
checked against the original on 2,000 compounds: max |delta p| 7.3e-08, mean 1.3e-08, class
agreement 100.00% at the 0.6 and 0.3 thresholds. The whole pipeline was then checked end to
end on a separate set: max |delta p| 3.3e-07, 12 of 12 calls identical.

The ONNX file is 423 MB against 750 MB for the pickle, and needs onnxruntime rather than a
pinned scikit-learn. `convert_model_to_onnx.py` documents how it was made.

## Files

| File | What it is |
|---|---|
| `server.py` | Web server and JSON API |
| `bad_filter.py` | Feature pipeline and scoring |
| `docs/index.html` | The page, also published on GitHub Pages |
| `docs/jsme/` | JSME structure editor, BSD licence, vendored so the page has no external dependencies |
| `columns_MM.json` | The 857 feature columns in training order |
| `scaler_MM.npz` | The fitted scaler's mean and scale |
| `convert_model_to_onnx.py` | Pickle to ONNX conversion |
| `space/` | The Hugging Face Space app |
| `app.py` | The 2024 Flask app, kept for reference |

## Limits on the hosted service

The hosted service is open to anyone, and descriptor calculation is the expensive part, so
one request is capped at 500 structures, 600 characters per SMILES, 150 atoms per
structure, a 1 MB body, and 60 seconds of descriptor time. Batches are scored one at a
time. None of these apply when you run it yourself; `BAD_COST_BUDGET` and `BAD_MAX_BATCH`
change them.

## The old address

The paper gives the tool's address as `molmodlab-aau.com/Tools.html`. That server is no longer
running. The GitHub Pages link at the top is the tool's current home.

## Cite

Abou Hajal A, Bryce RA, Amor B, Atatreh N. Boosting the Accuracy and Chemical Space Coverage
of the Detection of Small Colloidal Aggregating Molecules. *J Chem Inf Model* 2024.
https://doi.org/10.1021/acs.jcim.4c00363

## Licence

GPL-3.0. The bundled JSME editor is BSD, Novartis Institutes for BioMedical Research and
Bruno Bienfait.

## Contact

Abdallah Abou Hajal, abdallah.abouhajal@gmail.com
