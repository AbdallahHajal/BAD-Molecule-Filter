---
title: BAD Molecule Filter
emoji: 🧪
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: '3.11'
app_file: app.py
pinned: false
license: gpl-3.0
short_description: Predicts colloidal aggregation of small molecules
---

# BAD Molecule Filter

Returns the probability that a compound forms colloidal aggregates in a biochemical assay.

| Probability | Call |
|---|---|
| > 0.6 | Aggregator |
| 0.3 to 0.6 | Ambiguous. Treat as untested rather than clean. |
| < 0.3 | Non-aggregator |

To draw structures instead of typing them, use
[the main page](https://abdallahhajal.github.io/BAD-Molecule-Filter/), which sends its
results here.

Source: https://github.com/AbdallahHajal/BAD-Molecule-Filter

Abou Hajal A, Bryce RA, Amor B, Atatreh N. Boosting the Accuracy and Chemical Space
Coverage of the Detection of Small Colloidal Aggregating Molecules.
*J Chem Inf Model* 2024. https://doi.org/10.1021/acs.jcim.4c00363
