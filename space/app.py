"""Hugging Face Space for the BAD Molecule Filter.

Docker Spaces need a paid plan, so this runs on the Gradio SDK. It does two jobs:

1. It gives the Space its own working page, for anyone who lands here directly.
2. It answers the predict endpoint that
   https://abdallahhajal.github.io/BAD-Molecule-Filter/ calls, so the structure editor
   on that page can score what you draw.

The model and the feature pipeline are in bad_filter.py, shared with server.py in the
same repository. Nothing about the predictions is different here.
"""
import threading

import gradio as gr

import bad_filter

MAX_BATCH = 500

# The model is 423 MB and is fetched from the GitHub release on first use. Start that in
# the background at import so the Space can bind its port straight away, rather than
# leaving the first visitor to wait for the download.
threading.Thread(target=bad_filter.get_session, daemon=True).start()

EXAMPLE = "\n".join([
    "CC(=O)Oc1ccccc1C(=O)O",
    "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1ccnc1",
    "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
])


def _as_list(smiles):
    if isinstance(smiles, str):
        smiles = smiles.splitlines()
    if not isinstance(smiles, (list, tuple)):
        return []
    return [str(s).strip() for s in smiles if str(s).strip()]


def api_predict(smiles):
    """Score SMILES. Accepts a list, or one string with one SMILES per line.

    Returns {"error": str or None, "results": [{smiles, probability, call, error}, ...]}.
    """
    items = _as_list(smiles)
    if not items:
        return {"error": "No SMILES given.", "results": []}
    if len(items) > MAX_BATCH:
        return {"error": "That is %d structures. This service scores up to %d at a time."
                         % (len(items), MAX_BATCH),
                "results": []}
    try:
        return {"error": None, "results": bad_filter.predict(items)}
    except bad_filter.TooMuchWork as e:
        return {"error": str(e), "results": []}


def ui_predict(text):
    answer = api_predict(text)
    if answer["error"]:
        return [], answer["error"]
    rows = []
    for r in answer["results"]:
        if r["probability"] is None:
            rows.append([r["smiles"], "", r["error"] or "not scored"])
        else:
            rows.append([r["smiles"], round(r["probability"], 3), r["call"]])
    scored = sum(1 for r in answer["results"] if r["probability"] is not None)
    return rows, "%d of %d scored." % (scored, len(rows))


with gr.Blocks(title="BAD Molecule Filter") as demo:
    gr.Markdown(
        "# BAD Molecule Filter\n"
        "Returns the probability that a compound forms colloidal aggregates in a "
        "biochemical assay. Above 0.6 is an aggregator, below 0.3 is a non-aggregator, "
        "and in between is ambiguous and should be treated as untested rather than clean."
        "\n\n"
        "To draw structures instead of typing them, use "
        "[the main page](https://abdallahhajal.github.io/BAD-Molecule-Filter/), which "
        "sends its results here.\n\n"
        "Abou Hajal A, Bryce RA, Amor B, Atatreh N. *J Chem Inf Model* 2024. "
        "[10.1021/acs.jcim.4c00363](https://doi.org/10.1021/acs.jcim.4c00363)"
    )
    with gr.Row():
        with gr.Column(scale=1):
            box = gr.Textbox(label="SMILES, one per line", lines=10, value=EXAMPLE)
            go = gr.Button("Predict", variant="primary")
        with gr.Column(scale=1):
            table = gr.Dataframe(headers=["SMILES", "Probability", "Call"],
                                 datatype=["str", "number", "str"],
                                 label="Results", wrap=True)
            note = gr.Markdown()
    go.click(ui_predict, inputs=box, outputs=[table, note], api_name="ui_predict")

    # The endpoint the main page calls. Hidden, because it is not part of this UI.
    hidden_in = gr.JSON(visible=False)
    hidden_out = gr.JSON(visible=False)
    hidden_go = gr.Button(visible=False)
    hidden_go.click(api_predict, inputs=hidden_in, outputs=hidden_out,
                    api_name="predict")

if __name__ == "__main__":
    demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=7860)
