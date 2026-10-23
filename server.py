"""Web server for the BAD Molecule Filter.

Run it locally:

    pip install -r requirements.txt
    python server.py

then open http://localhost:7860. The 423 MB model is downloaded once on first use and
cached next to this file; set BAD_MODEL_PATH to keep it somewhere else.

This replaces the Flask app published in 2024. The differences are listed in
bad_filter.py. The predictions are the same.
"""
import os
import sys
import traceback

from flask import Flask, jsonify, request, send_from_directory

import bad_filter

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
MAX_BATCH = int(os.environ.get("BAD_MAX_BATCH", "500"))

app = Flask(__name__, static_folder=None)


@app.after_request
def allow_cross_origin(resp):
    """The page is also served from GitHub Pages, which is a different origin.

    Only SMILES are posted here and only predictions come back, so any page may call it.
    """
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/")
def index():
    return send_from_directory(DOCS, "index.html")


@app.route("/<path:path>")
def asset(path):
    return send_from_directory(DOCS, path)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": bad_filter.model_is_loaded(),
        "max_batch": MAX_BATCH,
        "thresholds": {"aggregator_above": bad_filter.AGGREGATOR_ABOVE,
                       "non_aggregator_below": bad_filter.NON_AGGREGATOR_BELOW},
    })


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    smiles = data.get("smiles")
    if isinstance(smiles, str):
        smiles = [smiles]
    if not isinstance(smiles, list) or not smiles:
        return jsonify({"error": "Send {\"smiles\": [\"...\"]} as JSON."}), 400
    smiles = [str(s).strip() for s in smiles if str(s).strip()]
    if not smiles:
        return jsonify({"error": "No SMILES in the request."}), 400
    if len(smiles) > MAX_BATCH:
        return jsonify({"error": "That is %d structures. This service scores up to %d "
                                 "at a time." % (len(smiles), MAX_BATCH)}), 400

    try:
        results = bad_filter.predict(smiles, log=lambda m: print(m, flush=True))
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Scoring failed. See the server log."}), 500

    return jsonify({"results": results, "n": len(results)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    if "--preload" in sys.argv:
        bad_filter.get_session()
    print("BAD Molecule Filter on http://localhost:%d" % port, flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
