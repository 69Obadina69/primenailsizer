import json
import os
from flask import Blueprint, jsonify

reference_bp = Blueprint("reference", __name__)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "references.json")


@reference_bp.route("/api/v1/references", methods=["GET"])
def list_references():
    with open(DATA_PATH) as f:
        data = json.load(f)
    return jsonify(data)
