
from flask import Flask, request, jsonify
from openai_extract_fields_combined import analyze_pdf
import os

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'file' not in request.files:
        return jsonify({ "error": "No file provided." }), 400

    file = request.files['file']
    temp_path = os.path.join("/tmp", file.filename)
    file.save(temp_path)

    try:
        result = analyze_pdf(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
