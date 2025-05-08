from flask import Flask, request, jsonify
import os
from openai_extract_fields_combined import analyze_pdf

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze_route():
    try:
        file = request.files["file"]
        user_id = request.form.get("userId")
        report_name = request.form.get("reportName") or file.filename

        if not user_id:
            return jsonify({"error": "Missing userId"}), 400

        file_path = os.path.join("/tmp", file.filename)
        file.save(file_path)

        print(f"📥 Received file: {file.filename}")
        print(f"👤 User ID: {user_id} | 📄 Report Name: {report_name}")

        result = analyze_pdf(file_path, user_id, report_name)
        return jsonify(result), 200

    except Exception as e:
        print(f"❌ Exception in /analyze: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health():
    return "🟢 Analyzer service is running", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
