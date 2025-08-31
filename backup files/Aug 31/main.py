from flask import Flask, request, jsonify
from openai_extract_fields_combined import analyze_file
import os

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    print("🟡 HIT /analyze endpoint", flush=True)

    # 👇 Parse form data
    form_data = request.form.to_dict()
    print(f"📬 Full request.form: {form_data}", flush=True)

    user_id = form_data.get("userId")
    report_name = form_data.get("reportName")
    report_date = form_data.get("reportDate")

    if 'file' not in request.files:
        return jsonify({ "error": "No file provided." }), 400

    file = request.files['file']
    temp_path = os.path.join("/tmp", file.filename)
    file.save(temp_path)

    print(f"🧾 user_id = {user_id}, report_name = {report_name}, report_date = {report_date}", flush=True)

    try:
        # Auto-detect PDF vs image
        result = analyze_file(temp_path, user_id, report_name, report_date)
        return jsonify(result)
    except Exception as e:
        print(f"❌ Exception in /analyze: {e}", flush=True)
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

