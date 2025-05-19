
from flask import Flask, request, jsonify
from openai_extract_fields_combined import analyze_pdf
import os

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])



def analyze():
    print("🟡 HIT /analyze endpoint", flush=True)

    if 'file' not in request.files:
        return jsonify({ "error": "No file provided." }), 400

    file = request.files['file']
    temp_path = os.path.join("/tmp", file.filename)
    file.save(temp_path)
    
    print(f"📬 Full form data: {request.form.to_dict()}", flush=True)

    user_id = request.form.get("userId")
    report_name = request.form.get("reportName")
    report_date = request.values.get("reportDate")

    print(f"📩 Received form keys: {list(request.form.keys())}", flush=True)
    print(f"🧾 user_id = {user_id}, report_name = {report_name}, report_date = {report_date}", flush=True)

    try:
        result = analyze_pdf(temp_path, user_id=user_id, report_name=report_name, report_date=report_date)
        return jsonify(result)
    except Exception as e:
        print(f"❌ Exception in /analyze: {str(e)}", flush=True)
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
