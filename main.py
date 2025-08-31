# main.py (drop-in)

import os
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

from openai_extract_fields_combined import analyze_file

app = Flask(__name__)

# ---- allowed types (liberal for images; PDF routed as-is) ----
ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
ALLOWED_DOC_EXTS   = {'.pdf'}

def _peek_is_pdf(path: str) -> bool:
    try:
        with open(path, 'rb') as f:
            head = f.read(5)
        return head.startswith(b'%PDF-')
    except Exception:
        return False

def _sniff_and_fix_extension(path: str) -> str:
    """
    If the file’s extension is unknown/odd, try to detect with Pillow
    and rename to a proper extension. Returns (possibly) new path.
    """
    try:
        # If it's actually a PDF, keep it as PDF
        if _peek_is_pdf(path):
            root, _ = os.path.splitext(path)
            new_path = root + '.pdf'
            if new_path != path:
                os.rename(path, new_path)
            return new_path

        # Otherwise try image formats
        with Image.open(path) as im:
            fmt = (im.format or '').lower()  # 'jpeg', 'png', 'tiff', 'bmp', ...
        if fmt in ('jpeg', 'png', 'tiff', 'bmp'):
            new_ext = '.jpg' if fmt == 'jpeg' else f'.{fmt}'
            root, _ = os.path.splitext(path)
            new_path = root + new_ext
            if new_path != path:
                os.rename(path, new_path)
            return new_path
    except UnidentifiedImageError:
        pass
    except Exception as e:
        print(f"⚠️  Sniff/rename error: {e}", flush=True)
    return path

@app.route("/analyze", methods=["POST"])
def analyze():
    print("🟡 HIT /analyze endpoint", flush=True)

    # Parse form data
    form_data = request.form.to_dict()
    print(f"📬 Full request.form: {form_data}", flush=True)

    user_id     = form_data.get("userId")
    report_name = form_data.get("reportName") or "Uploaded Report"
    report_date = form_data.get("reportDate")

    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400

    f = request.files['file']
    orig_name = f.filename or "upload"
    safe_name = secure_filename(orig_name) or "upload"
    ext = (os.path.splitext(safe_name)[1] or '').lower()

    # Save into a dedicated temp dir each request
    tmpdir = tempfile.mkdtemp(prefix="aether_")
    tmp_path = os.path.join(tmpdir, safe_name)
    f.save(tmp_path)

    print("📥 Saved upload:", {
        "path": tmp_path,
        "filename": orig_name,
        "safe_name": safe_name,
        "ext": ext,
        "mimetype": f.mimetype,
    }, flush=True)

    # If extension is unknown or missing, sniff and normalize
    if ext not in (ALLOWED_IMAGE_EXTS | ALLOWED_DOC_EXTS):
        tmp_path = _sniff_and_fix_extension(tmp_path)
        ext = (os.path.splitext(tmp_path)[1] or '').lower()

    print("🔎 Normalized file:", {"path": tmp_path, "ext": ext}, flush=True)

    try:
        # Hand off to your existing analyzer (auto-detects by extension)
        result = analyze_file(tmp_path, user_id, report_name, report_date)
        return jsonify(result)
    except Exception as e:
        print(f"❌ Exception in /analyze: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render listens on $PORT; default to 8080 for local
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
