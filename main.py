# main.py (drop-in)

from flask import Flask, request, jsonify
from openai_extract_fields_combined import analyze_file
import os, tempfile
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}

def _normalize_saved_file(path, original_name, mimetype):
    """
    Return {'path': <normalized_path>, 'ext': '.pdf'|'.jpg'|...}
    - Detect %PDF- header -> rename to .pdf
    - Else try Pillow to identify image format and adjust extension
    - Fallback: if mimetype says image, default to .jpg
    """
    ext = os.path.splitext(original_name or '')[1].lower()

    # Quick PDF sniff by magic bytes
    try:
        with open(path, 'rb') as f:
            head = f.read(5)
        if head.startswith(b'%PDF-'):
            new_path = os.path.splitext(path)[0] + '.pdf'
            os.replace(path, new_path)
            return {'path': new_path, 'ext': '.pdf'}
    except Exception as e:
        print('⚠️ PDF sniff warning:', e, flush=True)

    # Try Pillow to identify an image
    try:
        with Image.open(path) as im:
            fmt = (im.format or '').upper()
        mapping = {'JPEG': '.jpg', 'PNG': '.png', 'TIFF': '.tiff', 'BMP': '.bmp'}
        ext2 = mapping.get(fmt)
        if ext2:
            if ext != ext2:
                new_path = os.path.splitext(path)[0] + ext2
                os.replace(path, new_path)
                return {'path': new_path, 'ext': ext2}
            return {'path': path, 'ext': ext}
    except UnidentifiedImageError:
        pass  # not an image (or Pillow can't identify)
    except Exception as e:
        print('⚠️ Pillow identify warning:', e, flush=True)

    # Fallback based on mimetype
    if (mimetype or '').startswith('image/'):
        if ext not in IMG_EXTS:
            new_path = os.path.splitext(path)[0] + '.jpg'
            os.replace(path, new_path)
            return {'path': new_path, 'ext': '.jpg'}
        return {'path': path, 'ext': ext}

    # Unknown; keep as-is
    return {'path': path, 'ext': ext or '.bin'}


@app.route("/analyze", methods=["POST"])
def analyze():
    print("🟡 HIT /analyze endpoint", flush=True)

    form = request.form.to_dict()
    print(f"📬 Full request.form: {form}", flush=True)

    user_id     = form.get("userId")
    report_name = form.get("reportName")
    report_date = form.get("reportDate")

    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400

    uploaded = request.files['file']
    safe_name = secure_filename(uploaded.filename or "upload")

    tmpdir = tempfile.mkdtemp(prefix="aether_")
    saved_path = os.path.join(tmpdir, safe_name)
    uploaded.save(saved_path)

    print("📥 Saved upload:", {
        'path': saved_path,
        'filename': uploaded.filename,
        'safe_name': safe_name,
        'ext': os.path.splitext(saved_path)[1].lower(),
        'mimetype': uploaded.mimetype
    }, flush=True)

    try:
        normalized = _normalize_saved_file(saved_path, safe_name, uploaded.mimetype)
        print("🔎 Normalized file:", normalized, flush=True)

        ext = normalized['ext'].lower()
        print(f"🔧 Route decision ext={ext} (image={ext in IMG_EXTS}, pdf={ext == '.pdf'})", flush=True)

        # 🚫 Do NOT reject here; pass through to analyzer which
        # already branches correctly by extension.
        result = analyze_file(normalized['path'], user_id, report_name, report_date)
        return jsonify(result)
    except Exception as e:
        print(f"❌ Exception in /analyze: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
