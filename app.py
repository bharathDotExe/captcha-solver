import os
import sys
import base64
import tempfile
import cv2
import numpy as np
import ddddocr
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import Counter

app = Flask(__name__)
CORS(app)

# Initialize TWO ddddocr instances for different recognition modes
ocr_std = ddddocr.DdddOcr(show_ad=False)
ocr_beta = ddddocr.DdddOcr(beta=True, show_ad=False)

CAPTCHA_LEN = 6  # VTU captchas are always 6 characters

def preprocess_variants(img_data):
    """
    Generate multiple preprocessed versions of the captcha image
    to maximize the chance of a correct OCR read.
    
    VTU captchas have light gray background watermark text (NOTES, CAMPUS, etc.)
    and dark bold foreground captcha characters. We exploit that contrast difference.
    """
    # Decode raw bytes into cv2 image
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return [img_data]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants = []
    
    # === Strategy 1: Raw image (no processing) ===
    variants.append(("raw", img_data))
    
    # === Strategy 2: Aggressive dark-only threshold ===
    # Only keep very dark pixels (the bold captcha text), remove light watermarks
    _, thresh_dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # Invert so text is black on white (standard for OCR)
    _, buf = cv2.imencode('.png', thresh_dark)
    variants.append(("thresh-100", buf.tobytes()))
    
    # === Strategy 3: Medium threshold ===
    _, thresh_med = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    _, buf = cv2.imencode('.png', thresh_med)
    variants.append(("thresh-130", buf.tobytes()))
    
    # === Strategy 4: Otsu automatic threshold ===
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, buf = cv2.imencode('.png', thresh_otsu)
    variants.append(("otsu", buf.tobytes()))
    
    # === Strategy 5: Scale 2x + dark threshold (helps with small text) ===
    h, w = gray.shape
    scaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    _, thresh_scaled = cv2.threshold(scaled, 110, 255, cv2.THRESH_BINARY)
    # Clean up noise with morphological opening
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh_scaled, cv2.MORPH_OPEN, kernel)
    _, buf = cv2.imencode('.png', cleaned)
    variants.append(("scaled-thresh", buf.tobytes()))
    
    # === Strategy 6: Adaptive threshold ===
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 5)
    _, buf = cv2.imencode('.png', adaptive)
    variants.append(("adaptive", buf.tobytes()))
    
    return variants


def pick_best(candidates):
    """
    From a list of OCR results, pick the one most likely to be a valid
    6-character VTU captcha. Prefer:
      1. Exact 6-char alphanumeric results
      2. Most frequently occurring result (consensus)
      3. Closest to 6 chars
    """
    # Clean all candidates
    cleaned = []
    for name, text in candidates:
        t = ''.join(c for c in text if c.isalnum())
        if t:
            cleaned.append((name, t))
    
    if not cleaned:
        return ""
    
    # Filter to exactly 6-char results
    exact = [(n, t) for n, t in cleaned if len(t) == CAPTCHA_LEN]
    
    if exact:
        # If multiple 6-char results agree, use consensus
        texts = [t for _, t in exact]
        counts = Counter(texts)
        best = counts.most_common(1)[0][0]
        print(f"  [CHOSEN]  '{best}'  (from {texts})")
        return best
    
    # No exact 6-char match — pick closest to 6
    cleaned.sort(key=lambda x: abs(len(x[1]) - CAPTCHA_LEN))
    best = cleaned[0][1]
    # Truncate or return as-is
    if len(best) > CAPTCHA_LEN:
        best = best[:CAPTCHA_LEN]
    print(f"  [CHOSEN-approx]  '{best}'  (closest to {CAPTCHA_LEN} chars)")
    return best


@app.route('/solve', methods=['POST'])
def solve_captcha():
    data = request.json
    base64_img = data.get('image', '')
    
    if base64_img.startswith('data:image'):
        base64_img = base64_img.split(',')[1]

    try:
        img_data = base64.b64decode(base64_img)
    except Exception:
        return jsonify({"error": "Invalid base64 string", "text": ""}), 400

    try:
        variants = preprocess_variants(img_data)
        candidates = []
        
        for name, img_bytes in variants:
            try:
                text_std = ocr_std.classification(img_bytes)
                candidates.append((f"{name}-std", text_std))
                print(f"  [{name}-std]  '{text_std}'")
            except Exception as e:
                print(f"  [{name}-std]  ERROR: {e}")
            
            try:
                text_beta = ocr_beta.classification(img_bytes)
                candidates.append((f"{name}-beta", text_beta))
                print(f"  [{name}-beta] '{text_beta}'")
            except Exception as e:
                print(f"  [{name}-beta] ERROR: {e}")
        
        text = pick_best(candidates)
        print(f"[Backend] Final answer: '{text}'")
        
    except Exception as e:
        print("[Backend] Error:", e)
        text = ""

    return jsonify({"text": text})


# ═══════════════════════════════════════════════════════════════════
#  AI CAPTCHA SOLVER ENDPOINT (NEW — does NOT modify /solve above)
# ═══════════════════════════════════════════════════════════════════
from ai_solver import solve_captcha_ai, rebuild_templates

@app.route('/solve-ai', methods=['POST'])
def solve_captcha_ai_endpoint():
    """
    AI-powered captcha solver that uses template matching + pattern analysis
    from VTU captcha samples in data/vtu captcha/.
    """
    data = request.json
    base64_img = data.get('image', '')

    if base64_img.startswith('data:image'):
        base64_img = base64_img.split(',')[1]

    try:
        img_data = base64.b64decode(base64_img)
    except Exception:
        return jsonify({"error": "Invalid base64 string", "text": ""}), 400

    try:
        print("[AI-Solver] Processing captcha...")
        text = solve_captcha_ai(img_data)
        print(f"[AI-Solver] Final answer: '{text}'")
    except Exception as e:
        print(f"[AI-Solver] Error: {e}")
        text = ""

    return jsonify({"text": text})


@app.route('/rebuild-templates', methods=['POST'])
def rebuild_templates_endpoint():
    """Force rebuild of the AI template library from VTU captcha samples."""
    try:
        templates = rebuild_templates()
        count = sum(len(v) for v in templates.values())
        return jsonify({
            "status": "ok",
            "chars": len(templates),
            "templates": count
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
