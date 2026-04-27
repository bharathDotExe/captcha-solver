"""
AI Captcha Solver for VTU Captchas
===================================
Template-based character recognition specifically tuned for VTU captcha patterns.

VTU captchas have:
- 6 bold, dark characters (the actual captcha text)
- Many lighter background words (NOTES, CAMPUS, UNIVERSITY, etc.)
- A refresh icon in the corner

Strategy:
1. Aggressive dark-pixel isolation to remove background noise
2. Contour-based character segmentation
3. Template matching against a learned library
4. ddddocr fallback for unmatched characters
"""

import os
import cv2
import json
import numpy as np
import ddddocr
from pathlib import Path
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "data" / "templates"
VTU_SAMPLES_DIR = BASE_DIR / "data" / "vtu captcha"
TEMPLATE_INDEX_FILE = TEMPLATE_DIR / "index.json"

CAPTCHA_LEN = 6

# ── Lazy-loaded globals ────────────────────────────────────────────
_ocr_std = None
_ocr_beta = None
_templates = None  # dict: char -> list of template images


def _get_ocr_std():
    global _ocr_std
    if _ocr_std is None:
        _ocr_std = ddddocr.DdddOcr(show_ad=False)
    return _ocr_std


def _get_ocr_beta():
    global _ocr_beta
    if _ocr_beta is None:
        _ocr_beta = ddddocr.DdddOcr(beta=True, show_ad=False)
    return _ocr_beta


# ═══════════════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

def preprocess_for_segmentation(img_data):
    """
    Preprocess captcha image to isolate only the dark bold captcha text.
    Returns a clean binary image (white text on black background).
    """
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None

    # Scale up 2x for better character segmentation
    h, w = img.shape[:2]
    img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # VTU captcha text has gray values ~85-120, background noise is >140
    # Threshold at 120 captures the bold captcha text while excluding most noise
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to clean up noise
    # Small opening to remove tiny noise dots
    kernel_small = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

    # Slight dilation to connect broken character strokes
    kernel_dilate = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel_dilate, iterations=1)

    return binary, img


def preprocess_variants_ai(img_data):
    """
    Generate multiple threshold variants for robust recognition.
    """
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    h, w = img.shape[:2]
    img_scaled = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)

    variants = []
    for thresh_val in [110, 120, 130, 140]:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.dilate(binary, kernel, iterations=1)
        variants.append((f"thresh-{thresh_val}", binary))

    return variants


# ═══════════════════════════════════════════════════════════════════
#  CHARACTER SEGMENTATION
# ═══════════════════════════════════════════════════════════════════

def segment_characters(binary_img, min_char_h=15, min_char_w=5):
    """
    Segment individual characters from binary image using contour detection.
    Returns list of (x, y, w, h, cropped_char) sorted left-to-right.
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = binary_img.shape[:2]

    char_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter out noise: too small, or too large (background word remnants)
        if h < min_char_h or w < min_char_w:
            continue
        if h > img_h * 0.95:  # spans full height = probably not a character
            continue
        if w > img_w * 0.4:  # way too wide for a single character
            continue
        # Filter out the refresh icon area (usually bottom-left or top-right corner)
        if y > img_h * 0.75 and x < img_w * 0.15:
            continue
        if y < img_h * 0.1 and x > img_w * 0.85:
            continue

        # Character aspect ratio sanity check
        aspect = w / h
        if aspect > 2.5:  # way too wide, likely merged noise
            continue

        char_boxes.append((x, y, w, h))

    # Sort by x-coordinate (left to right)
    char_boxes.sort(key=lambda b: b[0])

    # Merge overlapping boxes (characters that overlap in x)
    merged = []
    for box in char_boxes:
        if merged and box[0] < merged[-1][0] + merged[-1][2] * 0.5:
            # Overlapping — merge into the larger box
            prev = merged[-1]
            x_min = min(prev[0], box[0])
            y_min = min(prev[1], box[1])
            x_max = max(prev[0] + prev[2], box[0] + box[2])
            y_max = max(prev[1] + prev[3], box[1] + box[3])
            merged[-1] = (x_min, y_min, x_max - x_min, y_max - y_min)
        else:
            merged.append(box)

    # Extract character images with padding
    chars = []
    for (x, y, w, h) in merged:
        pad = 3
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)
        char_img = binary_img[y1:y2, x1:x2]
        chars.append((x, y, w, h, char_img))

    return chars


def select_best_6_chars(chars, img_w):
    """
    From segmented characters, pick the 6 that are most likely the captcha text.
    Captcha chars are the biggest/tallest and roughly centered.
    """
    if len(chars) <= CAPTCHA_LEN:
        return chars

    # Score each character: prefer taller, wider characters in the center region
    scored = []
    center_x = img_w / 2
    for (x, y, w, h, char_img) in chars:
        char_center = x + w / 2
        # Height is the strongest signal — captcha text is much taller
        size_score = h * 1.5 + w
        # Slight bonus for being near center
        dist_from_center = abs(char_center - center_x) / img_w
        center_score = 1.0 - dist_from_center * 0.3
        total = size_score * center_score
        scored.append((total, x, y, w, h, char_img))

    # Sort by score descending, take top 6
    scored.sort(key=lambda s: s[0], reverse=True)
    top6 = scored[:CAPTCHA_LEN]

    # Re-sort by x position (left to right)
    top6.sort(key=lambda s: s[1])

    return [(s[1], s[2], s[3], s[4], s[5]) for s in top6]


# ═══════════════════════════════════════════════════════════════════
#  TEMPLATE LIBRARY
# ═══════════════════════════════════════════════════════════════════

def normalize_char_img(char_img, target_size=(28, 28)):
    """Resize character image to standard size for template matching."""
    return cv2.resize(char_img, target_size, interpolation=cv2.INTER_AREA)


def build_template_library():
    """
    Build template library from VTU captcha samples.
    Uses ddddocr to label characters, then stores extracted character images.
    """
    global _templates
    
    print("[AI-Solver] Building template library from VTU captcha samples...")
    
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    
    ocr_std = _get_ocr_std()
    ocr_beta = _get_ocr_beta()
    
    templates = {}  # char -> list of normalized images
    
    if not VTU_SAMPLES_DIR.exists():
        print(f"[AI-Solver] Warning: VTU samples directory not found: {VTU_SAMPLES_DIR}")
        _templates = templates
        return templates
    
    sample_files = sorted(VTU_SAMPLES_DIR.glob("captcha_*.*"))
    print(f"[AI-Solver] Found {len(sample_files)} sample captcha files")
    
    for fpath in sample_files:
        try:
            img_data = fpath.read_bytes()
            binary, original = preprocess_for_segmentation(img_data)
            if binary is None:
                continue
            
            chars = segment_characters(binary)
            chars = select_best_6_chars(chars, binary.shape[1])
            
            if len(chars) != CAPTCHA_LEN:
                print(f"  [{fpath.name}] Segmented {len(chars)} chars (expected {CAPTCHA_LEN}), skipping")
                continue
            
            # Use ddddocr to identify each character
            # We'll run OCR on the full preprocessed image to get the label
            _, full_binary = cv2.threshold(
                cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY
            )
            _, buf = cv2.imencode('.png', full_binary)
            full_bytes = buf.tobytes()
            
            # Get labels from both OCR engines
            label_std = ocr_std.classification(full_bytes)
            label_beta = ocr_beta.classification(full_bytes)
            
            # Clean labels
            label_std = ''.join(c for c in label_std if c.isalnum())
            label_beta = ''.join(c for c in label_beta if c.isalnum())
            
            # Try different threshold variants
            best_label = None
            for label in [label_std, label_beta]:
                if len(label) == CAPTCHA_LEN:
                    best_label = label
                    break
            
            if not best_label:
                # Try with raw image
                raw_std = ocr_std.classification(img_data)
                raw_beta = ocr_beta.classification(img_data)
                raw_std = ''.join(c for c in raw_std if c.isalnum())
                raw_beta = ''.join(c for c in raw_beta if c.isalnum())
                for label in [raw_std, raw_beta]:
                    if len(label) == CAPTCHA_LEN:
                        best_label = label
                        break
            
            if not best_label:
                print(f"  [{fpath.name}] No 6-char OCR label found (std='{label_std}' beta='{label_beta}'), skipping")
                continue
            
            print(f"  [{fpath.name}] Label: '{best_label}' — storing {len(chars)} character templates")
            
            for i, (x, y, w, h, char_img) in enumerate(chars):
                char_label = best_label[i].upper()
                normalized = normalize_char_img(char_img)
                
                if char_label not in templates:
                    templates[char_label] = []
                templates[char_label].append(normalized)
                
                # Save template image to disk
                char_dir = TEMPLATE_DIR / char_label
                char_dir.mkdir(exist_ok=True)
                idx = len(list(char_dir.glob("*.png")))
                cv2.imwrite(str(char_dir / f"{idx:04d}.png"), normalized)
                
        except Exception as e:
            print(f"  [{fpath.name}] Error: {e}")
    
    # Save index
    index = {char: len(imgs) for char, imgs in templates.items()}
    with open(TEMPLATE_INDEX_FILE, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"[AI-Solver] Template library built: {len(templates)} unique characters, "
          f"{sum(len(v) for v in templates.values())} total templates")
    
    _templates = templates
    return templates


def load_template_library():
    """Load template library from disk."""
    global _templates
    
    if _templates is not None:
        return _templates
    
    if not TEMPLATE_INDEX_FILE.exists():
        return build_template_library()
    
    templates = {}
    try:
        with open(TEMPLATE_INDEX_FILE, 'r') as f:
            index = json.load(f)
        
        for char, count in index.items():
            char_dir = TEMPLATE_DIR / char
            if not char_dir.exists():
                continue
            templates[char] = []
            for img_path in sorted(char_dir.glob("*.png")):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[char].append(img)
        
        print(f"[AI-Solver] Loaded template library: {len(templates)} chars, "
              f"{sum(len(v) for v in templates.values())} templates")
        
    except Exception as e:
        print(f"[AI-Solver] Error loading templates: {e}")
        return build_template_library()
    
    _templates = templates
    return templates


# ═══════════════════════════════════════════════════════════════════
#  CHARACTER MATCHING
# ═══════════════════════════════════════════════════════════════════

def match_character(char_img, templates):
    """
    Match a character image against the template library.
    Returns (best_char, confidence) tuple.
    """
    normalized = normalize_char_img(char_img)
    
    best_char = '?'
    best_score = -1
    
    for char, template_list in templates.items():
        for template in template_list:
            # Normalized cross-correlation
            result = cv2.matchTemplate(
                normalized.astype(np.float32),
                template.astype(np.float32),
                cv2.TM_CCOEFF_NORMED
            )
            score = result[0][0] if result.size > 0 else 0
            
            if score > best_score:
                best_score = score
                best_char = char
    
    return best_char, best_score


def match_character_multi(char_img, templates):
    """
    Match using multiple similarity metrics and voting.
    """
    normalized = normalize_char_img(char_img)
    
    scores = {}  # char -> list of scores
    
    for char, template_list in templates.items():
        char_scores = []
        for template in template_list:
            # Method 1: Template matching (cross-correlation)
            result = cv2.matchTemplate(
                normalized.astype(np.float32),
                template.astype(np.float32),
                cv2.TM_CCOEFF_NORMED
            )
            tm_score = result[0][0] if result.size > 0 else 0
            
            # Method 2: Pixel difference (inverted — lower diff = higher similarity)
            diff = np.mean(np.abs(normalized.astype(float) - template.astype(float)))
            diff_score = 1.0 - (diff / 255.0)
            
            # Method 3: Structural similarity via histogram comparison
            hist_n = cv2.calcHist([normalized], [0], None, [16], [0, 256])
            hist_t = cv2.calcHist([template], [0], None, [16], [0, 256])
            cv2.normalize(hist_n, hist_n)
            cv2.normalize(hist_t, hist_t)
            hist_score = cv2.compareHist(hist_n, hist_t, cv2.HISTCMP_CORREL)
            
            combined = tm_score * 0.5 + diff_score * 0.3 + hist_score * 0.2
            char_scores.append(combined)
        
        if char_scores:
            scores[char] = max(char_scores)
    
    if not scores:
        return '?', 0.0
    
    best_char = max(scores, key=scores.get)
    return best_char, scores[best_char]


# ═══════════════════════════════════════════════════════════════════
#  MAIN SOLVER
# ═══════════════════════════════════════════════════════════════════

def solve_captcha_ai(img_data):
    """
    Main AI solver entry point.
    Takes raw image bytes, returns solved captcha string.
    """
    templates = load_template_library()
    
    results = []
    
    # Strategy 1: Template matching with multiple threshold variants
    if templates:
        variants = preprocess_variants_ai(img_data)
        
        for name, binary in variants:
            chars = segment_characters(binary)
            chars = select_best_6_chars(chars, binary.shape[1])
            
            if len(chars) != CAPTCHA_LEN:
                continue
            
            text = ''
            total_confidence = 0
            for (x, y, w, h, char_img) in chars:
                char, conf = match_character_multi(char_img, templates)
                text += char
                total_confidence += conf
            
            avg_conf = total_confidence / CAPTCHA_LEN
            results.append((name, text, avg_conf))
            print(f"  [AI-{name}] '{text}' (avg confidence: {avg_conf:.3f})")
    
    # Strategy 2: ddddocr fallback on preprocessed variants
    ocr_std = _get_ocr_std()
    ocr_beta = _get_ocr_beta()
    
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        for thresh_val in [110, 120, 135]:
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            
            # Scale up
            scaled = cv2.resize(thresh, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(scaled, cv2.MORPH_OPEN, kernel)
            
            _, buf = cv2.imencode('.png', cleaned)
            img_bytes = buf.tobytes()
            
            try:
                text_std = ocr_std.classification(img_bytes)
                text_std = ''.join(c for c in text_std if c.isalnum())
                if text_std:
                    results.append((f"ocr-std-{thresh_val}", text_std, 0.5))
                    print(f"  [OCR-std-{thresh_val}] '{text_std}'")
            except Exception:
                pass
            
            try:
                text_beta = ocr_beta.classification(img_bytes)
                text_beta = ''.join(c for c in text_beta if c.isalnum())
                if text_beta:
                    results.append((f"ocr-beta-{thresh_val}", text_beta, 0.45))
                    print(f"  [OCR-beta-{thresh_val}] '{text_beta}'")
            except Exception:
                pass
    
    # Also try raw image with ddddocr
    try:
        text_raw = ocr_std.classification(img_data)
        text_raw = ''.join(c for c in text_raw if c.isalnum())
        if text_raw:
            results.append(("ocr-raw-std", text_raw, 0.4))
    except Exception:
        pass
    
    try:
        text_raw_b = ocr_beta.classification(img_data)
        text_raw_b = ''.join(c for c in text_raw_b if c.isalnum())
        if text_raw_b:
            results.append(("ocr-raw-beta", text_raw_b, 0.35))
    except Exception:
        pass
    
    # ── Pick best result ──────────────────────────────────────────
    return pick_best_ai(results)


def pick_best_ai(results):
    """
    Pick the best captcha text from multiple strategy results.
    Prefers: exact 6-char → highest confidence → consensus → closest to 6.
    """
    if not results:
        return ""
    
    # Filter to 6-char results
    exact = [(name, text, conf) for name, text, conf in results if len(text) == CAPTCHA_LEN]
    
    if exact:
        # Check for consensus among 6-char results
        texts = [t for _, t, _ in exact]
        counts = Counter(texts)
        most_common_text, most_common_count = counts.most_common(1)[0]
        
        if most_common_count >= 2:
            # Multiple strategies agree
            print(f"  [AI-CHOSEN] '{most_common_text}' (consensus: {most_common_count} agree)")
            return most_common_text
        
        # Pick highest confidence
        exact.sort(key=lambda x: x[2], reverse=True)
        best = exact[0]
        print(f"  [AI-CHOSEN] '{best[1]}' (from {best[0]}, confidence: {best[2]:.3f})")
        return best[1]
    
    # No exact 6-char match — pick closest to 6
    results.sort(key=lambda x: (abs(len(x[1]) - CAPTCHA_LEN), -x[2]))
    best = results[0]
    text = best[1]
    if len(text) > CAPTCHA_LEN:
        text = text[:CAPTCHA_LEN]
    
    print(f"  [AI-CHOSEN-approx] '{text}' (from {best[0]})")
    return text


def rebuild_templates():
    """Force rebuild of template library."""
    global _templates
    _templates = None
    # Clear existing templates
    if TEMPLATE_DIR.exists():
        import shutil
        shutil.rmtree(TEMPLATE_DIR)
    return build_template_library()
