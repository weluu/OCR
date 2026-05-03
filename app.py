from flask import Flask, request, jsonify, send_from_directory, send_file
from segmword import segmenter_mots
from segment_letters import segmenter_lettres
from predict import charger_modele, predire_lettre, WEIGHTS_PATH

import cv2
import os
import base64
import tempfile
import zipfile
import io

from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)   # test

OUTPUT_DIR        = 'mot_seg'
LETTRES_DIR       = 'lettres_decoupees'

# Chargement unique du modèle au démarrage du serveur
print("Chargement du modèle...")
MODEL = charger_modele(WEIGHTS_PATH)
print("Modèle prêt.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def image_to_b64(image_cv) -> str:
    """Convertit une image OpenCV en base64 PNG."""
    _, buffer = cv2.imencode('.png', image_cv)
    return base64.b64encode(buffer).decode('utf-8')


def traiter_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    mots_info     = segmenter_mots(image_path, output_dir=OUTPUT_DIR, debug=False)
    image_annotee = image.copy()
    mots_b64      = []

    for mot in mots_info:
        x, y, w, h = mot['bbox']
        mot_img    = image[y:y+h, x:x+w]

        mots_b64.append({
            'index'    : mot['index'],
            'bbox'     : [x, y, w, h],
            'image_b64': image_to_b64(mot_img),
            'filename' : f"mot_{mot['index']:03d}.png"
        })

        cv2.rectangle(image_annotee, (x, y), (x+w, y+h), (0, 200, 255), 2)
        cv2.putText(image_annotee, str(mot['index']), (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    return mots_b64, image_to_b64(image_annotee)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/segment', methods=['POST'])
def segment():
    """Étape 1 — Segmentation des mots."""
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image reçue'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Fichier vide'}), 400

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        mots, annotated_b64 = traiter_image(tmp_path)
        os.unlink(tmp_path)
        return jsonify({
            'success'         : True,
            'nb_mots'         : len(mots),
            'mots'            : mots,
            'annotated_image' : annotated_b64
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({'error': str(e)}), 500


@app.route('/segment-letters', methods=['POST'])
def segment_letters():
    """
    Étape 2 — Segmentation des lettres.
    Lit mot_seg/, découpe les lettres et retourne les images en base64.
    """
    if not os.path.exists(OUTPUT_DIR) or not os.listdir(OUTPUT_DIR):
        return jsonify({'error': 'Aucun mot segmenté. Lance /segment d\'abord.'}), 400

    try:
        resultats = segmenter_lettres(mots_dir=OUTPUT_DIR, lettres_dir=LETTRES_DIR)

        # Encode chaque lettre en base64 pour l'affichage
        mots_avec_lettres = []
        for mot_info in resultats:
            lettres_b64 = []
            for lettre in mot_info['lettres']:
                path = os.path.join(LETTRES_DIR, lettre['file'])
                img  = cv2.imread(path)
                if img is not None:
                    lettres_b64.append({
                        'file'     : lettre['file'],
                        'image_b64': image_to_b64(img)
                    })

            mots_avec_lettres.append({
                'mot_file'     : mot_info['mot_file'],
                'mot_index'    : mot_info['mot_index'],
                'nb_lettres'   : len(lettres_b64),
                'lettres'      : lettres_b64
            })

        total_lettres = sum(m['nb_lettres'] for m in mots_avec_lettres)

        return jsonify({
            'success'     : True,
            'nb_mots'     : len(mots_avec_lettres),
            'nb_lettres'  : total_lettres,
            'mots'        : mots_avec_lettres
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Étape 3 — Reconnaissance des lettres et reconstruction ASCII.
    Lit lettres_decoupees/ et retourne le texte reconnu.
    """
    if not os.path.exists(LETTRES_DIR) or not os.listdir(LETTRES_DIR):
        return jsonify({'error': 'Aucune lettre segmentée. Lance /segment-letters d\'abord.'}), 400

    try:
        fichiers = sorted([
            f for f in os.listdir(LETTRES_DIR) if f.endswith('.png')
        ])

        # Prédiction lettre par lettre + regroupement par mot
        mots = {}
        resultats_detail = []

        for fname in fichiers:
            parts   = fname.replace('.png', '').split('_')
            mot_idx = int(parts[1])
            l_idx   = int(parts[2][1:])

            path       = os.path.join(LETTRES_DIR, fname)
            char, conf = predire_lettre(MODEL, path)

            mots.setdefault(mot_idx, []).append((l_idx, char, conf))
            resultats_detail.append({
                'file': fname,
                'char': char,
                'conf': conf
            })

        # Reconstruction de la phrase
        mots_texte = []
        for mot_idx in sorted(mots):
            lettres = sorted(mots[mot_idx], key=lambda x: x[0])
            mots_texte.append("".join(char for _, char, _ in lettres))

        texte = " ".join(mots_texte)

        return jsonify({
            'success' : True,
            'texte'   : texte,
            'detail'  : resultats_detail
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download-zip')
def download_zip():
    """Télécharge tous les mots segmentés en ZIP."""
    if not os.path.exists(OUTPUT_DIR):
        return jsonify({'error': 'Aucun mot segmenté'}), 404

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(OUTPUT_DIR)):
            if fname.endswith('.png'):
                zf.write(os.path.join(OUTPUT_DIR, fname), fname)
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip',
                     as_attachment=True, download_name='mots_segmentes.zip')


if __name__ == '__main__':
    print("Serveur démarré sur http://localhost:5000")
    app.run(debug=True, port=5000)