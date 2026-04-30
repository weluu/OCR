from flask import Flask, request, jsonify, send_from_directory, send_file
#flask pour créer serveur, request pour recevoir les données envoyées par le navigateur, 
from segmword import segmenter_mots
import cv2
import os
import base64
import tempfile
import zipfile
import io

app = Flask(__name__, static_folder='.')


OUTPUT_DIR = 'mots_decoupes' # dossier où les mots seront sauvegardé 


def traiter_image(image_path):
    # transforme les images des mots découpés en texte pour pouvoir les envoyer au navigateur
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    # Appel au code de segmentation original (segmword.py)
    mots_info = segmenter_mots(image_path, output_dir=OUTPUT_DIR, debug=False)

    # Construire image annotée
    image_annotee = image.copy()
    mots_b64 = []

    for mot in mots_info:
        x, y, w, h = mot['bbox']
        mot_img = image[y:y+h, x:x+w]

        # Encoder chaque mot en base64
        _, buffer = cv2.imencode('.png', mot_img)
        b64 = base64.b64encode(buffer).decode('utf-8')

        mots_b64.append({
            'index': mot['index'],
            'bbox': [x, y, w, h],
            'image_b64': b64,
            'filename': f"mot_{mot['index']:03d}.png"
        })

        cv2.rectangle(image_annotee, (x, y), (x+w, y+h), (0, 200, 255), 2)
        cv2.putText(image_annotee, str(mot['index']), (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # Encoder image annotée en base64
    _, buffer = cv2.imencode('.png', image_annotee)
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')

    return mots_b64, annotated_b64


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image reçue'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Fichier vide'}), 400

    # Sauvegarder temporairement l'image uploadée
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        mots, annotated_b64 = traiter_image(tmp_path)
        os.unlink(tmp_path)

        return jsonify({
            'success': True,
            'nb_mots': len(mots),
            'mots': mots,
            'annotated_image': annotated_b64
        })
    except Exception as e:
        import traceback # pour comprendre les erreurs quand y en a 
        traceback.print_exc()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
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