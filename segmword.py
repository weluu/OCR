import cv2
import numpy as np
import os

def segmenter_mots(image_path, output_dir='mots_decoupes', debug=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Calcule la taille typique des lettres
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    hauteurs = []
    for i in range(1, num_labels):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]
        if 5 < h < h_img * 0.5 and 2 < w < w_img * 0.3 and area > 10:
            hauteurs.append(h)

    if not hauteurs:
        return []

    hauteur_mediane = int(np.median(hauteurs))

    # Kernel basé sur la taille des lettres 
    kernel_w = max(int(hauteur_mediane * 0.4), 3)
    kernel_h = max(int(hauteur_mediane * 0.1), 1.5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrage et tri gauche→droite, haut→bas
    boites = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 10:
            boites.append((x, y, w, h))

    tolerance = int(hauteur_mediane * 0.5)
    boites_triees = sorted(boites, key=lambda b: (b[1] // tolerance, b[0]))

    os.makedirs(output_dir, exist_ok=True)
    image_annotee = image.copy()
    mots = []

    for i, (x, y, w, h) in enumerate(boites_triees):
        cv2.imwrite(os.path.join(output_dir, f'mot_{i:03d}.png'), image[y:y+h, x:x+w])
        mots.append({'index': i, 'bbox': (x, y, w, h)})
        cv2.rectangle(image_annotee, (x, y), (x+w, y+h), (0, 200, 255), 2)
        cv2.putText(image_annotee, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    return mots