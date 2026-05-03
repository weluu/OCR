import cv2
import numpy as np
import os


def segmenter_lettres(mots_dir: str = 'mot_seg', lettres_dir: str = 'lettres_decoupees') -> list[dict]:
    """
    Prend le dossier de mots produit par segmword.py et découpe chaque mot
    en lettres individuelles sauvegardées dans lettres_dir.

    Args:
        mots_dir   : dossier contenant les PNG des mots (défaut: 'mot_seg')
        lettres_dir: dossier de sortie pour les lettres (défaut: 'lettres_decoupees')

    Returns:
        liste de dict :
        {
            'mot_file' : 'mot_000.png',
            'mot_index': 0,
            'lettres'  : [
                {'file': 'mot_000_L00.png', 'bbox': (x0, x1)},
                ...
            ]
        }
    """
    import shutil

    if os.path.exists(lettres_dir):
        shutil.rmtree(lettres_dir)
    os.makedirs(lettres_dir)

    os.makedirs(lettres_dir, exist_ok=True)

    fichiers_mots = sorted([
        f for f in os.listdir(mots_dir) if f.endswith('.png')
    ])

    if not fichiers_mots:
        print(f"Aucun PNG trouvé dans '{mots_dir}'")
        return []

    resultats = []

    for mot_file in fichiers_mots:
        mot_path = os.path.join(mots_dir, mot_file)
        mot_idx  = int(mot_file.replace('mot_', '').replace('.png', ''))

        image_mot = cv2.imread(mot_path)
        if image_mot is None:
            print(f"  Impossible de lire {mot_file}, ignoré.")
            continue

        gray = cv2.cvtColor(image_mot, cv2.COLOR_BGR2GRAY)

        # Binarisation — lettres blanches sur fond noir
        _, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Nettoyage du bruit
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)

        # Dilatation horizontale pour boucher les creux des lettres rondes
        kernel_merge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        binary = cv2.dilate(binary, kernel_merge, iterations=1)

        # Projection verticale : somme des pixels blancs par colonne
        v_proj = np.sum(binary, axis=0)

        # Détection des segments de lettres
        lettres_info = []
        in_char = False
        x_start = 0
        W       = binary.shape[1]
        PADDING = 3
        MIN_W   = 4

        for x, val in enumerate(v_proj):
            if not in_char and val > 0:
                in_char = True
                x_start = x
            elif in_char and val == 0:
                in_char = False
                if (x - x_start) >= MIN_W:
                    x0 = max(0, x_start - PADDING)
                    x1 = min(W, x + PADDING)
                    lettres_info.append((x0, x1))

        # Dernière lettre si on termine en pleine colonne
        if in_char and (W - x_start) >= MIN_W:
            x0 = max(0, x_start - PADDING)
            lettres_info.append((x0, W))

        # Sauvegarde de chaque lettre
        lettres_sauvegardees = []
        for l_idx, (x0, x1) in enumerate(lettres_info):
            patch   = image_mot[:, x0:x1]
            l_fname = f"mot_{mot_idx:03d}_L{l_idx:02d}.png"
            l_path  = os.path.join(lettres_dir, l_fname)
            cv2.imwrite(l_path, patch)
            lettres_sauvegardees.append({
                'file': l_fname,
                'bbox': (x0, x1)
            })

        resultats.append({
            'mot_file' : mot_file,
            'mot_index': mot_idx,
            'lettres'  : lettres_sauvegardees
        })

        print(f"  {mot_file} → {len(lettres_sauvegardees)} lettre(s)")

    total = sum(len(r['lettres']) for r in resultats)
    print(f"\n{total} lettre(s) sauvegardées dans '{lettres_dir}/'")
    return resultats


if __name__ == "__main__":
    resultats = segmenter_lettres(
        mots_dir   ='mot_seg',
        lettres_dir='lettres_decoupees'
    )