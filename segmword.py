import cv2 #CV2 sert pour le traitement d'image 
import numpy as np #pour les calculs
import os #pour gerer les fichiers 


def segmenter_mots(image_path, output_dir='mots_decoupes', debug=False): # debug = false car fonctionne pas avec flask donc va faire crasher le code 
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}") #Si pas d'image retourne une erreur

    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # on passe en niveaux de gris

    # Binarisation OTSU -> passe en noir et blanc et on inverse le noir et le blanc (lettre blanche sur fond noir)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # OTSU detecte les seuils

    # Hauteur médiane des caractères via composantes connexes
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8) # detecte les groupes de pixels (blancs) connectés entre eux 
    hauteurs, largeurs = [], []
    for i in range(1, num_labels): # on recupere la largeur et la hauteur de chaque composante pour garder que les mots et pas les "bruits"
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if 5 < h < h_img * 0.5 and 2 < w < w_img * 0.3 and area > 10:
            hauteurs.append(h)
            largeurs.append(w)

    if not hauteurs: #permet de vérifier qu'il y a des mots sur la pages
        print("Aucun caractère détecté.")
        return []

    hauteur_mediane = int(np.median(hauteurs)) # pour det la taille type d'une lettre (plus fiable que la moyenne car ignore les valeurs trop grandes ou trop petites)
    largeur_mediane = int(np.median(largeurs))

    # Isole les lignes de texte (lignes ou il y a des pixels representant de l'ecriture) permet de savoir où sont les lignes de textes 
    projection_y = np.sum(thresh, axis=1)
    seuil_ligne = projection_y.max() * 0.05
    # on parcours chaque ligne pour voir où elles s'arrêtent 
    in_ligne, lignes, y_debut = False, [], 0
    for y, val in enumerate(projection_y):
        if val > seuil_ligne and not in_ligne:
            in_ligne, y_debut = True, y
        elif val <= seuil_ligne and in_ligne:
            in_ligne = False
            if y - y_debut > hauteur_mediane * 0.3:
                lignes.append((y_debut, y))
    if in_ligne:
        lignes.append((y_debut, h_img))

    # on parcours de façon vertical pour trouvers l'espaces entres les mots : on récupère les groupes de pixels et on cherche l'écart entre ces groupes 
    all_gaps = []
    for (y1, y2) in lignes: #Y1 correspond au haut de la ligne et Y2 au bas de la ligne 
        bande = thresh[y1:y2, :]
        proj_x = np.sum(bande, axis=0) > 0
        gap_len, in_gap = 0, False
        for val in proj_x:
            if not val:
                in_gap, gap_len = True, gap_len + 1
            else:
                if in_gap and gap_len > 1:
                    all_gaps.append(gap_len)
                in_gap, gap_len = False, 0

# on cherche les espaces entre les mots de sorte à ce qu'ils soient pas confondu par les espaces entre les lettres 
    if len(all_gaps) > 2:
        mediane_gap = int(np.median(all_gaps))
        grands_gaps = [g for g in all_gaps if g > mediane_gap]
        seuil_inter_mot = int(np.median(grands_gaps)) if grands_gaps else mediane_gap
        kernel_w = max(int(seuil_inter_mot * 0.5), 2)
    else:
        kernel_w = max(largeur_mediane, 5)

    kernel_h = max(int(hauteur_mediane * 1.1), 5)

    # On dilate cad on grossit les lignes de textes pour que les mots forment un seul bloc entre eux
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h)) #cv2.MORPH_RECT: définit la forme du noyau de dilatation = rectangle ici
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #on detecte les contours RETR_EXTERNAL : contours externe, CHAIN_APPROX_SIMPLE : on prend les cono-tours en utilisant les coins des pixels pas tout le long des pixels 

    # chaque contour devient une "boite" et les boites trop petites sont supprimées 
    min_w = max(int(largeur_mediane * 0.3), 3)
    min_h = max(int(hauteur_mediane * 0.3), 3)
    boites = [(x, y, w, h) for cnt in contours
              for x, y, w, h in [cv2.boundingRect(cnt)]
              if w > min_w and h > min_h]
    # on trie les boites de gauche a droite et de haut en bas 
    tolerance = int(hauteur_mediane * 0.5)
    boites_triees = sorted(boites, key=lambda b: (b[1] // tolerance, b[0]))

    # Pour chaque mot detecté par le code on telecharge ça dans un nouveau png
    os.makedirs(output_dir, exist_ok=True)
    image_annotee = image.copy()
    mots = []
    for i, (x, y, w, h) in enumerate(boites_triees):
        cv2.imwrite(os.path.join(output_dir, f'mot_{i:03d}.png'), image[y:y+h, x:x+w])
        mots.append({'index': i, 'bbox': (x, y, w, h)})
        cv2.rectangle(image_annotee, (x, y), (x+w, y+h), (0, 200, 255), 2)
        cv2.putText(image_annotee, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # Mode debug : affiche les étapes intermédiaires -> none au debut donc ici inutile (a juste été utilisé pendant le process)
    if debug:
        # Étape A : image binarisée
        cv2.imshow('[DEBUG] 1 - Binarisation OTSU', thresh)
        cv2.waitKey(0)

        # Étape B : lignes détectées
        debug_lignes = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)
        for (y1, y2) in lignes:
            cv2.rectangle(debug_lignes, (0, y1), (w_img, y2), (0, 255, 0), 2)
        cv2.imshow('[DEBUG] 2 - Lignes detectees (vert)', debug_lignes)
        cv2.waitKey(0)

        # Étape C : image dilatée (ce que voit le détecteur de contours)
        cv2.imshow('[DEBUG] 3 - Image dilatee (kernel {}x{})'.format(kernel_w, kernel_h), dilated)
        cv2.waitKey(0)

        # Étape D : histogramme des gaps avec seuil
        if all_gaps:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.hist(all_gaps, bins=40, color='steelblue', edgecolor='white')
            plt.axvline(mediane_gap if len(all_gaps) > 2 else kernel_w,
                        color='orange', linestyle='--', label=f'médiane gaps ({mediane_gap}px)')
            plt.axvline(seuil_inter_mot if len(all_gaps) > 2 else kernel_w,
                        color='red', linestyle='--', label=f'seuil inter-mot ({seuil_inter_mot}px)')
            plt.axvline(kernel_w, color='green', linestyle='-', label=f'kernel_w choisi ({kernel_w}px)')
            plt.title('Distribution des gaps horizontaux')
            plt.xlabel('Taille du gap (px)')
            plt.ylabel('Fréquence')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'debug_gaps.png'))
            plt.show()

        print(f"\n[DEBUG] hauteur_mediane={hauteur_mediane}px | largeur_mediane={largeur_mediane}px")
        print(f"[DEBUG] kernel_w={kernel_w}px | kernel_h={kernel_h}px")
        print(f"[DEBUG] {len(lignes)} ligne(s) | {len(all_gaps)} gaps analysés")
        print(f"[DEBUG] {len(boites_triees)} mots détectés")

    # 10. Affichage résultat final
    #max_display = 1200
    #scale = min(max_display / w_img, max_display / h_img, 1.0)
    #display = cv2.resize(image_annotee, (int(w_img * scale), int(h_img * scale))) if scale < 1.0 else image_annotee
    #cv2.imshow('Mots détectés', display)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return mots


if __name__ == "__main__":
    # Passe debug=True pour voir les étapes intermédiaires et diagnostiquer
    resultats = segmenter_mots('sentence.png', output_dir='mots_decoupes', debug=True)
    print(f"\n{len(resultats)} mots sauvegardés dans 'mots_decoupes/'")