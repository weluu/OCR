import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
import keras
print(keras.backend.backend())  # debug pour voir si keras est bien charger doit print 'torch'
import numpy as np
import torch.nn.functional as F

from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision.transforms import v2

os.environ["KERAS_BACKEND"] = "torch"

# ── Configuration ─────────────────────────────────────────────────────────────
LETTRES_DIR  = 'lettres_decoupees'
WEIGHTS_PATH = 'Pytorch/best_emnist_model.weights.h5'  # CHemin du modèle à modifier si changement de chemin
NUM_CLASSES  = 62

# Classes EMNIST byclass — liste statique pour ne pas de charger le dataset entièrement
CLASSES = (
    [str(d) for d in range(10)]         # 0-9
    + [chr(c) for c in range(65, 91)]   # A-Z
    + [chr(c) for c in range(97, 123)]  # a-z
)

# ── Architecture du modèle ────────────────────────────────────────────────────
class EMNISTModel(keras.Model):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1  = keras.layers.Conv2D(32,  kernel_size=3, padding="same", activation="relu")
        self.bn1    = keras.layers.BatchNormalization()
        self.sdrop1 = keras.layers.SpatialDropout2D(0.1)
        self.pool1  = keras.layers.MaxPooling2D(2)
        self.conv2  = keras.layers.Conv2D(64,  kernel_size=3, padding="same", activation="relu")
        self.bn2    = keras.layers.BatchNormalization()
        self.sdrop2 = keras.layers.SpatialDropout2D(0.1)
        self.pool2  = keras.layers.MaxPooling2D(2)
        self.conv3  = keras.layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")
        self.bn3    = keras.layers.BatchNormalization()
        self.sdrop3 = keras.layers.SpatialDropout2D(0.1)
        self.conv4  = keras.layers.Conv2D(256, kernel_size=3, padding="same", activation="relu")
        self.bn4    = keras.layers.BatchNormalization()
        self.gap      = keras.layers.GlobalAveragePooling2D()
        self.dropout1 = keras.layers.Dropout(0.4)
        self.fc1      = keras.layers.Dense(128, activation="relu",
                            kernel_regularizer=keras.regularizers.L2(1e-4))
        self.dropout2 = keras.layers.Dropout(0.3)
        self.fc2      = keras.layers.Dense(num_classes)

    def call(self, x, training=False):
        x = keras.ops.transpose(x, axes=(0, 2, 3, 1))
        x = self.sdrop1(self.pool1(self.bn1(self.conv1(x), training=training)), training=training)
        x = self.sdrop2(self.pool2(self.bn2(self.conv2(x), training=training)), training=training)
        x = self.sdrop3(self.bn3(self.conv3(x), training=training), training=training)
        x = self.bn4(self.conv4(x), training=training)
        x = self.gap(x)
        x = self.dropout1(x, training=training)
        x = self.fc1(x)
        x = self.dropout2(x, training=training)
        return self.fc2(x)


# ── Chargement du modèle ──────────────────────────────────────────────────────
def charger_modele(weights_path: str) -> keras.Model:
    model = EMNISTModel()
    model(torch.zeros(1, 1, 28, 28), training=False)  # build
    model.load_weights(weights_path)
    model.trainable = False
    print(f"Modele chargé depuis '{weights_path}'")
    return model


# ── Preprocessing d'un patch → tenseur 28×28 ─────────────────────────────────
_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.1307], std=[0.3081])
])

def patch_to_tensor(image_path: str) -> torch.Tensor:
    """
    Charge une image lettre et la convertit en tenseur (1, 1, 28, 28).
    Même logique que preprocess_image() dans EMNIST_test_images_v2.ipynb.
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    # Inverser si fond blanc
    if arr.mean() > 127:
        arr = 255 - arr

    # Crop serré
    arr_smooth = gaussian_filter(arr.astype(np.float32), sigma=1)
    threshold  = max(20, arr_smooth.max() * 0.08)
    rows = np.any(arr_smooth > threshold, axis=1)
    cols = np.any(arr_smooth > threshold, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        arr = arr[r0:r1+1, c0:c1+1]

    # Resize proportionnel dans 20×20 (marge 4px)
    MARGIN = 4
    h, w   = arr.shape
    ratio  = (28 - 2 * MARGIN) / max(h, w)
    new_h  = max(1, int(h * ratio))
    new_w  = max(1, int(w * ratio))
    img_r  = Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS)

    # Centrage sur canvas 28×28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off  = (28 - new_h) // 2
    x_off  = (28 - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = np.array(img_r)

    tensor = _transform(Image.fromarray(canvas))
    return tensor.unsqueeze(0)  # (1, 1, 28, 28)


# ── Prédiction ────────────────────────────────────────────────────────────────
def predire_lettre(model: keras.Model, image_path: str) -> tuple[str, float]:
    """
    Prédit le caractère d'une image lettre.

    Returns:
        (caractère, confiance en %)
    """
    tensor = patch_to_tensor(image_path)
    with torch.no_grad():
        logits = model(tensor, training=False)
        probs  = F.softmax(logits, dim=1).cpu().squeeze()
    idx  = probs.argmax().item()
    return CLASSES[idx], round(probs[idx].item() * 100, 1)


# ── Reconstruction ASCII ──────────────────────────────────────────────────────
def reconstruire_texte(lettres_dir: str, model: keras.Model) -> str:
    """
    Parcourt lettres_decoupees/, prédit chaque lettre et reconstruit
    la phrase ligne par ligne, mot par mot.

    Format des fichiers attendu : mot_000_L00.png
    """
    fichiers = sorted([
        f for f in os.listdir(lettres_dir) if f.endswith('.png')
    ])

    if not fichiers:
        print(f"Aucune lettre trouvée dans '{lettres_dir}'")
        return ""

    # Regroupement par mot : { mot_idx: [ (lettre_idx, char, conf) ] }
    mots = {}
    for fname in fichiers:
        parts    = fname.replace('.png', '').split('_')  # ['mot', '000', 'L00']
        mot_idx  = int(parts[1])
        l_idx    = int(parts[2][1:])  # retire le 'L'

        path       = os.path.join(lettres_dir, fname)
        char, conf = predire_lettre(model, path)

        mots.setdefault(mot_idx, []).append((l_idx, char, conf))
        print(f"  {fname} → '{char}'  ({conf}%)")

    # Reconstruction mot par mot dans l'ordre
    mots_texte = []
    for mot_idx in sorted(mots):
        lettres = sorted(mots[mot_idx], key=lambda x: x[0])
        mot_str = "".join(char for _, char, _ in lettres)
        mots_texte.append(mot_str)

    texte = " ".join(mots_texte)

    print(f"\n{'='*50}")
    print("  TEXTE RECONNU :")
    print(f"{'='*50}")
    print(f"  {texte}")
    print(f"{'='*50}\n")

    return texte


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = charger_modele(WEIGHTS_PATH)
    texte = reconstruire_texte(LETTRES_DIR, model)
