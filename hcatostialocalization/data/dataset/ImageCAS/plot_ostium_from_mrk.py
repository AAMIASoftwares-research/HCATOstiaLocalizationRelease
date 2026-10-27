"""
plot_ostium_from_mrk.py
=======================
Visualizza le annotazioni degli osti (R e L) direttamente dai file .mrk.json
dei 3 annotatori, senza bisogno del JSON finale di consenso.

Mostra:
  - La slice assiale all'altezza del centroide di consenso (media dei 3 annotatori)
  - I 3 punti dei singoli annotatori (colori diversi)
  - Il centroide di consenso (stella bianca)

UTILIZZO:
  Modifica i PATH in fondo al file e lancia:
      python plot_ostium_from_mrk.py
  oppure incollalo in una cella di notebook Jupyter.
"""

import os
import glob
import json
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

# ==============================================================================
# 0. CONFIGURAZIONE PERCORSI 
# ==============================================================================
current_dir = os.path.abspath(os.getcwd())
while current_dir != os.path.dirname(current_dir):
    if os.path.exists(os.path.join(current_dir, "hcatostialocalization")):
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        break
    current_dir = os.path.dirname(current_dir)

from hcatostialocalization.config.config import config
from hcatostialocalization.data.dataset.Base_Classes_Dataset import LabeledDataset
from hcatostialocalization.data.dataset.Heatmap_Config import HeatmapParams

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def lps_to_ras(lps: list) -> list:
    """Converte da LPS (coordinateSystem di 3D Slicer) a RAS."""
    return [-lps[0], -lps[1], lps[2]]

def get_ijk_from_lps(img: sitk.Image, lps: list) -> tuple:
    """
    Converte coordinate fisiche LPS in indici voxel IJK.
    SimpleITK usa LPS internamente, quindi TransformPhysicalPointToIndex
    accetta direttamente LPS.
    """
    return img.TransformPhysicalPointToIndex(lps)

def load_points_from_mrk(mrk_path: str) -> dict:
    """
    Legge un file .mrk.json di 3D Slicer e restituisce un dict:
      {"R": [x, y, z], "L": [x, y, z]}  in coordinate LPS.
    Ritorna None per il lato se non trovato.
    """
    points = {"R": None, "L": None}
    with open(mrk_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    control_points = data.get("markups", [{}])[0].get("controlPoints", [])
    for pt in control_points:
        label = pt.get("label", "").strip().upper()
        pos   = pt.get("position")
        if pos is None:
            continue
        # Riconoscimento robusto: "R", "RCA", "RIGHT" → lato R
        #                         "L", "LCA", "LEFT"  → lato L
        # NOTA: il check 'R' in label deve venire PRIMA di 'L' in label
        # perché "RL" o "LR" potrebbero creare ambiguità (non comune).
        if label in ("R", "RCA", "RIGHT") or (label.startswith("R") and "L" not in label):
            points["R"] = pos
        elif label in ("L", "LCA", "LEFT") or (label.startswith("L") and "R" not in label):
            points["L"] = pos
        # Fallback generico per etichette non standard
        elif "R" in label and "L" not in label:
            points["R"] = pos
        elif "L" in label and "R" not in label:
            points["L"] = pos

    return points


def plot_ostium_from_annotators(
    patient_id: str,
    mrk_files: list[str],
    image_path: str,
    side: str = "both",
    window_min: int = -200,
    window_max: int = 800,
):
    """
    Plotta la slice assiale con i punti degli annotatori e il centroide di consenso.

    Parameters
    ----------
    patient_id : str
        ID del paziente (es. "387"), usato solo per il titolo.
    mrk_files : list[str]
        Lista dei percorsi assoluti ai file .mrk.json degli annotatori.
    image_path : str
        Percorso al file .nii.gz (o .nii) dell'immagine del paziente.
    side : str
        "R", "L", o "both" (default). Se "both" plotta entrambi i lati
        su due subplot affiancati.
    window_min, window_max : int
        Range HU per il windowing (default: [-200, 800] buono per contrasto coronarico).
    """
    if not os.path.exists(image_path):
        print(f"[!] Immagine non trovata: {image_path}")
        return

    # --- 1. Carica l'immagine ---
    img = sitk.ReadImage(image_path)
    img_array = sitk.GetArrayFromImage(img)  # shape: (Z, Y, X) = (K, J, I)

    # --- 2. Raccoglie i punti da tutti gli annotatori ---
    annotator_colors = ["#FF6B6B", "#FFD93D", "#6BCB77"]  # rosso, giallo, verde
    annotator_labels = [f"Annotatore {i+1}" for i in range(len(mrk_files))]

    points_per_side = {"R": [], "L": []}

    for mrk_f in mrk_files:
        pts = load_points_from_mrk(mrk_f)
        for s in ["R", "L"]:
            if pts[s] is not None:
                points_per_side[s].append(pts[s])
            else:
                print(f"[!] Lato {s} non trovato in: {os.path.basename(mrk_f)}")

    # --- 3. Calcola i centroidi di consenso (media = equivalente a K-means 1 cluster) ---
    consensus = {}
    ijk_consensus = {}
    for s in ["R", "L"]:
        if len(points_per_side[s]) == 0:
            print(f"[!] Nessun punto trovato per il lato {s} — non verrà plottato.")
            consensus[s] = None
            ijk_consensus[s] = None
        else:
            mean_lps = np.mean(points_per_side[s], axis=0).tolist()
            consensus[s] = mean_lps
            ijk_consensus[s] = get_ijk_from_lps(img, mean_lps)
            print(f"[+] Lato {s} — centroide LPS: {[round(c, 2) for c in mean_lps]}")
            print(f"    → IJK: {ijk_consensus[s]}")
            print(f"    → Punti annotatori ({len(points_per_side[s])}): OK")

    # --- 4. Determina i lati da plottare ---
    sides_to_plot = []
    if side == "both":
        sides_to_plot = [s for s in ["R", "L"] if ijk_consensus[s] is not None]
    elif side in ["R", "L"] and ijk_consensus.get(side) is not None:
        sides_to_plot = [side]
    else:
        print(f"[!] Lato '{side}' non disponibile.")
        return

    n_plots = len(sides_to_plot)
    if n_plots == 0:
        print("[!] Nessun lato da plottare.")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(9 * n_plots, 8))
    if n_plots == 1:
        axes = [axes]

    for ax, s in zip(axes, sides_to_plot):
        i_c, j_c, k_c = ijk_consensus[s]   # IJK (X, Y, Z)
        k_c = int(np.clip(k_c, 0, img_array.shape[0] - 1))

        # Estrai e applica windowing alla slice
        slice_2d = img_array[k_c, :, :]
        slice_2d = np.clip(slice_2d, window_min, window_max)

        ax.imshow(slice_2d, cmap="gray", origin="upper")

        # Plotta i singoli annotatori
        for idx, (lps_pt, color) in enumerate(zip(points_per_side[s], annotator_colors)):
            ijk_pt = get_ijk_from_lps(img, lps_pt)
            i_pt, j_pt, k_pt_ann = int(ijk_pt[0]), int(ijk_pt[1]), int(ijk_pt[2])
            # Nota: imshow usa (X=colonne, Y=righe) → (i, j) = (X, Y)
            ax.scatter(i_pt, j_pt, color=color, marker="o", s=120, zorder=4,
                       label=annotator_labels[idx] if s == sides_to_plot[0] else "")
            # Piccola freccia se il punto è su slice diversa dal consenso
            if abs(k_pt_ann - k_c) > 2:
                ax.annotate(
                    f"z={k_pt_ann}", (i_pt, j_pt),
                    textcoords="offset points", xytext=(5, -10),
                    color=color, fontsize=7
                )

        # Plotta il centroide di consenso (stella bianca)
        ax.scatter(int(i_c), int(j_c), color="white", marker="*", s=400,
                   edgecolors="black", linewidths=0.8, zorder=5, label="Consenso")

        side_name = "Destro (RCA)" if s == "R" else "Sinistro (LCA)"
        ax.set_title(
            f"Paziente {patient_id} — Ostio {side_name}\n"
            f"Slice K={k_c}  |  Consenso IJK=({int(i_c)}, {int(j_c)}, {k_c})",
            fontsize=12, pad=10
        )
        ax.axis("off")

    # Legenda comune
    legend_patches = [
        mpatches.Patch(color=annotator_colors[i], label=annotator_labels[i])
        for i in range(len(mrk_files))
    ]
    legend_patches.append(
        mpatches.Patch(color="white", label="Consenso (media)")
    )
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(legend_patches),
               fontsize=10, framealpha=0.8)

    fig.suptitle(f"Verifica Annotazioni Osti — Paziente {patient_id}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# ESECUZIONE — MODIFICA QUI I PERCORSI
# ==============================================================================
if __name__ == "__main__":
    from hcatostialocalization.config.config import config

    PATIENT_ID = "387"

    # Cartella con i file .mrk.json degli annotatori
    MRK_DIR = config.active_paths.data_root + "/OstiumAnnotations_ImageCAS/SlicerObjects"

    # Immagine del paziente
    IMAGE_PATH = config.active_paths.data_root + f"/Datasets/ImageCAS/Extracted_Images/{PATIENT_ID}.nii.gz"
    if not os.path.exists(IMAGE_PATH):
        IMAGE_PATH = config.active_paths.data_root + f"/Datasets/ImageCAS/Extracted_Images/{PATIENT_ID}.nii"

    # Cerca tutti i file .mrk.json relativi al paziente 387
    # Pattern: "387_*.mrk.json" oppure "0387_*.mrk.json"
    mrk_files = sorted(
        glob.glob(os.path.join(MRK_DIR, f"{PATIENT_ID}_*.mrk.json")) +
        glob.glob(os.path.join(MRK_DIR, f"{int(PATIENT_ID):04d}_*.mrk.json"))
    )

    if not mrk_files:
        print(f"[!] Nessun file .mrk.json trovato per il paziente {PATIENT_ID} in:\n    {MRK_DIR}")
    else:
        print(f"[+] Trovati {len(mrk_files)} file di annotazione:")
        for f in mrk_files:
            print(f"    {os.path.basename(f)}")
        print()

        plot_ostium_from_annotators(
            patient_id=PATIENT_ID,
            mrk_files=mrk_files,
            image_path=IMAGE_PATH,
            side="both",         # "R", "L", o "both"
            window_min=-200,
            window_max=800,
        )
