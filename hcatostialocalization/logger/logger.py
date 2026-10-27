import os
import sys
import glob
import tempfile
import csv
import re
import torch
import mlflow
import numpy as np
from datetime import datetime
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# 0. PROJECT ROOT RESOLUTION
# ==============================================================================
current_dir = os.path.abspath(os.getcwd())
PROJECT_ROOT = current_dir
while current_dir != os.path.dirname(current_dir):
    if os.path.exists(os.path.join(current_dir, "hcatostialocalization")):
        PROJECT_ROOT = current_dir
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        break
    current_dir = os.path.dirname(current_dir)


def resolve_project_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


# ==============================================================================
# 1. EXPERIMENT LOGGER
# ==============================================================================
class ExperimentLogger:
    """
    Handles experiment tracking via MLflow, model checkpointing, and
    generation of diagnostic visualizations.

    Each training run is associated with a unique MLflow run ID that is
    persisted in the checkpoint, enabling seamless resumption of logging
    across interrupted experiments.
    """

    def __init__(self, config):
        # Extract logging parameters from the experiment configuration.
        self.exp_name = config.get("logger", {}).get("experiment_name", "Ostia_Localization")
        self.run_name = config.get("logger", {}).get("run_name", "Baseline_Run")
        self.model_name = config.get("model", {}).get("name", "unknown_model")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger_cfg = config.get("logger", {})
        resume_dir = logger_cfg.get("resume_dir", None)
        self.mlflow_run_id = logger_cfg.get("mlflow_run_id", None)

        # Resolve checkpoint directory (reuse existing path when resuming).
        if resume_dir:
            self.save_dir = resolve_project_path(resume_dir)
        else:
            base_save_dir = resolve_project_path(logger_cfg.get("save_dir", "./checkpoints"))
            run_folder_name = f"{self.timestamp}_{self.run_name}"
            self.save_dir = os.path.join(base_save_dir, self.model_name, run_folder_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.epoch_history = list(logger_cfg.get("epoch_history", []))

        self.best_val_loss = float('inf')

        # Initialize MLflow tracking backend.
        tracking_dir = resolve_project_path(logger_cfg.get("tracking_dir", "./mlruns"))
        os.makedirs(tracking_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{tracking_dir}")
        mlflow.set_experiment(self.exp_name)
        if mlflow.active_run():
            mlflow.end_run()

        # Resume an existing run or start a new one.
        if self.mlflow_run_id:
            self.run = mlflow.start_run(run_id=self.mlflow_run_id)
        else:
            self.run = mlflow.start_run(run_name=self.run_name)
            self.mlflow_run_id = self.run.info.run_id
        print(f"[*] MLflow run started: '{self.run_name}' (experiment: {self.exp_name})")

        # =================================================================
        # --- CLEAN AND TABULAR LOGGING ON MLFLOW ---
        # =================================================================
        from omegaconf import OmegaConf
        
        # 1. Save clean and readable YAML text file (without !!python tags)
        clean_yaml_str = OmegaConf.to_yaml(config, resolve=True)
        mlflow.log_text(clean_yaml_str, "clean_config.yaml")
        
        # 2. Extract parameters for main MLflow table
        primitive_config = OmegaConf.to_container(config, resolve=True)
        
        # Internal function to "flatten" dictionary (e.g. from nested dict to "loss.ce_weight")
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
            
        flat_params = flatten_dict(primitive_config)
        
        # Log all parameters one by one in the graphical interface
        for key, value in flat_params.items():
            try:
                # MLflow accepts strings, numbers or booleans for parameters
                mlflow.log_param(key, str(value) if isinstance(value, list) else value)
            except Exception:
                pass
        # =================================================================

        mlflow.log_param("checkpoint_dir", self.save_dir)
        mlflow.log_param("mlflow_tracking_dir", tracking_dir)
        self.outlier_records = []

    def log_metrics(self, metrics_dict, epoch):
        mlflow.log_metrics(metrics_dict, step=epoch)

    def log_epoch_metrics(
        self,
        epoch,
        train_loss,
        learning_rate,
        timing_metrics=None,
    ):
        """Logs per-epoch training metrics to MLflow."""
        metrics = {
            "train_loss":    train_loss,
            "learning_rate": learning_rate,
        }
        if timing_metrics:
            metrics.update(timing_metrics)

        self.log_metrics(metrics, epoch)

        self.epoch_history.append({
            "epoch":        epoch,
            "train_loss":   train_loss,
            "learning_rate": learning_rate,
            **(timing_metrics or {}),
        })

    def save_checkpoint(self, model, optimizer, epoch, val_loss, scheduler=None, training_state=None):
        """
        Saves a full training checkpoint containing model weights, optimizer
        state, scheduler state, and logging metadata.

        Two checkpoint files are maintained:
            - last_model.pth:  Overwritten at every epoch.
            - best_model.pth:  Saved only when validation loss improves.
        """
        is_best = val_loss < self.best_val_loss
        if is_best:
            print(f"    [*] New best model: val_loss {self.best_val_loss:.4f} -> {val_loss:.4f}")
            self.best_val_loss = val_loss

        scheduler_state = None
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            scheduler_state = scheduler.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler_state,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'save_dir': self.save_dir,
            'mlflow_run_id': self.mlflow_run_id,
            'epoch_history': self.epoch_history,
            'training_state': training_state or {},
            'model_name': self.model_name,
            'run_name': self.run_name,
            'timestamp': self.timestamp,
        }

        last_path = os.path.join(self.save_dir, "last_model.pth")
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(checkpoint, best_path)

    def save_loss_plot(self):
        """Saves the end-of-training train loss curve to MLflow."""
        if not self.epoch_history:
            return

        epochs       = [item["epoch"]      for item in self.epoch_history]
        train_losses = [item["train_loss"] for item in self.epoch_history]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_losses, marker="o", label="Train loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        plot_path = os.path.join(tempfile.gettempdir(), f"loss_train_{self.timestamp}.png")
        plt.savefig(plot_path)
        plt.close(fig)
        mlflow.log_artifact(plot_path, "training_curves")
        os.remove(plot_path)

    def save_validation_plot(self):
        """Saves the end-of-training validation curves (Median, p75, Mean) to MLflow."""
        val_entries = [item for item in self.epoch_history if "val_dist_R_median" in item or "val_global_median" in item]
        if not val_entries:
            return

        epochs = [item["epoch"] for item in val_entries]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Global Error
        if "val_global_median" in val_entries[0]:
            med_glob = [item.get("val_global_median", 0.0) for item in val_entries]
            p75_glob = [item.get("val_global_p75", 0.0) for item in val_entries]
            mean_glob = [item.get("val_global_mean", 0.0) for item in val_entries]
            axes[0].plot(epochs, med_glob, marker="o", color="blue", linewidth=2, label="Median (p50)")
            axes[0].plot(epochs, p75_glob, marker="s", color="orange", linewidth=1.8, linestyle="--", label="75th percentile")
            axes[0].plot(epochs, mean_glob, marker="^", color="gray", linewidth=1.2, linestyle=":", label="Mean")
            axes[0].set_title("Global Validation Error (mm)")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Distance (mm)")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

        # 2. Right Ostium (R)
        if "val_dist_R_median" in val_entries[0]:
            med_r = [item.get("val_dist_R_median", 0.0) for item in val_entries]
            p75_r = [item.get("val_dist_R_p75", 0.0) for item in val_entries]
            mean_r = [item.get("val_dist_R_mean", 0.0) for item in val_entries]
            axes[1].plot(epochs, med_r, marker="o", color="green", linewidth=2, label="Median (p50)")
            axes[1].plot(epochs, p75_r, marker="s", color="lime", linewidth=1.8, linestyle="--", label="75th percentile")
            axes[1].plot(epochs, mean_r, marker="^", color="gray", linewidth=1.2, linestyle=":", label="Mean")
            axes[1].set_title("Right Ostium (RCA) Error (mm)")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Distance (mm)")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

        # 3. Left Ostium (L)
        if "val_dist_L_median" in val_entries[0]:
            med_l = [item.get("val_dist_L_median", 0.0) for item in val_entries]
            p75_l = [item.get("val_dist_L_p75", 0.0) for item in val_entries]
            mean_l = [item.get("val_dist_L_mean", 0.0) for item in val_entries]
            axes[2].plot(epochs, med_l, marker="o", color="red", linewidth=2, label="Median (p50)")
            axes[2].plot(epochs, p75_l, marker="s", color="salmon", linewidth=1.8, linestyle="--", label="75th percentile")
            axes[2].plot(epochs, mean_l, marker="^", color="gray", linewidth=1.2, linestyle=":", label="Mean")
            axes[2].set_title("Left Ostium (LCA) Error (mm)")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Distance (mm)")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()

        fig.suptitle("Validation Localization Trends: Median vs p75 vs Mean", fontsize=13)
        plt.tight_layout()

        plot_path = os.path.join(tempfile.gettempdir(), f"val_metrics_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(plot_path, "training_curves")
        os.remove(plot_path)

    @staticmethod
    def _safe_name(value) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"

    @staticmethod
    def _to_numpy_zyx(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32).reshape(3)

    @staticmethod
    def _set_orientation_labels(ax, xlabel: str, ylabel: str):
        """Hides ticks but shows anatomical direction of each axis (e.g. 'L → R')"""
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)

    @staticmethod
    def _draw_point_if_inside(ax, coord_a: float, coord_b: float, marker: str, color: str, label: str):
        ax.plot(coord_a, coord_b, marker=marker, color=color, markersize=8,
                markeredgewidth=2, fillstyle="none", label=label)

    def _build_outlier_figure(
        self,
        img_vol: np.ndarray,
        gt_zyx: np.ndarray,
        pred_zyx: np.ndarray,
        side: str,
        epoch: int,
        sample_id: str,
        dataset_name: str,
        distance_mm: float,
    ) -> plt.Figure:
        """
        Build a compact 3-view patch around GT/prediction for outlier inspection.
        Volume and coordinates are in RAS voxel space: axis 0 = R, axis 1 = A, axis 2 = S
        (the *_zyx names are historical: element 0/1/2 = index along axis 0/1/2).
        Every view is shown with origin='lower' and the first displayed index horizontal:
          Axial    x: L -> R, y (bottom -> top): P -> A
          Coronal  x: L -> R, y (bottom -> top): I -> S
          Sagittal x: P -> A, y (bottom -> top): I -> S
        """
        D, H, W = img_vol.shape
        center = (gt_zyx + pred_zyx) / 2.0
        max_delta = float(np.max(np.abs(gt_zyx - pred_zyx)))
        half = int(np.clip(np.ceil(max_delta / 2.0) + 16, 32, 96))

        z0 = max(0, int(round(center[0])) - half)
        z1 = min(D, int(round(center[0])) + half + 1)
        y0 = max(0, int(round(center[1])) - half)
        y1 = min(H, int(round(center[1])) + half + 1)
        x0 = max(0, int(round(center[2])) - half)
        x1 = min(W, int(round(center[2])) + half + 1)

        patch = img_vol[z0:z1, y0:y1, x0:x1]
        if patch.size == 0:
            patch = img_vol
            z0, y0, x0 = 0, 0, 0
            z1, y1, x1 = D, H, W

        gt = gt_zyx - np.array([z0, y0, x0], dtype=np.float32)
        pred = pred_zyx - np.array([z0, y0, x0], dtype=np.float32)
        center_local = np.clip((gt + pred) / 2.0, 0, np.array(patch.shape) - 1)
        ic, jc, kc = [int(round(v)) for v in center_local]  # (R, A, S) indices

        finite = patch[np.isfinite(patch)]
        vmin, vmax = (np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0))
        if vmin == vmax:
            vmax = vmin + 1.0

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle(
            f"OUTLIER | epoch {epoch} | {dataset_name} | {sample_id} | "
            f"{side} | dist={distance_mm:.2f} mm",
            fontsize=9,
        )

        # (title, transposed 2D slice -> rows = vertical axis, point (x, y), axis labels)
        views = [
            ("Axial",    patch[:, :, kc].T, (gt[0], gt[1]), (pred[0], pred[1]), "L → R", "P → A"),
            ("Coronal",  patch[:, jc, :].T, (gt[0], gt[2]), (pred[0], pred[2]), "L → R", "I → S"),
            ("Sagittal", patch[ic, :, :].T, (gt[1], gt[2]), (pred[1], pred[2]), "P → A", "I → S"),
        ]

        for ax, (title, img, gt_xy, pred_xy, xlabel, ylabel) in zip(axes, views):
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            self._draw_point_if_inside(ax, gt_xy[0], gt_xy[1], "+", "lime", "GT")
            self._draw_point_if_inside(ax, pred_xy[0], pred_xy[1], "o", "cyan", "Pred")
            ax.plot([gt_xy[0], pred_xy[0]], [gt_xy[1], pred_xy[1]], color="yellow",
                    linewidth=1, alpha=0.8)
            ax.set_title(f"{title} patch")
            self._set_orientation_labels(ax, xlabel, ylabel)
            ax.legend(loc="upper right", fontsize=7)

        plt.tight_layout()
        return fig

    def log_outlier(
        self,
        epoch: int,
        sample_id: str,
        dataset_name: str,
        side: str,
        image_volume,
        gt_zyx,
        pred_zyx,
        distance_mm: float,
        spacing,
        threshold_mm: float,
    ):
        """Track validation outliers with a CSV row and a GT-vs-pred patch figure."""
        sample_id = str(sample_id)
        dataset_name = str(dataset_name)
        gt_np = self._to_numpy_zyx(gt_zyx)
        pred_np = self._to_numpy_zyx(pred_zyx)
        spacing_np = np.asarray(spacing.detach().cpu().numpy() if isinstance(spacing, torch.Tensor) else spacing,
                                dtype=np.float32).reshape(3)

        record = {
            "epoch": int(epoch),
            "dataset": dataset_name,
            "sample_id": sample_id,
            "side": side,
            "distance_mm": float(distance_mm),
            "threshold_mm": float(threshold_mm),
            "gt_z": float(gt_np[0]),
            "gt_y": float(gt_np[1]),
            "gt_x": float(gt_np[2]),
            "pred_z": float(pred_np[0]),
            "pred_y": float(pred_np[1]),
            "pred_x": float(pred_np[2]),
            "spacing_z": float(spacing_np[0]),
            "spacing_y": float(spacing_np[1]),
            "spacing_x": float(spacing_np[2]),
        }
        self.outlier_records.append(record)

        outlier_dir = os.path.join(self.save_dir, "outliers")
        os.makedirs(outlier_dir, exist_ok=True)
        csv_path = os.path.join(outlier_dir, "outliers.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            writer.writeheader()
            writer.writerows(self.outlier_records)
        mlflow.log_artifact(csv_path, "outliers")

        img_np = image_volume.detach().cpu().float().numpy() if isinstance(image_volume, torch.Tensor) else image_volume
        img_np = np.asarray(img_np, dtype=np.float32)
        while img_np.ndim > 3:
            img_np = np.squeeze(img_np, axis=0)

        fig = self._build_outlier_figure(
            img_np, gt_np, pred_np, side, epoch, sample_id, dataset_name, distance_mm
        )
        safe_id = self._safe_name(sample_id)
        fig_path = os.path.join(
            outlier_dir,
            f"epoch_{epoch:03d}_{safe_id}_{side}_dist_{distance_mm:.1f}mm.png",
        )
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(fig_path, f"outliers/epoch_{epoch:03d}/{safe_id}_{side}")

    # ===========================================================================
    # FIXED VISUALIZATION: per-epoch 3-panel monitoring + GIF compilation
    # ===========================================================================

    def init_fixed_viz(self, train_samples: list, val_samples: list):
        """
        Register fixed samples for qualitative per-epoch monitoring.

        train_samples — 3 dicts (1 per dataset), each with:
            'image_R'      (1, pz, py, px) crop centred on the right ostium.
            'gt_in_crop_R' [z, y, x] GT in crop coordinates.
            'image_L'      (1, pz, py, px) crop centred on the left ostium.
            'gt_in_crop_L' [z, y, x] GT in crop coordinates.
            'sample_id', 'dataset_name', 'spacing'.
            Trainer populates: 'pred_R', 'pred_L'  — (pz, py, px) CPU tensors.

        val_samples — 3 dicts (1 per dataset), each with:
            'image_full'   (1, D, H, W) full preprocessed volume.
            'gt_R_zyx'     [z, y, x] right ostium GT in full-volume coords.
            'gt_L_zyx'     [z, y, x] left ostium GT in full-volume coords.
            'sample_id', 'dataset_name', 'spacing'.
            Trainer populates: 'pred_R_full', 'pred_L_full' — (D, H, W) CPU tensors
            via sliding-window inference.

        Train samples produce GIF frames; val samples do not.
        """
        self._train_viz = train_samples
        self._val_viz   = val_samples
        self._gif_frames_dir = os.path.join(self.save_dir, "gif_frames")
        os.makedirs(self._gif_frames_dir, exist_ok=True)

    @staticmethod
    def _gaussian_slice_2d(gt_y: float, gt_x: float, sigma: float, shape: tuple) -> np.ndarray:
        """
        2D Gaussian centred at (gt_y, gt_x). Returns a single-voxel impulse when sigma=0.
        """
        H, W = shape
        if sigma <= 0.0:
            out = np.zeros((H, W), dtype=np.float32)
            yi, xi = int(round(gt_y)), int(round(gt_x))
            if 0 <= yi < H and 0 <= xi < W:
                out[yi, xi] = 1.0
            return out
        gy, gx = np.mgrid[:H, :W].astype(np.float32)
        return np.exp(-((gy - gt_y) ** 2 + (gx - gt_x) ** 2) / (2 * sigma ** 2))

    def _build_panel_figure(
        self,
        img_slice:      np.ndarray,
        pred_vol:       np.ndarray,
        gt_y:           float,
        gt_x:           float,
        gt_z:           float,
        gaussian_slice: np.ndarray,
        side:           str,
        epoch:          int,
        sigma:          float,
        sample_id:      str,
        dataset_name:   str,
    ) -> plt.Figure:
        """
        Build a 4-panel monitoring figure for one ostium (R or L).

        Panel 1 — CT + GT crosshair (ground-truth reference point).
        Panel 2 — CT + GT crosshair + Gaussian target heatmap (curriculum target).
        Panel 3 — CT + model prediction heatmap + GT crosshair.
        Panel 4 — S activation profile: max prediction per axial slice vs GT S index.

        pred_vol is (R, A, S). img_slice is the axial slice already transposed to (A, R);
        gt_x = R index, gt_y = A index, gt_z = S index. With origin='lower':
        x axis L -> R, y axis (bottom -> top) P -> A.
        """
        k = max(0, min(int(round(gt_z)), pred_vol.shape[2] - 1))
        pred_slice = pred_vol[:, :, k].T  # (A, R), same orientation as img_slice
        cmap       = 'Greens' if side == 'R' else 'Reds'
        side_label = 'Right'  if side == 'R' else 'Left'

        finite = img_slice[np.isfinite(img_slice)]
        vmin, vmax = (np.percentile(finite, [1, 99]) if finite.size > 0 else (0.0, 1.0))
        if vmin == vmax:
            vmax = vmin + 1.0

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(
            f"Epoch {epoch}  |  {dataset_name}  |  {sample_id}  |  "
            f"{side_label} ostium  |  σ={sigma:.3f}",
            fontsize=9,
        )

        # Panel 1: CT + GT crosshair (Thin).
        axes[0].imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[0].plot(gt_x, gt_y, 'r+', markersize=10, markeredgewidth=1, label="GT")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_title("CT — Ground Truth")
        self._set_orientation_labels(axes[0], 'L → R', 'P → A')

        # Panel 2: CT + GT crosshair + Gaussian target heatmap.
        axes[1].imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[1].imshow(
            np.ma.masked_where(gaussian_slice < 0.05, gaussian_slice),
            cmap=cmap, alpha=0.6, vmin=0.0, vmax=1.0, origin='lower',
        )
        axes[1].plot(gt_x, gt_y, 'r+', markersize=10, markeredgewidth=1)
        axes[1].set_title(f"CT — Target Heatmap (σ={sigma:.3f})")
        self._set_orientation_labels(axes[1], 'L → R', 'P → A')

        # Panel 3: CT + prediction heatmap + GT crosshair + PRED DOT.
        # Extract exact peak coordinates of prediction to draw the dot
        pred_y, pred_x = np.unravel_index(np.argmax(pred_slice), pred_slice.shape)
        
        axes[2].imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[2].imshow(
            np.ma.masked_where(pred_slice < 0.1, pred_slice),
            cmap=cmap, alpha=0.6, vmin=0.0, vmax=1.0, origin='lower',
        )
        # Cross for GT
        axes[2].plot(gt_x, gt_y, 'r+', markersize=10, markeredgewidth=1, label="GT")
        
        # Dot for Prediction (Use Cyan to make it stand out against gray CT and red/green heatmaps)
        axes[2].plot(pred_x, pred_y, marker='o', color='cyan', markersize=4, label="Pred")
        axes[2].legend(loc="upper right", fontsize=8)
        
        axes[2].set_title(f"CT — Prediction (max={pred_slice.max():.2f})")
        self._set_orientation_labels(axes[2], 'L → R', 'P → A')

        # Panel 4: S (axial) activation profile.
        max_per_z = pred_vol.max(axis=(0, 1))  # (S,)
        z_indices  = np.arange(len(max_per_z))
        color      = 'green' if side == 'R' else 'red'
        axes[3].plot(z_indices, max_per_z, color=color, linewidth=1.5)
        axes[3].axvline(x=gt_z, color='red', linestyle='--', linewidth=1.5,
                        label=f'GT S={gt_z:.0f}')
        axes[3].set_xlabel('S slice (axial, I → S)')
        axes[3].set_ylabel('Max prediction')
        axes[3].set_title('S activation profile')
        axes[3].set_ylim(0, max(max_per_z.max() * 1.1, 0.1))
        axes[3].legend(fontsize=8)
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _log_one_side(
        self,
        img_vol:        np.ndarray,    # (R, A, S) — crop or full volume
        pred_vol:       np.ndarray,    # (R, A, S)
        gt_zyx:         torch.Tensor,  # [i, j, k] = indices along (R, A, S)
        side:           str,
        sigma:          float,
        epoch:          int,
        sample_id:      str,
        dataset_name:   str,
        artifact_dir:   str,
        save_gif_frame: bool = False,
    ):
        """
        Render and log a 3-panel figure for one ostium side.
        When save_gif_frame=True the prediction panel (panel 3) is also cropped
        and saved to disk as a frame for end-of-training GIF compilation.
        """
        # Volumes in RAS (R, A, S): gt_zyx[0/1/2] = index along R / A / S.
        # Axial slice along S, transposed to (A, R) to display with origin='lower'
        # (x: L -> R, y from bottom to top: P -> A).
        gt_x = float(gt_zyx[0])  # R
        gt_y = float(gt_zyx[1])  # A
        gt_z = float(gt_zyx[2])  # S
        slice_idx = max(0, min(int(round(gt_z)), img_vol.shape[2] - 1))

        img_sl = img_vol[:, :, slice_idx].T  # (A, R)
        gauss  = self._gaussian_slice_2d(gt_y, gt_x, sigma, img_sl.shape)

        fig = self._build_panel_figure(
            img_sl, pred_vol, gt_y, gt_x, gt_z, gauss,
            side, epoch, sigma, sample_id, dataset_name,
        )

        key = f"{sample_id}_{side}"
        tmp = os.path.join(
            tempfile.gettempdir(),
            f"fviz_{key}_{epoch:03d}_{self.timestamp}.png",
        )
        fig.savefig(tmp, dpi=100)

        if save_gif_frame:
            full_img     = Image.open(tmp)
            W_img, H_img = full_img.size
            panel_w      = W_img // 4
            # Crop panel 3 (prediction) — third of four panels.
            pred_panel   = full_img.crop((2 * panel_w, 0, 3 * panel_w, H_img))
            frame_path   = os.path.join(
                self._gif_frames_dir, f"{key}_epoch_{epoch:03d}.png"
            )
            pred_panel.save(frame_path)

        mlflow.log_artifact(tmp, artifact_dir)
        plt.close(fig)
        os.remove(tmp)

    def log_fixed_train_visualizations(self, epoch: int, sigma: float):
        """
        Log 3-panel monitoring figures for all training fixed samples and save
        the prediction panel as a GIF frame for end-of-training animation.

        The trainer must populate 'pred_R' and 'pred_L' in each sample dict
        (direct forward pass on pre-extracted crops) before calling this.
        """
        if not hasattr(self, '_train_viz'):
            return
        for s in self._train_viz:
            if 'pred_R' not in s or 'pred_L' not in s:
                continue
            img_R = s['image_R'][0].float().numpy()   # (pz, py, px)
            img_L = s['image_L'][0].float().numpy()
            self._log_one_side(
                img_R, s['pred_R'].numpy(), s['gt_in_crop_R'],
                'R', sigma, epoch, s['sample_id'], s['dataset_name'],
                artifact_dir=f"fixed_train/{s['sample_id']}_R",
                save_gif_frame=True,
            )
            self._log_one_side(
                img_L, s['pred_L'].numpy(), s['gt_in_crop_L'],
                'L', sigma, epoch, s['sample_id'], s['dataset_name'],
                artifact_dir=f"fixed_train/{s['sample_id']}_L",
                save_gif_frame=True,
            )

    def log_fixed_val_visualizations(self, epoch: int, sigma: float):
        """
        Log 3-panel monitoring figures for all validation fixed samples.
        Uses the full-volume sliding-window predictions — no GIF frames saved.

        The trainer must populate 'pred_R_full' and 'pred_L_full' in each
        sample dict (via sliding-window inference) before calling this.
        """
        if not hasattr(self, '_val_viz'):
            return
        for s in self._val_viz:
            if 'pred_R_full' not in s or 'pred_L_full' not in s:
                continue
            img_vol = s['image_full'][0].float().numpy()  # (D, H, W)
            self._log_one_side(
                img_vol, s['pred_R_full'].numpy(), s['gt_R_zyx'],
                'R', sigma, epoch, s['sample_id'], s['dataset_name'],
                artifact_dir=f"fixed_val/{s['sample_id']}_R",
            )
            self._log_one_side(
                img_vol, s['pred_L_full'].numpy(), s['gt_L_zyx'],
                'L', sigma, epoch, s['sample_id'], s['dataset_name'],
                artifact_dir=f"fixed_val/{s['sample_id']}_L",
            )

    def save_prediction_gifs(self):
        """
        Compile per-volume prediction-panel GIFs from saved training frames and
        log to MLflow. One GIF per (volume, side) pair — 6 total (3 volumes × 2 sides).
        """
        if not hasattr(self, '_gif_frames_dir') or not hasattr(self, '_train_viz'):
            return
        for s in self._train_viz:
            for side in ['R', 'L']:
                key         = f"{s['sample_id']}_{side}"
                frame_paths = sorted(glob.glob(
                    os.path.join(self._gif_frames_dir, f"{key}_epoch_*.png")
                ))
                if len(frame_paths) < 2:
                    continue
                frames  = [Image.open(p) for p in frame_paths]
                gif_tmp = os.path.join(
                    tempfile.gettempdir(),
                    f"prediction_evolution_{key}_{self.timestamp}.gif",
                )
                frames[0].save(
                    gif_tmp,
                    save_all=True,
                    append_images=frames[1:],
                    duration=400,   # ms per frame
                    loop=0,
                )
                mlflow.log_artifact(gif_tmp, "prediction_gifs")
                os.remove(gif_tmp)

    def close(self):
        """Terminates the active MLflow run."""
        mlflow.end_run()
        print("[*] MLflow run closed.")


# ==============================================================================
# 2. LOCAL SMOKE TEST
# ==============================================================================
if __name__ == "__main__":
    from omegaconf import OmegaConf
    import torch.nn as nn

    print("\n[*] Testing ExperimentLogger...")

    config_mock = OmegaConf.create({
        "logger": {
            "experiment_name": "Test_Logger_System",
            "run_name": "Smoke_Test",
            "save_dir": "./test_checkpoints",
        }
    })

    logger = ExperimentLogger(config_mock)

    dummy_model     = nn.Linear(10, 2)
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)

    val_mm   = [8.0, 5.0, 4.0]
    val_mm_R = [9.0, 5.5, 4.2]
    val_mm_L = [7.0, 4.5, 3.8]

    for epoch in range(1, 4):
        print(f"\n--- Epoch {epoch} ---")
        logger.log_epoch_metrics(
            epoch=epoch,
            train_loss=val_mm[epoch - 1] * 0.1,
            learning_rate=1e-4,
        )
        logger.save_checkpoint(dummy_model, dummy_optimizer, epoch, val_mm[epoch - 1] * 0.1)

    logger.save_loss_plot()
    logger.close()

    print("\n[*] Artifact check:")
    print(f"    mlruns dir exists:      {os.path.exists('mlruns')}")
    print(f"    last_model.pth exists:  {os.path.exists('./test_checkpoints/last_model.pth')}")
    if os.path.exists("./test_checkpoints/best_model.pth"):
        print("[OK] Smoke test passed.")

