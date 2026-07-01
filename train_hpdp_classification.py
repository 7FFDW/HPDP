"""
train_hpdp_classification.py

Classification-only training and testing script for HPDP.

Expected CSV format
-------------------
Use either one manifest CSV with a `split` column, or three separate CSV files.
Each sample row should contain at least:
    - label: class label, integer or string
    - feature_path: path to a feature file
Optional columns:
    - coord_path: path to coordinate file, if coords are not stored together with features
    - text: LLM-generated morphology description
    - text_path: path to a .txt file containing the description
    - slide_id / case_id / id: sample identifier
    - split: train / val / test, when using --manifest_csv

Supported feature formats
-------------------------
1) .pt/.pth: tensor or dict. Dict keys can include:
       features / feature / feats / x / data
       coords / coord / coordinates
2) .npz: arrays with keys features/feats/x and coords/coordinates
3) .npy: feature array only, use --coord_path/coord_path for coords
4) .h5/.hdf5: datasets with keys features/feats/x and coords/coordinates

Example
-------
python train_hpdp_classification.py \
    --manifest_csv data/manifest.csv \
    --feature_root data/features \
    --text_encoder_path /path/to/biobert \
    --save_dir runs/hpdp_cls \
    --n_classes 2 \
    --epochs 200 \
    --early_stop 20

After training, the script automatically loads the best validation checkpoint
and evaluates it on the test set. It saves:
    - best_model.pt
    - last_model.pt
    - test_predictions.csv
    - metrics.json
    - label_mapping.json
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import h5py
except ImportError:
    h5py = None

try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
except ImportError:
    accuracy_score = None
    f1_score = None
    roc_auc_score = None

from transformers import AutoModel, AutoTokenizer


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------------
# Model definition
# -----------------------------

class TextEncoder(nn.Module):
    def __init__(self, output_dim: int = 512, model_name: str = "dmis-lab/biobert-base-cased-v1.1"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # The paper uses a frozen BioBERT encoder. Only the projection layer is trainable.
        for p in self.bert.parameters():
            p.requires_grad = False

        bert_output_dim = self.bert.config.hidden_size
        self.projection = nn.Linear(bert_output_dim, output_dim)

    def forward(self, text_list: List[str], device: torch.device) -> torch.Tensor:
        inputs = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = self.bert(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        projected_embedding = self.projection(cls_embedding)
        return projected_embedding


class FilmFusion(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.param_generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
        )
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, patch_features: torch.Tensor, text_feature: torch.Tensor) -> torch.Tensor:
        params = self.param_generator(text_feature)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        fused_features = gamma * patch_features + beta
        final_features = self.ln(patch_features + fused_features)
        return final_features


class MoEP(nn.Module):
    def __init__(
        self,
        num_prior_prototypes: int,
        num_adaptive_prototypes: int,
        embed_dim: int = 512,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.prior_prototypes = nn.Parameter(torch.randn(num_prior_prototypes, embed_dim))
        self.adaptive_prototypes = nn.Parameter(torch.randn(num_adaptive_prototypes, embed_dim))
        self.temperature = temperature

    def forward(self, slide_features: torch.Tensor) -> torch.Tensor:
        slide_features_norm = F.normalize(slide_features, dim=1)
        prior_prototypes_norm = F.normalize(self.prior_prototypes, dim=1)
        cosine_sim_prior = torch.matmul(slide_features_norm, prior_prototypes_norm.T)
        routing_weights = F.softmax(cosine_sim_prior / self.temperature, dim=1)

        interaction_scores = torch.matmul(slide_features, self.adaptive_prototypes.T)
        transformation_weights = F.softmax(interaction_scores / self.temperature, dim=1)

        all_attn_weights = torch.cat([routing_weights, transformation_weights], dim=1)
        aggregated_prototypes = torch.matmul(all_attn_weights.T, slide_features)
        return aggregated_prototypes


class HierarchicalFusion(nn.Module):
    def __init__(self, embed_dim: int = 512, num_supervised_prototypes: int = 4, num_free_prototypes: int = 12):
        super().__init__()
        self.num_supervised = num_supervised_prototypes
        self.num_free = num_free_prototypes
        self.num_prototypes = num_supervised_prototypes + num_free_prototypes
        self.fusion_module = FilmFusion(embed_dim)
        self.propagation_attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, aggregated_prototypes: torch.Tensor, text_feature: torch.Tensor) -> torch.Tensor:
        fused_prototypes = self.fusion_module(aggregated_prototypes, text_feature)
        slide_features_b = aggregated_prototypes.unsqueeze(0)
        fused_prototypes_b = fused_prototypes.unsqueeze(0)
        propagated_info, _ = self.propagation_attn(
            query=slide_features_b,
            key=fused_prototypes_b,
            value=fused_prototypes_b,
        )
        final_features = self.ln(slide_features_b + propagated_info)
        return final_features.squeeze(0)


class FixedSinusoidalEncoder(nn.Module):
    def __init__(self, output_dim: int = 512):
        super().__init__()
        if output_dim % 4 != 0:
            raise ValueError(f"output_dim must be divisible by 4, but got {output_dim}")
        self.output_dim = output_dim
        half_dim = output_dim // 2
        div_term = torch.exp(torch.arange(0, half_dim, 2).float() * (-math.log(10000.0) / half_dim))
        self.register_buffer("div_term", div_term)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        div_term_on_device = self.div_term.to(coords.device)
        pe_x = coords[:, 0:1] * div_term_on_device
        pe_y = coords[:, 1:2] * div_term_on_device
        sin_x = torch.sin(pe_x)
        cos_x = torch.cos(pe_x)
        sin_y = torch.sin(pe_y)
        cos_y = torch.cos(pe_y)
        return torch.cat([sin_x, cos_x, sin_y, cos_y], dim=1)


class HPDP(nn.Module):
    def __init__(
        self,
        n_classes: int = 2,
        dropout: float = 0.25,
        num_supervised_prototypes: int = 4,
        num_free_prototypes: int = 12,
        text_encoder_path: str = "dmis-lab/biobert-base-cased-v1.1",
    ):
        super().__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        self.visual_feature_extractor = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.LayerNorm(1024),
            nn.Linear(1024, self.L),
        )

        self.pos_encoder = FixedSinusoidalEncoder(output_dim=self.L)
        self.text_encoder = TextEncoder(output_dim=self.L, model_name=text_encoder_path)
        self.moep = MoEP(num_supervised_prototypes, num_free_prototypes, embed_dim=self.L)
        self.fusion_module = HierarchicalFusion(
            embed_dim=self.L,
            num_supervised_prototypes=num_supervised_prototypes,
            num_free_prototypes=num_free_prototypes,
        )

        self.attention_a = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh(), nn.Dropout(dropout))
        self.attention_b = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_c = nn.Linear(self.D, self.K)
        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, n_classes))

    def forward(self, x: torch.Tensor, coords: torch.Tensor, text: List[str]):
        device = x.device
        initial_instance_features = self.visual_feature_extractor(x.squeeze(0))
        positional_embedding = self.pos_encoder(coords.squeeze(0))
        slide_features = initial_instance_features + positional_embedding

        global_text_feature = self.text_encoder(text, device)
        aggregated_prototypes = self.moep(slide_features)
        multimodal_features = self.fusion_module(aggregated_prototypes, global_text_feature)

        a = self.attention_a(multimodal_features)
        b = self.attention_b(multimodal_features)
        A = a.mul(b)
        A = self.attention_c(A)
        A = torch.transpose(A, 0, 1)
        A = F.softmax(A, dim=1)
        bag_feature = torch.mm(A, multimodal_features)

        logits = self.classifier(bag_feature)
        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.topk(logits, 1, dim=1)[1]
        return logits, y_prob, y_hat, A, bag_feature.mean(dim=0, keepdim=True)


# -----------------------------
# Data utilities
# -----------------------------

FEATURE_KEYS = ["features", "feature", "feats", "feat", "x", "data"]
COORD_KEYS = ["coords", "coord", "coordinates", "coordinate"]
ID_COLUMNS = ["slide_id", "case_id", "sample_id", "id", "ID", "name"]


def first_existing_key(obj: Any, keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in obj:
            return k
    return None


def resolve_path(path_value: Any, root: Optional[str]) -> Optional[str]:
    if path_value is None or (isinstance(path_value, float) and np.isnan(path_value)):
        return None
    path_str = str(path_value)
    if path_str == "":
        return None
    p = Path(path_str)
    if p.is_absolute() or root is None or root == "":
        return str(p)
    return str(Path(root) / p)


def load_tensor_file(path: str) -> Any:
    suffix = Path(path).suffix.lower()
    if suffix in [".pt", ".pth"]:
        return torch.load(path, map_location="cpu")
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".npz":
        return np.load(path, allow_pickle=True)
    if suffix in [".h5", ".hdf5"]:
        if h5py is None:
            raise ImportError("h5py is required to read .h5/.hdf5 feature files. Please `pip install h5py`.")
        with h5py.File(path, "r") as f:
            out = {}
            for k in list(f.keys()):
                out[k] = f[k][()]
            return out
    raise ValueError(f"Unsupported feature file format: {path}")


def to_float_tensor(arr: Any) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr.float()
    return torch.as_tensor(arr, dtype=torch.float32)


def extract_features_and_coords(feature_path: str, coord_path: Optional[str] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    obj = load_tensor_file(feature_path)
    coords = None

    if isinstance(obj, dict) or hasattr(obj, "keys"):
        f_key = first_existing_key(obj, FEATURE_KEYS)
        if f_key is None:
            raise KeyError(f"No feature key found in {feature_path}. Expected one of {FEATURE_KEYS}.")
        features = to_float_tensor(obj[f_key])
        c_key = first_existing_key(obj, COORD_KEYS)
        if c_key is not None:
            coords = to_float_tensor(obj[c_key])
    else:
        features = to_float_tensor(obj)

    if coord_path is not None:
        coord_obj = load_tensor_file(coord_path)
        if isinstance(coord_obj, dict) or hasattr(coord_obj, "keys"):
            c_key = first_existing_key(coord_obj, COORD_KEYS + FEATURE_KEYS)
            if c_key is None:
                raise KeyError(f"No coordinate key found in {coord_path}.")
            coords = to_float_tensor(coord_obj[c_key])
        else:
            coords = to_float_tensor(coord_obj)

    return features, coords


def normalize_coordinates(coords: torch.Tensor) -> torch.Tensor:
    coords = coords.float()
    if coords.numel() == 0:
        return coords
    min_xy = coords.min(dim=0, keepdim=True).values
    max_xy = coords.max(dim=0, keepdim=True).values
    denom = (max_xy - min_xy).clamp_min(1e-6)
    return (coords - min_xy) / denom


def read_text_from_row(row: pd.Series, text_root: Optional[str]) -> str:
    if "text" in row and not pd.isna(row["text"]):
        return str(row["text"])
    if "description" in row and not pd.isna(row["description"]):
        return str(row["description"])
    if "text_path" in row and not pd.isna(row["text_path"]):
        tpath = resolve_path(row["text_path"], text_root)
        if tpath is not None and os.path.exists(tpath):
            with open(tpath, "r", encoding="utf-8") as f:
                return f.read().strip()
    return "No additional histopathological description is available."


@dataclass
class SampleRecord:
    slide_id: str
    feature_path: str
    coord_path: Optional[str]
    text: str
    label: int


class WSIFeatureDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_to_index: Dict[str, int],
        feature_root: Optional[str] = None,
        coord_root: Optional[str] = None,
        text_root: Optional[str] = None,
        normalize_coords: bool = True,
    ):
        self.records: List[SampleRecord] = []
        self.normalize_coords = normalize_coords

        if "feature_path" not in dataframe.columns:
            raise ValueError("CSV must contain a `feature_path` column.")
        if "label" not in dataframe.columns:
            raise ValueError("CSV must contain a `label` column.")

        for idx, row in dataframe.reset_index(drop=True).iterrows():
            slide_id = None
            for c in ID_COLUMNS:
                if c in row and not pd.isna(row[c]):
                    slide_id = str(row[c])
                    break
            if slide_id is None:
                slide_id = f"sample_{idx}"

            fpath = resolve_path(row["feature_path"], feature_root)
            cpath = None
            if "coord_path" in row and not pd.isna(row["coord_path"]):
                cpath = resolve_path(row["coord_path"], coord_root)

            text = read_text_from_row(row, text_root)
            raw_label = str(row["label"])
            label = label_to_index[raw_label]
            self.records.append(SampleRecord(slide_id, fpath, cpath, text, label))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self.records[index]
        if not os.path.exists(rec.feature_path):
            raise FileNotFoundError(f"Feature file not found: {rec.feature_path}")

        features, coords = extract_features_and_coords(rec.feature_path, rec.coord_path)
        if features.dim() != 2:
            raise ValueError(f"Expected features with shape [N, 1024], got {tuple(features.shape)} for {rec.feature_path}")
        if features.size(1) != 1024:
            raise ValueError(f"Expected feature dimension 1024, got {features.size(1)} for {rec.feature_path}")

        if coords is None:
            # Fallback: no coordinates are available. This keeps code runnable, but true HPDP should use WSI coordinates.
            coords = torch.zeros(features.size(0), 2, dtype=torch.float32)
        if coords.dim() != 2 or coords.size(1) < 2:
            raise ValueError(f"Expected coords with shape [N, 2], got {tuple(coords.shape)} for {rec.feature_path}")
        coords = coords[:, :2].float()
        if coords.size(0) != features.size(0):
            raise ValueError(
                f"Feature/coord length mismatch for {rec.feature_path}: "
                f"features={features.size(0)}, coords={coords.size(0)}"
            )
        if self.normalize_coords:
            coords = normalize_coordinates(coords)

        return {
            "features": features.unsqueeze(0),  # [1, N, 1024]
            "coords": coords.unsqueeze(0),      # [1, N, 2]
            "text": rec.text,
            "label": torch.tensor(rec.label, dtype=torch.long),
            "slide_id": rec.slide_id,
        }


def collate_slide(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # MIL training uses one WSI per step because each slide has a different number of patches.
    if len(batch) != 1:
        raise ValueError("Please use batch_size=1 because each WSI has a variable number of patches.")
    return batch[0]


# -----------------------------
# CSV loading and label mapping
# -----------------------------

def load_dataframes(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.manifest_csv:
        df = pd.read_csv(args.manifest_csv)
        if "split" not in df.columns:
            raise ValueError("When using --manifest_csv, the CSV must contain a `split` column with train/val/test.")
        split = df["split"].astype(str).str.lower()
        train_df = df[split == "train"].copy()
        val_df = df[split.isin(["val", "valid", "validation"])].copy()
        test_df = df[split == "test"].copy()
    else:
        if not (args.train_csv and args.val_csv and args.test_csv):
            raise ValueError("Provide either --manifest_csv or all of --train_csv, --val_csv, and --test_csv.")
        train_df = pd.read_csv(args.train_csv)
        val_df = pd.read_csv(args.val_csv)
        test_df = pd.read_csv(args.test_csv)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(f"Empty split detected: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def build_label_mapping(*dfs: pd.DataFrame) -> Dict[str, int]:
    labels = []
    for df in dfs:
        if "label" not in df.columns:
            raise ValueError("All CSV files must contain a `label` column.")
        labels.extend([str(x) for x in df["label"].tolist()])

    unique = sorted(set(labels))
    # If labels are numeric-like, sort numerically.
    try:
        unique = [str(x) for x in sorted([int(float(u)) for u in unique])]
    except Exception:
        pass
    return {lab: i for i, lab in enumerate(unique)}


# -----------------------------
# Losses and metrics
# -----------------------------

def load_teacher_prototypes(path: Optional[str]) -> Optional[torch.Tensor]:
    if path is None or path == "":
        return None
    obj = load_tensor_file(path)
    if isinstance(obj, dict) or hasattr(obj, "keys"):
        key = first_existing_key(obj, ["prototypes", "prototype", "centroids", "kmeans_centroids", "teacher"])
        if key is None:
            raise KeyError(f"No teacher prototype key found in {path}.")
        obj = obj[key]
    proto = to_float_tensor(obj)
    if proto.dim() != 2:
        raise ValueError(f"Teacher prototypes should have shape [K, D], got {tuple(proto.shape)}")
    return proto


def prototype_alignment_loss(model: HPDP, teacher_proto: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    prior = model.moep.prior_prototypes
    teacher_proto = teacher_proto.to(prior.device)
    if teacher_proto.size() != prior.size():
        raise ValueError(
            f"Teacher prototypes shape {tuple(teacher_proto.shape)} does not match "
            f"model prior prototypes shape {tuple(prior.shape)}."
        )
    prior_norm = F.normalize(prior, dim=1)
    teacher_norm = F.normalize(teacher_proto, dim=1)
    logits = torch.matmul(prior_norm, teacher_norm.T) / temperature
    target = torch.arange(prior.size(0), device=prior.device)
    return F.cross_entropy(logits, target)


def safe_metrics(y_true: List[int], y_pred: List[int], y_prob: np.ndarray, n_classes: int) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"acc": float("nan"), "auc": float("nan"), "f1": float("nan")}

    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    if accuracy_score is not None:
        acc = float(accuracy_score(y_true_np, y_pred_np))
    else:
        acc = float((y_true_np == y_pred_np).mean())

    if f1_score is not None:
        avg = "binary" if n_classes == 2 else "macro"
        try:
            f1 = float(f1_score(y_true_np, y_pred_np, average=avg, zero_division=0))
        except Exception:
            f1 = float("nan")
    else:
        f1 = float("nan")

    auc = float("nan")
    if roc_auc_score is not None:
        try:
            if n_classes == 2:
                auc = float(roc_auc_score(y_true_np, y_prob[:, 1]))
            else:
                auc = float(roc_auc_score(y_true_np, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            auc = float("nan")

    return {"acc": acc, "auc": auc, "f1": f1}


# -----------------------------
# Train / eval loops
# -----------------------------

def forward_one_batch(model: HPDP, batch: Dict[str, Any], device: torch.device):
    features = batch["features"].to(device, non_blocking=True)
    coords = batch["coords"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True).view(1)
    text = [batch["text"]]
    logits, y_prob, y_hat, attn, bag_feature = model(features, coords, text)
    return logits, y_prob, y_hat, label, attn, bag_feature


def run_epoch(
    model: HPDP,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    teacher_proto: Optional[torch.Tensor] = None,
    lambda_proto: float = 0.0,
    proto_temperature: float = 0.1,
    n_classes: int = 2,
    amp: bool = False,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_probs: List[np.ndarray] = []
    pred_rows: List[Dict[str, Any]] = []

    pbar = tqdm(loader, total=len(loader), leave=False, desc="train" if is_train else "eval")
    for batch in pbar:
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=amp):
                logits, prob, pred, label, _, _ = forward_one_batch(model, batch, device)
                loss = criterion(logits, label)
                if teacher_proto is not None and lambda_proto > 0:
                    loss = loss + lambda_proto * prototype_alignment_loss(model, teacher_proto, proto_temperature)

            if is_train:
                if scaler is not None and amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        prob_np = prob.detach().cpu().numpy()[0]
        pred_int = int(pred.detach().cpu().view(-1)[0].item())
        label_int = int(label.detach().cpu().view(-1)[0].item())

        y_true.append(label_int)
        y_pred.append(pred_int)
        y_probs.append(prob_np)
        row = {
            "slide_id": batch["slide_id"],
            "label": label_int,
            "pred": pred_int,
        }
        for c in range(n_classes):
            row[f"prob_{c}"] = float(prob_np[c])
        pred_rows.append(row)

        pbar.set_postfix(loss=f"{total_loss / max(1, len(y_true)):.4f}")

    avg_loss = total_loss / max(1, len(loader))
    y_prob_arr = np.vstack(y_probs) if y_probs else np.zeros((0, n_classes))
    metrics = safe_metrics(y_true, y_pred, y_prob_arr, n_classes)
    metrics["loss"] = float(avg_loss)
    return avg_loss, metrics, pd.DataFrame(pred_rows)


def save_checkpoint(
    path: str,
    model: HPDP,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
    label_mapping: Dict[str, int],
) -> None:
    ckpt = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
        "label_mapping": label_mapping,
    }
    torch.save(ckpt, path)


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test HPDP for WSI classification.")

    # Data
    parser.add_argument("--manifest_csv", type=str, default=None, help="CSV with columns feature_path,label,text/split.")
    parser.add_argument("--train_csv", type=str, default=None, help="Training CSV if not using manifest_csv.")
    parser.add_argument("--val_csv", type=str, default=None, help="Validation CSV if not using manifest_csv.")
    parser.add_argument("--test_csv", type=str, default=None, help="Test CSV if not using manifest_csv.")
    parser.add_argument("--feature_root", type=str, default=None, help="Root directory for relative feature_path values.")
    parser.add_argument("--coord_root", type=str, default=None, help="Root directory for relative coord_path values.")
    parser.add_argument("--text_root", type=str, default=None, help="Root directory for relative text_path values.")
    parser.add_argument("--no_normalize_coords", action="store_true", help="Disable per-slide min-max coordinate normalization.")

    # Model
    parser.add_argument("--text_encoder_path", type=str, default="dmis-lab/biobert-base-cased-v1.1",
                        help="Local or HuggingFace path of BioBERT/text encoder.")
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--num_prior", type=int, default=4, help="Number of Prior Experts, K_prior.")
    parser.add_argument("--num_adaptive", type=int, default=12, help="Number of Adaptive Experts.")
    parser.add_argument("--teacher_proto_path", type=str, default=None,
                        help="Optional K-means teacher prototypes with shape [num_prior, 512].")
    parser.add_argument("--lambda_proto", type=float, default=0.0,
                        help="Weight of prototype alignment loss. Set to 0 if teacher prototypes are unavailable.")
    parser.add_argument("--proto_temperature", type=float, default=0.1)

    # Optimization, following the paper's classification setting as default.
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision.")

    # Output
    parser.add_argument("--save_dir", type=str, default="runs/hpdp_classification")
    parser.add_argument("--metric_for_best", type=str, default="auc", choices=["auc", "f1", "acc", "loss"])
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    train_df, val_df, test_df = load_dataframes(args)
    label_mapping = build_label_mapping(train_df, val_df, test_df)
    if args.n_classes != len(label_mapping):
        print(
            f"[Warning] --n_classes={args.n_classes}, but CSV contains {len(label_mapping)} unique labels. "
            f"Using n_classes={len(label_mapping)}."
        )
        args.n_classes = len(label_mapping)

    with open(save_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    with open(save_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    train_set = WSIFeatureDataset(
        train_df,
        label_to_index=label_mapping,
        feature_root=args.feature_root,
        coord_root=args.coord_root,
        text_root=args.text_root,
        normalize_coords=not args.no_normalize_coords,
    )
    val_set = WSIFeatureDataset(
        val_df,
        label_to_index=label_mapping,
        feature_root=args.feature_root,
        coord_root=args.coord_root,
        text_root=args.text_root,
        normalize_coords=not args.no_normalize_coords,
    )
    test_set = WSIFeatureDataset(
        test_df,
        label_to_index=label_mapping,
        feature_root=args.feature_root,
        coord_root=args.coord_root,
        text_root=args.text_root,
        normalize_coords=not args.no_normalize_coords,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_slide,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_slide,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_slide,
    )

    model = HPDP(
        n_classes=args.n_classes,
        dropout=args.dropout,
        num_supervised_prototypes=args.num_prior,
        num_free_prototypes=args.num_adaptive,
        text_encoder_path=args.text_encoder_path,
    ).to(device)

    teacher_proto = load_teacher_prototypes(args.teacher_proto_path)
    if teacher_proto is not None:
        print(f"Loaded teacher prototypes: {tuple(teacher_proto.shape)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_metric = float("inf") if args.metric_for_best == "loss" else -float("inf")
    best_epoch = -1
    patience = 0
    history: List[Dict[str, Any]] = []

    print(f"Train/Val/Test: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    print(f"Label mapping: {label_mapping}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _ = run_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer=optimizer,
            scaler=scaler,
            teacher_proto=teacher_proto,
            lambda_proto=args.lambda_proto,
            proto_temperature=args.proto_temperature,
            n_classes=args.n_classes,
            amp=args.amp,
        )
        val_loss, val_metrics, _ = run_epoch(
            model,
            val_loader,
            device,
            criterion,
            optimizer=None,
            teacher_proto=teacher_proto,
            lambda_proto=args.lambda_proto,
            proto_temperature=args.proto_temperature,
            n_classes=args.n_classes,
            amp=False,
        )
        scheduler.step()

        current = val_metrics[args.metric_for_best]
        if args.metric_for_best == "loss":
            improved = current < best_metric
        else:
            improved = current > best_metric

        if improved:
            best_metric = current
            best_epoch = epoch
            patience = 0
            save_checkpoint(
                str(save_dir / "best_model.pt"),
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                args,
                label_mapping,
            )
        else:
            patience += 1

        save_checkpoint(
            str(save_dir / "last_model.pt"),
            model,
            optimizer,
            scheduler,
            epoch,
            best_metric,
            args,
            label_mapping,
        )

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_auc": train_metrics["auc"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_auc": val_metrics["auc"],
            "val_f1": val_metrics["f1"],
            "best_metric": best_metric,
            "best_epoch": best_epoch,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(save_dir / "train_log.csv", index=False)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} "
            f"auc {train_metrics['auc']:.4f} f1 {train_metrics['f1']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} "
            f"auc {val_metrics['auc']:.4f} f1 {val_metrics['f1']:.4f} | "
            f"best {args.metric_for_best}={best_metric:.4f} @ epoch {best_epoch}"
        )

        if patience >= args.early_stop:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    # Test with best checkpoint
    best_path = save_dir / "best_model.pt"
    if not best_path.exists():
        raise FileNotFoundError("No best_model.pt was saved. Please check validation metrics and data.")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_metrics, test_pred = run_epoch(
        model,
        test_loader,
        device,
        criterion,
        optimizer=None,
        teacher_proto=teacher_proto,
        lambda_proto=args.lambda_proto,
        proto_temperature=args.proto_temperature,
        n_classes=args.n_classes,
        amp=False,
    )

    test_pred.to_csv(save_dir / "test_predictions.csv", index=False)
    all_metrics = {
        "best_epoch": int(best_epoch),
        "best_val_metric": float(best_metric),
        "test": test_metrics,
        "label_mapping": label_mapping,
    }
    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print("\nFinal test results using best validation checkpoint:")
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved outputs to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
