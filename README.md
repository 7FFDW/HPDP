# HPDP



This repository provides a PyTorch implementation of **HPDP** for whole-slide image classification. The current version focuses on the **classification task only**. After training, the script automatically saves the best checkpoint on the validation set and evaluates it on the test set.

## Overview

`train_hpdp_classification.py` supports:

* Weakly supervised MIL classification using WSI patch features;
* Spatial encoding with a fixed Sinusoidal Positional Encoder;
* LLM-generated histopathological text descriptions encoded by BioBERT;
* Prototype-based aggregation with Prior Experts and Adaptive Experts;
* Hierarchical Cross-Modal Alignment for visual-text feature refinement;
* Optional K-means teacher prototype supervision;
* Automatic testing with the saved best model;
* Output of ACC, AUC, and F1-score.

## Environment

Recommended environment:

```bash
python >= 3.8
pytorch >= 1.12
```

Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers pandas numpy scikit-learn tqdm h5py
```

If a local BioBERT model is used, please specify its path with `--text_encoder_path`.

## Data Format

The script supports two data-splitting modes.

### Option 1: Single manifest CSV

The CSV file should contain a `split` column with values from `train`, `val`, and `test`.

```csv
slide_id,feature_path,coord_path,text,label,split
case001,case001.pt,case001_coord.npy,"Tumor glands with stromal reaction.",0,train
case002,case002.pt,case002_coord.npy,"Invasive tumor with necrosis.",1,val
case003,case003.pt,case003_coord.npy,"Dense tumor cells and lymphocyte infiltration.",0,test
```

Run with:

```bash
python train_hpdp_classification.py \
  --manifest_csv data/manifest.csv \
  --feature_root data/features \
  --coord_root data/coords \
  --text_encoder_path /path/to/biobert \
  --save_dir runs/hpdp_cls \
  --n_classes 2
```

### Option 2: Separate train/val/test CSV files

```bash
python train_hpdp_classification.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --test_csv data/test.csv \
  --feature_root data/features \
  --coord_root data/coords \
  --text_encoder_path /path/to/biobert \
  --save_dir runs/hpdp_cls \
  --n_classes 2
```

Each CSV should contain at least:

```csv
slide_id,feature_path,label,text
case001,case001.pt,0,"Tumor glands with stromal reaction."
case002,case002.pt,1,"Invasive tumor with necrosis."
```

## Required CSV Fields

| Field          | Description                                           |
| -------------- | ----------------------------------------------------- |
| `feature_path` | Path to WSI patch feature file                        |
| `label`        | Slide-level class label                               |
| `split`        | Data split, required only when using `--manifest_csv` |

## Optional CSV Fields

| Field                                       | Description                                      |
| ------------------------------------------- | ------------------------------------------------ |
| `slide_id` / `case_id` / `sample_id` / `id` | Sample identifier                                |
| `coord_path`                                | Path to patch coordinate file                    |
| `text`                                      | LLM-generated histopathological description      |
| `text_path`                                 | Path to a `.txt` file containing the description |

If both `text` and `text_path` are provided, `text` is used by default.

## Feature File Format

Each WSI should have a patch feature file with shape:

```text
[N, 1024]
```

where `N` is the number of patches and `1024` is the feature dimension extracted by ResNet-50.

Supported formats:

* `.pt` / `.pth`
* `.npy`
* `.npz`
* `.h5` / `.hdf5`

### Recommended `.pt` format

```python
torch.save({
    "features": features,  # [N, 1024]
    "coords": coords       # [N, 2]
}, "case001.pt")
```

Supported feature keys:

```text
features / feature / feats / feat / x / data
```

Supported coordinate keys:

```text
coords / coord / coordinates / coordinate
```

## Coordinate Format

Patch coordinates should have shape:

```text
[N, 2]
```

Each row corresponds to the `(x, y)` location of one patch.

By default, coordinates are min-max normalized within each WSI. To disable coordinate normalization:

```bash
--no_normalize_coords
```

If coordinates are not provided, the script uses zero coordinates. However, this weakens the effect of the spatial encoding module, so real patch coordinates are recommended.

## LLM Text Descriptions

Each sample can provide an LLM-generated histopathological description through either `text` or `text_path`.

Example:

```csv
slide_id,feature_path,coord_path,text,label,split
case001,case001.pt,case001_coord.npy,"The slide shows irregular tumor glands, nuclear atypia, and stromal reaction.",1,train
```

If no text is provided, the script uses the default placeholder:

```text
No additional histopathological description is available.
```

For formal experiments, it is recommended to use a fixed prompt template and consistent API settings for all datasets.

## Training

Basic training command:

```bash
python train_hpdp_classification.py \
  --manifest_csv data/manifest.csv \
  --feature_root data/features \
  --coord_root data/coords \
  --text_encoder_path /path/to/biobert \
  --save_dir runs/hpdp_cls \
  --n_classes 2 \
  --epochs 200 \
  --early_stop 20
```

The script trains the model, saves the best checkpoint according to the validation metric, and then automatically evaluates the best model on the test set.

## K-means Teacher Prototypes

If K-means teacher prototypes are available, they can be used for prototype supervision:

```bash
--teacher_proto_path data/kmeans_centroids.pt \
--lambda_proto 0.1
```

The teacher prototype tensor should have shape:

```text
[num_prior, 512]
```

For example, when `--num_prior 4`, the expected shape is:

```text
[4, 512]
```

Supported keys in the prototype file:

```text
prototypes / prototype / centroids / kmeans_centroids / teacher
```

If no teacher prototype is used, set:

```bash
--lambda_proto 0.0
```

## Main Arguments

| Argument              | Default | Description                                 |
| --------------------- | ------: | ------------------------------------------- |
| `--n_classes`         |     `2` | Number of classes                           |
| `--epochs`            |   `200` | Maximum number of training epochs           |
| `--early_stop`        |    `20` | Early stopping patience                     |
| `--lr`                |  `3e-4` | Initial learning rate                       |
| `--min_lr`            |  `1e-4` | Minimum learning rate for scheduler         |
| `--weight_decay`      |  `1e-5` | Weight decay                                |
| `--dropout`           |  `0.25` | Dropout rate                                |
| `--num_prior`         |     `4` | Number of Prior Experts                     |
| `--num_adaptive`      |    `12` | Number of Adaptive Experts                  |
| `--lambda_proto`      |   `0.0` | Weight of prototype supervision loss        |
| `--proto_temperature` |   `0.1` | Temperature for prototype alignment         |
| `--metric_for_best`   |   `auc` | Validation metric for saving the best model |
| `--amp`               |   False | Enable automatic mixed precision            |
| `--num_workers`       |     `0` | Number of DataLoader workers                |
| `--seed`              |  `2026` | Random seed                                 |

## Outputs

All outputs are saved to `--save_dir`.

```text
runs/hpdp_cls/
├── best_model.pt
├── last_model.pt
├── test_predictions.csv
├── metrics.json
├── train_log.csv
├── label_mapping.json
└── args.json
```

| File                   | Description                                              |
| ---------------------- | -------------------------------------------------------- |
| `best_model.pt`        | Best model checkpoint selected by validation performance |
| `last_model.pt`        | Last epoch checkpoint                                    |
| `test_predictions.csv` | Slide-level test predictions                             |
| `metrics.json`         | Validation and test metrics                              |
| `train_log.csv`        | Training log of each epoch                               |
| `label_mapping.json`   | Label-to-index mapping                                   |
| `args.json`            | Running configuration                                    |

## Test Prediction File

After training, the script automatically evaluates the best model and saves:

```csv
slide_id,label,pred,prob_0,prob_1
case003,0,0,0.8123,0.1877
case004,1,1,0.2311,0.7689
```

| Field                   | Description                         |
| ----------------------- | ----------------------------------- |
| `slide_id`              | Sample ID                           |
| `label`                 | Ground-truth label                  |
| `pred`                  | Predicted label                     |
| `prob_0`, `prob_1`, ... | Predicted probability of each class |

## Metrics

The script reports:

| Metric | Description                                                                      |
| ------ | -------------------------------------------------------------------------------- |
| `ACC`  | Classification accuracy                                                          |
| `AUC`  | ROC-AUC, computed using the positive-class probability for binary classification |
| `F1`   | Binary F1 for binary classification and macro F1 for multi-class classification  |

If only one class appears in the validation or test set, AUC cannot be computed and will be returned as `nan`.

## Example Project Structure

```text
HPDP/
├── train_hpdp_classification.py
├── README.md
├── data/
│   ├── manifest.csv
│   ├── features/
│   │   ├── case001.pt
│   │   ├── case002.pt
│   │   └── case003.pt
│   ├── coords/
│   │   ├── case001_coord.npy
│   │   ├── case002_coord.npy
│   │   └── case003_coord.npy
│   └── kmeans_centroids.pt
└── runs/
    └── hpdp_cls/
        ├── best_model.pt
        ├── last_model.pt
        ├── test_predictions.csv
        ├── metrics.json
        ├── train_log.csv
        ├── label_mapping.json
        └── args.json
```

## Notes

1. This script only supports classification and does not include survival analysis.
2. Each batch contains one WSI because different slides usually have different numbers of patches.
3. The default input feature dimension is `1024`, corresponding to ResNet-50 features.
4. If features from other encoders such as CTransPath or UNI are used, the input dimension of `visual_feature_extractor` should be modified accordingly.
5. Real patch coordinates are recommended for enabling the spatial encoding module.
6. If GPU memory is insufficient, consider reducing the number of patches per WSI or preselecting tissue patches.

## Reproducibility

For reproducibility, please keep the following files and settings unchanged:

* Data split CSV files;
* Patch feature extraction settings;
* Patch coordinate generation settings;
* K-means clustering settings;
* LLM prompt and API settings;
* Random seed;
* Training hyperparameters.

Example:

```bash
--seed 2026
```

## Citation

If this code is useful for your research, please cite our paper:

```bibtex
@article{hpdp,
  title={Hierarchical Prototype-based Domain Priors for Multiple Instance Learning in Multimodal Histopathology Analysis},
  author={Qiu, Xuemei and Fan, Dawei and Huang, Yebin and Chen, Yanping and Wei, Lifang},
  journal={},
  year={}
}
```
