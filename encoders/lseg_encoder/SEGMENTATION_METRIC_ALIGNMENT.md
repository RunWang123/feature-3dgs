# Segmentation Metrics Alignment with LSM

This document verifies that `segmentation_metric_gt.py` uses **exactly the same logic** as the LSM project for semantic segmentation evaluation.

## Line-by-Line Correspondence

### 1. Label Remapping Function

**LSM:** `large_spatial_model/datasets/testdata.py:12-22`
```python
def map_func(label_path, labels=['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other']):
    labels = [label.lower() for label in labels]

    df = pd.read_csv(label_path, sep='\t')
    id_to_nyu40class = pd.Series(df['nyu40class'].str.lower().values, index=df['id']).to_dict()

    nyu40class_to_newid = {cls: labels.index(cls) + 1 if cls in labels else labels.index('other') + 1 for cls in set(id_to_nyu40class.values())}

    id_to_newid = {id_: nyu40class_to_newid[cls] for id_, cls in id_to_nyu40class.items()}

    return np.vectorize(lambda x: id_to_newid.get(x, labels.index('other') + 1) if x != 0 else 0)
```

**Our Script:** `segmentation_metric_gt.py:41-65`
```python
def create_label_remapping(label_path, target_labels):
    target_labels = [label.lower() for label in target_labels]
    
    # Read the label mapping file
    df = pd.read_csv(label_path, sep='\t')
    id_to_nyu40class = pd.Series(df['nyu40class'].str.lower().values, index=df['id']).to_dict()
    
    # Map nyu40 classes to new IDs (1-indexed, 0 is background)
    nyu40class_to_newid = {
        cls: target_labels.index(cls) + 1 if cls in target_labels else target_labels.index('other') + 1 
        for cls in set(id_to_nyu40class.values())
    }
    
    # Create final ID mapping
    id_to_newid = {id_: nyu40class_to_newid[cls] for id_, cls in id_to_nyu40class.items()}
    
    # Vectorized remapping function (0 stays 0 for background/unlabeled)
    return np.vectorize(lambda x: id_to_newid.get(x, target_labels.index('other') + 1) if x != 0 else 0)
```

✅ **IDENTICAL LOGIC**

---

### 2. Feature Decoding Logic

**LSM:** `large_spatial_model/lseg.py:91-98`
```python
image_features = image_features.permute(0,2,3,1).reshape(-1, self.out_c)

# normalized features
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

logits_per_image = self.logit_scale * image_features.half() @ text_features.t()
out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)
```

**Our Script:** `segmentation_metric_gt.py:237-247`
```python
# Reshape: (C, H, W) -> (H, W, C) -> (H*W, C)
image_features = feature.permute(1, 2, 0).reshape(-1, c)

# Normalize features (EXACT same as LSM)
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

# Compute logits with scale (EXACT same as LSM)
logits_per_image = logit_scale * image_features.half() @ text_features_norm.t()

# Reshape back: (H*W, num_labels) -> (H, W, num_labels) -> (num_labels, H, W)
logits = logits_per_image.float().view(h, w, -1).permute(2, 0, 1)
```

✅ **IDENTICAL LOGIC** (adjusted for single image instead of batch)

---

### 3. Semantic Map Generation

**LSM:** `large_spatial_model/utils/visualization_utils.py:302-303`
```python
logits = model.lseg_feature_extractor.decode_feature(feature_map, labelset=LABELS)
semantic_map = torch.argmax(logits, dim=1) + 1
```

**Our Script:** `segmentation_metric_gt.py:250`
```python
# Get semantic map (EXACT same as LSM: visualization_utils.py line 303)
pred_class = torch.argmax(logits, dim=0) + 1  # +1 to make 1-indexed
```

✅ **IDENTICAL LOGIC** (adjusted for single image instead of batch)

---

### 4. Metrics Initialization

**LSM:** `large_spatial_model/loss.py:120-121`
```python
self.miou = JaccardIndex(num_classes=len(self.labels) + 1, task='multiclass', ignore_index=0)
self.accuracy = Accuracy(num_classes=len(self.labels) + 1, task='multiclass', ignore_index=0)
```

**Our Script:** `segmentation_metric_gt.py:178-179`
```python
miou_metric = JaccardIndex(num_classes=num_classes, task='multiclass', ignore_index=0).to(device)
accuracy_metric = Accuracy(num_classes=num_classes, task='multiclass', ignore_index=0).to(device)
```

✅ **IDENTICAL PARAMETERS**

---

### 5. Logit Scale

**LSM:** Uses `self.logit_scale` which is initialized in LSeg model
```python
# From submodules/lang_seg/modules/models/lseg_net.py
self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
```

**Our Script:** `segmentation_metric_gt.py:167`
```python
logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp().cuda()
```

✅ **IDENTICAL VALUE**

---

## Key Differences from Old Script

### Old Script (`segmentation_metric.py`)
- ❌ Compared prediction vs LSeg teacher features
- ❌ Did not use ground truth labels
- ❌ Not a true evaluation

### New Script (`segmentation_metric_gt.py`)
- ✅ Compares prediction vs ground truth labels
- ✅ Uses LSM's exact label remapping
- ✅ Uses LSM's exact feature decoding
- ✅ Uses LSM's exact semantic map generation
- ✅ True evaluation against GT

---

## Verification Checklist

- [x] Label remapping matches LSM's `map_func`
- [x] Feature normalization matches LSM
- [x] Logit computation uses `logit_scale * features @ text.t()`
- [x] Uses `.half()` for image features (same as LSM)
- [x] Uses `.float()` for final logits (same as LSM)
- [x] Semantic map uses `argmax + 1` (same as LSM)
- [x] Metrics use `ignore_index=0` for background
- [x] Number of classes is `len(labels) + 1` (same as LSM)

---

## Usage Example

```bash
cd /home/runw/project/feature-3dgs/encoders/lseg_encoder

python -u segmentation_metric_gt.py \
  --data /scratch/.../scene0686_01_case0 \
  --scene_data_path /scratch/.../scannet_test_feature3dgs/scene0686_01 \
  --iteration 7000 \
  --label_src "wall,floor,ceiling,chair,table,sofa,bed,other" \
  --label_mapping_file /home/runw/Project/data/colmap/data/scannet_test/scannetv2-labels.combined.tsv \
  --backbone clip_vitl16_384 \
  --weights demo_e200.ckpt
```

---

## Expected Output

```
============================================================
Results for test split:
============================================================
mIoU:     0.XXXX
Accuracy: 0.XXXX
============================================================

============================================================
Results for train split:
============================================================
mIoU:     0.XXXX
Accuracy: 0.XXXX
============================================================
```

These metrics are computed using **exactly the same methodology** as LSM, ensuring fair comparison.

