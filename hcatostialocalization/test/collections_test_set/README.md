# Test set of the paper (58 CCTA volumes)

`ImageCAS_*` (50 volumes), `ASOCA_*` (4) and `CAT08_*` (4); `all_*` is the concatenation of the three, images and labels in the
same order. **Every path is relative to your data folder** (`HCAT_DATA_ROOT`, the folder that contains `ImageCAS/`, `ASOCA/`,
`CAT08/` and `Annotations/`; see `config/machine.yaml`). To make lists with absolute paths:

```bash
export HCAT_DATA_ROOT=/path/to/your/data
sed "s#^#$HCAT_DATA_ROOT/#" hcatostialocalization/test/collections_test_set/all_images.txt > /tmp/all_images.txt
sed "s#^#$HCAT_DATA_ROOT/#" hcatostialocalization/test/collections_test_set/all_labels.txt > /tmp/all_labels.txt
```

Run the inference on them and split the metrics per dataset:

```bash
python -m hcatostialocalization.infer -i /tmp/all_images.txt -l /tmp/all_labels.txt \
    -m SwinUNETRv2_DSNT --gpu --metrics all -a marker -o results/test_set
python -m hcatostialocalization.test.split_metrics_by_dataset --metrics results/test_set/metrics__SwinUNETRv2_DSNT.json
```

The ASOCA and CAT08 reference ostia are not part of the released dataset: with only the ImageCAS annotations use the
`ImageCAS_images.txt` / `ImageCAS_labels.txt` lists.
