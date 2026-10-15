# HCAT analysis viewer

Analyze existing ostia predictions without running inference.

The input is a CSV/TSV file. Nothing is mandatory: the script computes only what
is possible from the available columns.

Common one-row-per-ostium format:

```csv
case_id,dataset,side,image,label,prediction,heatmap
case_001,ImageCAS,R,/data/case_001.nii.gz,"50,62,44","51,64,45",/out/case_001_R_heatmap.nii.gz
case_001,ImageCAS,L,/data/case_001.nii.gz,"49,70,82","48,72,80",/out/case_001_L_heatmap.nii.gz
```

Alternative one-row-per-case format:

```csv
case_id,dataset,image,label_R,prediction_R,heatmap_R,label_L,prediction_L,heatmap_L
case_001,ImageCAS,/data/case_001.nii.gz,"50,62,44","51,64,45",/out/case_001_R_heatmap.nii.gz,"49,70,82","48,72,80",/out/case_001_L_heatmap.nii.gz
```

Coordinates are expected in `z,y,x`, the same order used by the training
validation code. If spacing is not available from the image or heatmap file, add
`spacing_z,spacing_y,spacing_x` columns.

Run:

```bash
python -m hcatostialocalization.analysis.analyze_predictions predictions_manifest.csv --output-dir analysis_output
```

Open the viewer:

```bash
python -m hcatostialocalization.analysis.serve_viewer analysis_output
```

Then open:

```text
http://localhost:8765
```

On a VM, open an SSH tunnel to port `8765`, then use the same browser link.
