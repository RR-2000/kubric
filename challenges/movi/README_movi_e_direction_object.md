# MOVi-E direction-versus-object export

This pipeline downloads the public `movi_e/256x256` validation split and builds
a deterministic matched direction-versus-object diagnostic.

## Outputs

Raw local exports are stored in `/home/ramanathan/data/movi_e_export`. Each
sequence contains RGB frames, `metadata.json`, and—by default—segmentation,
depth, flow, normal, and object-coordinate modalities. `export_info.json`
records the source TFDS configuration and export counts.

Processed files are stored in `/home/ramanathan/data/movi_e_better_sample`:

- `movi_e_validation_direction_object.jsonl`
- `movi_e_validation_direction_object.parquet`
- `sample_pairs.jsonl`
- `audit_report.json`
- `dataset_info.json`

MOVi-E objects use humanized Google Scanned Objects `asset_id` values as their
names. Category, scale, dynamic/static state, visibility, boxes, and 3D
positions remain in each object record.

For each accepted `(image, anchor, target, relation)` source, the builder emits
one direction-answer row and one object-answer row. Directions are `left`,
`right`, `front`, and `behind` in the local coordinate frame of an anchor facing
the camera. Both rows share the same source ID, image, objects, relation, and
candidate pool.

Object choices are selected deterministically. After selection, every displayed
option is classified relative to the anchor. The entire pair is dropped if more
than one displayed object has the gold relation. Duplicate object names and
records with fewer than four visible unique objects are also dropped.

## Run

Submit the combined export/build job:

```bash
qsub /home/ramanathan/VLM/kubric/pbs_run_movi_e_direction_object.sh
```

For a quick local build check over already exported sequences:

```bash
python challenges/movi/build_movi_e_direction_object_dataset.py \
  --input-dir /home/ramanathan/data/movi_e_export \
  --output-dir /home/ramanathan/data/movi_e_better_sample \
  --max-sequences 2
```

The exporter is resumable: complete sequence directories are skipped, while an
incomplete directory is rebuilt. Pass `--overwrite` to force re-exporting all
selected examples.
