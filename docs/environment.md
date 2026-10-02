# Environment preparation

Provide an Enamine catalog with one `SMILES<TAB>ID` record per line:

```text
CCN	BB001
CC(=O)O	BB002
```

```bash
python scripts/prepare.py \
  --building-blocks /path/to/enamine_stock.smi \
  --template-dir data/templates \
  --env-dir /path/to/prepared/enamine \
  --num-workers 16 \
  --min-library-size 10
```

- Use a new output directory; set `data.env_dir` to it when training.
- Source building blocks are desalted and filtered to at most 50 heavy atoms before conversion.
- `--min-library-size` counts unique oriented synthons per library. Use `1` for small trial catalogs.
- Bricks have one attachment site; linkers have two. Linker orientation fixes which site attaches next.
- Original building-block IDs are retained for tracing generated paths.

The directory contains synthon libraries, precomputed features, templates, action-space definitions and an environment signature. Keep these files together; regenerate after changing the catalog or templates.
