# Vendored MaRO-GS training backend

Source: https://github.com/kgh1234/MaRO-GS
Revision: `901e18cd35a6d86aff84b257f7f4181f5422d438` (retrieved 2026-11-10).
The training entry point and its arguments/scene/utils/renderer dependencies are
included. The upstream README is retained for attribution and reference; its
evaluation scripts, assets and dataset tools are not bundled.

License: [LICENSE.md](LICENSE.md), non-commercial research/evaluation.
Retain upstream copyright notices and license when redistributing.

Local compatibility changes:

- `scene/mask_readers.py`: prefer the exact mask stem before wildcard fallback
  (otherwise `1.png` also selects `10.png`); transpose reversed dimensions only
  when the current shape differs from the expected shape (square images keep
  their orientation).
- `gaussian_renderer/__init__.py`: accept 4-output rasterizers used by this repo;
  the first three outputs are RGB, radii, depth. Additional alpha is unused here.
- `gaussian_renderer/network_gui.py`: create the viewer socket in `init` only,
  so `--disable_viewer` does not open an unused socket.

The official view filtering, pruning/densification, merging and losses remain
in this backend. Run through `scripts/train_maro.py` to prepare data and isolate
its Python module names from those of the enclosing repository.
