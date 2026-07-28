# Ward CGM simulator pixel assets

This directory contains the commissioned orthogonal top-down pixel-art set
described in `docs/ASSET_BRIEF.md`.

- `tiles.png`: 36 ordered 16×16 tiles, 6×6 with no padding.
- `characters.png`: 84 ordered 16×24 sprites, 28 rows × 3 columns.
- `patients_in_bed.png`: one indexed 16×16 bed-aligned patient overlay.
- `overlays_bed.png`: 11 ordered 16×16 status overlays.
- `overlays_effect.png`: one required 16×8 player ring followed by the optional
  16×16 busy marker, packed vertically without padding.
- `assets-index.json`: coordinates, ordering, the shared palette, and swappable
  skin/blanket palette indices and tables.
- `docs/asset-qa/`: nearest-neighbour visual checks; not runtime assets.

All runtime sheets share a 48-entry indexed palette. Palette index 0 is fully
transparent. Outer silhouettes contain no partial alpha. Skin uses indices 28,
46 and 47; blanket/identity trim uses indices 36 and 37.

## Originality and licence

The artwork was created from scratch for this repository from the written
commission brief and a newly generated visual-direction reference. No Nintendo,
Game Freak, Two Point, Project Hospital, or other commercial-game artwork was
traced, recoloured, copied, or included.

To the extent permitted by law, the asset author dedicates these files to the
public domain under **CC0 1.0 Universal**. See `LICENSE-CC0.txt`.

Regenerate with:

```bash
python scripts/generate_ward_assets.py
```

Pillow is development-only; the simulator runtime remains pygame-ce-only.
