# Ward CGM simulator pixel assets

This directory contains the orthogonal top-down pixel-art set described in
`docs/ASSET_BRIEF.md`.

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

Every pixel here is drawn by `scripts/generate_ward_assets.py`, which is part of
this repository: the sheets are program output rather than imported artwork,
and re-running the script reproduces them byte for byte. They were written from
scratch against `docs/ASSET_BRIEF.md`. No Nintendo, Game Freak, Two Point,
Project Hospital, or other commercial-game artwork was traced, recoloured,
copied, or included.

The authors of this repository — Isabel Smith and Aatish Thakerar, per the root
`LICENSE` — therefore hold the rights in this artwork in full, and to the extent
permitted by law dedicate it to the public domain under **CC0 1.0 Universal**.
See `LICENSE-CC0.txt`.

Regenerate with:

```bash
python scripts/generate_ward_assets.py
```

Pillow is development-only; the simulator runtime remains pygame-ce-only.
