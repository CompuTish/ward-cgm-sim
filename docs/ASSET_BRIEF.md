# Art brief — ward CGM simulator tileset

A commission brief for a pixel-art tileset and character set for an academic
hospital-ward simulator. Everything here is a hard requirement unless marked
*optional*. If any requirement is impractical, say so rather than substituting —
a mismatch in tile size or perspective is expensive to discover late.

---

## 0. The single most important constraint

**The game is ORTHOGONAL top-down, not isometric.**

The camera looks straight down at a square grid. Think Pokémon FireRed /
Emerald, Zelda: A Link to the Past, Stardew Valley — *not* Project Hospital,
Two Point Hospital, The Sims, or anything drawn on a 2:1 diamond.

If isometric art is supplied it cannot be used: the renderer, the ward map and
the movement grid are all square-tile, and converting would mean rebuilding
them. Reference images of isometric hospital games may have been shared as a
*quality* target — the fidelity and readability are the goal, the projection is
not.

**Perspective rules within that:**
- Floors are drawn flat, straight down.
- Furniture and walls are drawn with a slight forward lean so the *front face*
  is visible — the standard Pokémon cheat. A bed shows its top surface plus a
  hint of the near side. No vanishing point, no true perspective.
- Every object faces the viewer (south). No rotated variants needed unless
  listed below.
- Light comes from the top-left. Shadows fall bottom-right, soft and short.

---

## 1. Technical specification

| Property | Value |
|---|---|
| Tile size | **16 × 16 px** |
| Display scale | 2× (rendered at 32 × 32 on screen, nearest-neighbour) |
| Format | PNG-32 with real alpha, no matte, no semi-transparent AA on outer edges |
| Colour depth | Indexed or RGB, but see the palette rule below |
| Grid | Assets must tile seamlessly on a 16 px grid with no bleed into neighbours |
| Sheet layout | One PNG per category, tiles packed left-to-right, top-to-bottom, no padding, no margin |

**Pixel discipline (this is a pixel-art brief, not a smooth-illustration brief):**
- 1 px hard outlines in a darker shade of the object's own colour, not black.
- **No anti-aliasing on outer silhouettes.** Interior AA is fine and encouraged
  for curves and shading, at most 2 intermediate shades.
- No gradients, no soft airbrush, no drop shadows with blur, no bloom.
- No sub-pixel detail that vanishes at 1×: assume the worst case is a viewer on
  a 1× display.

**Palette:** limited and shared. Aim for **≤ 48 colours across the whole set**,
with a common set of neutrals so tiles sit together. The existing prototype uses
these anchors — match them approximately so the HUD and the art agree:

| Role | Hex |
|---|---|
| Floor light | `#E8E2D4` |
| Floor mid | `#DED7C8` |
| Wall dark | `#566070` |
| Wall light | `#788496` |
| Bed frame | `#B0B6C4` |
| Bed linen | `#F4F6FA` |
| Pillow | `#D6DEEC` |
| Desk wood | `#806A4E` |
| Screen glow | `#4AA8C4` |
| Alarm red | `#D8443A` |
| Alarm amber | `#E89E3E` |
| OK green | `#48B080` |
| Inactive grey | `#8C8C96` |

---

## 2. Tileset — `tiles.png`

16 × 16 each. Order matters; index them in this sequence.

**Floor (6 tiles)**
1. Ward floor, plain vinyl, light
2. Ward floor, plain vinyl, mid (subtle alternate for a checker feel)
3. Ward floor with a faint scuff / seam detail
4. Corridor floor (visually distinct from bay floor — slightly cooler or with a directional line)
5. Bay threshold / doorway floor
6. Entrance mat

**Walls (8 tiles)** — a wall must read as a wall from above, so give it a
visible top surface *and* a front face.
7. Wall, horizontal run (front face visible)
8. Wall, horizontal, top edge (viewed from behind)
9. Wall, vertical run, left side
10. Wall, vertical run, right side
11. Wall corner, outer top-left
12. Wall corner, outer top-right
13. Wall corner, inner
14. Wall with a window / vision panel

**Ward furniture (10 tiles)**
15. Hospital bed, empty, made — head at the top of the tile
16. Hospital bed, empty, sheets disturbed
17. Bedside cabinet / locker
18. IV drip stand
19. Curtain rail, drawn open
20. Curtain rail, drawn closed (privacy screen)
21. Chair, visitor
22. Sink / hand-wash basin
23. Alcohol gel dispenser (wall-mounted)
24. Clinical waste bin

**Nurse station (6 tiles)** — this is a 4×2 block in the ward, so these need to
combine into a desk run.
25. Desk, left end
26. Desk, middle run
27. Desk, right end
28. Desk with telemetry monitor, screen ON (the central dashboard)
29. Desk with telemetry monitor, screen showing an ALARM state (red)
30. Desk with keyboard / notes

**Rooms and doors (6 tiles)**
31. Drug room door, closed (should read as secure/locked — a keypad or a green cross)
32. Drug room floor / interior hint
33. Ward entrance doors, double, closed
34. Ward entrance doors, double, open
35. Store cupboard
36. Sluice / dirty utility door

*Optional but valuable:* a few decorative tiles — a wall clock, a noticeboard,
a potted plant, a hand-hygiene poster. These do nothing mechanically but make
the ward feel inhabited.

---

## 3. Characters — `characters.png`

**Sprite size: 16 × 24 px** (taller than a tile — the head overlaps the tile
above, as in Pokémon). Anchored bottom-centre.

Each character needs **4 directions × 3 frames** = 12 sprites, laid out as
4 rows (down, left, right, up) × 3 columns (left-step, idle, right-step).
The idle frame is the middle column and is used when standing still.

| # | Character | Notes |
|---|---|---|
| 1 | **Ward nurse (the player)** | Navy blue scrubs. Must be instantly distinguishable from every other character — this is who the viewer follows. Consider a lanyard or a slightly brighter tone. |
| 2 | Healthcare assistant (HCA) | Pale blue / light teal tunic |
| 3 | Staff nurse | Mid blue tunic, distinct from both the player and the HCA |
| 4 | Doctor | White coat over dark trousers, stethoscope if it reads at this size |
| 5 | Surgeon | Green/teal scrubs, scrub cap |
| 6 | Diabetes specialist nurse | Purple tunic |
| 7 | Patient, walking | Hospital gown, pale, slightly slower/stooped posture |

**Skin tones:** each character needs **5 skin-tone variants**. Supply either as
separate rows, or — strongly preferred — design so that skin occupies a
dedicated palette index that can be swapped programmatically. Suggested tones:
`#F0CDB2`, `#DEB08C`, `#BE8C68`, `#966848`, `#6C4A34`. Hair colour should vary
too; two or three options per character is plenty.

**Patient in bed — `patients_in_bed.png`, 16 × 16, 5 tiles**
A patient lying under a blanket, head on the pillow, one per skin tone. This
overlays the bed tile, so it must align with tile 15 exactly.

---

## 4. Status overlays — `overlays.png`

16 × 16 each, drawn on top of a bed tile. These carry the actual clinical
meaning in the simulation, so **readability at a glance beats prettiness**.

1. **Sensor attached, working** — a small green pip or a stylised CGM patch,
   top-right corner of the tile
2. **Sensor attached, signal lost** — same shape, grey, with a question mark or
   a broken-signal motif
3. **Alarm: hypoglycaemia** — amber/orange border treatment or a corner badge
4. **Alarm: severe hypoglycaemia** — red, visually louder than #3
5. **Alarm: hyperglycaemia** — yellow
6. **Alarm: rapid fall** — orange with a downward arrow
7. **Alarm: rapid rise** — yellow with an upward arrow
8. **Point-of-care test in progress** — a small glucometer or droplet icon
9. **Treatment given** — a tick or a small syringe/gel icon
10. **Patient ready for discharge** — a suitcase or an open-door icon
11. **Player highlight ring** — a soft ring/arrow that sits under the player
    sprite so they are never lost on a busy ward
12. **Selected/adjacent bed indicator** — a subtle frame showing which bed the
    player can currently interact with

Alarm overlays 3–7 should be designed to **animate by alternating with an empty
frame** (a simple two-state blink), so please keep them as a single tile each;
the code handles the blink.

---

## 5. What the ward looks like (context for composition)

The map is a fixed 25 × 21 tile grid:

- **Four bays of eight beds**, arranged as two columns of four beds each, with a
  walkable gap between the columns. Bays sit in the four corners.
- **Nurse station** in the centre: a 4 × 2 block of desk tiles carrying the
  telemetry dashboard.
- **Drug room** on the right-hand wall.
- **Ward entrance** at the bottom centre; patients queue outside it waiting for
  a bed.
- Wide corridors between everything.

A patient occupies a bed tile. Staff walk the corridors. The player walks up to
a bed to interact with it.

---

## 6. Deliverables

- `tiles.png` — the tileset, packed in the order above
- `characters.png` — 7 characters × 4 directions × 3 frames
- `patients_in_bed.png` — 5 tiles
- `overlays.png` — 12 tiles
- A plain-text or JSON index listing what is at each tile position
- *Optional:* the source file (Aseprite `.ase` strongly preferred, or `.pyxel`,
  or layered `.psd`)

**Licensing — non-negotiable.** All artwork must be **original**. It must not be
traced from, derived from, or recoloured from Nintendo, Game Freak, Two Point,
Project Hospital, or any other commercial game's assets. This repository is
public and academic; it will be cited in a Master's project. Please confirm the
artwork is original and state the licence you are granting (MIT or CC0 preferred
so it can live in the public repo).

---

## 7. How it will be used

Assets drop into `ward_cgm_sim/render/assets/` and are loaded by
`ward_cgm_sim/render/sprites.py`, which currently generates every sprite
procedurally from rectangles. That procedural fallback stays in the codebase, so
partial delivery is fine — tiles can land before characters, and anything not yet
supplied keeps its placeholder.

The simulator is an academic model of inpatient glucose monitoring. It is not a
game being sold, and it is not clinical software. Nothing in the art should
imply a real hospital, a real product, or real clinical guidance.
