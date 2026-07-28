# Art brief - ward CGM simulator tileset

A commission brief for a pixel-art tileset and character set for an academic
hospital-ward simulator. Everything here is a hard requirement unless marked
*optional*. If any requirement is impractical, say so rather than substituting -
a mismatch in tile size or perspective is expensive to discover late.

---

## 0. The single most important constraint

**The game is ORTHOGONAL top-down, not isometric.**

The camera looks straight down at a square grid. **The direct reference is the
Pokémon Centre interior from the GBA-era games** (FireRed/LeafGreen, Ruby/
Sapphire/Emerald) - that exact projection, tile size, and readability. Also
Zelda: A Link to the Past, Stardew Valley. It is *not* Project Hospital, Two
Point Hospital, The Sims, or anything drawn on a 2:1 diamond.

If isometric art is supplied it cannot be used: the renderer, the ward map and
the movement grid are all square-tile, and converting would mean rebuilding
them. Reference images of isometric hospital games may have been shared as a
*quality* target - the fidelity and readability are the goal, the projection is
not.

**Perspective rules within that:**
- Floors are drawn flat, straight down.
- Furniture and walls are drawn with a slight forward lean so the *front face*
  is visible - the standard Pokémon cheat. A bed shows its top surface plus a
  hint of the near side. No vanishing point, no true perspective.
- Every object faces the viewer (south). No rotated variants needed unless
  listed below.
- Light comes from the top-left. Shadows fall bottom-right, soft and short.

---

## 0b. Role legibility - the thing that makes this style work

In a Pokémon Centre you know instantly that Nurse Joy is a nurse. Not because
of a label, and not because of her tunic colour - because of the **cap with the
cross**, the silhouette, and the colour block, all reading together at 16 px.

This simulation lives or dies on the same thing. A viewer watching a shift needs
to tell, at a glance and without pausing:

- who the **player nurse** is, among five other staff on screen
- which colleague just walked over - a doctor, a surgeon, an HCA?
- **which patient** is which, so a specific patient can be followed across the
  shift as their glucose changes

Three hard rules follow:

1. **Every role must be identifiable by a distinct ICONIC ATTRIBUTE, not by
   tunic colour.** Colour is the secondary cue, never the primary one. Two
   characters differing only in shade of blue is a failure.
2. **The silhouette test.** Fill any character sprite with solid black. The role
   must still be guessable from the outline alone - cap shape, coat length,
   headwear, what they carry. If two silhouettes are identical, redesign one.
3. **The squint test.** At 1× (16 × 24 px, unmagnified), blur or squint. Role
   and identity must survive. Detail that only appears at 4× is decoration, not
   information.

Assume some viewers are colourblind. Deuteranopia is the common case: a green
scrub top and a red-brown one become the same colour. The attribute must carry
the meaning so the colour never has to.

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
these anchors - match them approximately so the HUD and the art agree:

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

## 2. Tileset - `tiles.png`

16 × 16 each. Order matters; index them in this sequence.

**Floor (6 tiles)**
1. Ward floor, plain vinyl, light
2. Ward floor, plain vinyl, mid (subtle alternate for a checker feel)
3. Ward floor with a faint scuff / seam detail
4. Corridor floor (visually distinct from bay floor - slightly cooler or with a directional line)
5. Bay threshold / doorway floor
6. Entrance mat

**Walls (8 tiles)** - a wall must read as a wall from above, so give it a
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
15. Hospital bed, empty, made - head at the top of the tile. The blanket area
    must be a flat, unshaded region so the per-patient blanket colour can be
    swapped in cleanly without fighting a baked-in gradient.
16. Hospital bed, empty, sheets disturbed
17. Bedside cabinet / locker
18. IV drip stand
19. Curtain rail, drawn open
20. Curtain rail, drawn closed (privacy screen)
21. Chair, visitor
22. Sink / hand-wash basin
23. Alcohol gel dispenser (wall-mounted)
24. Clinical waste bin

**Nurse station (6 tiles)** - this is a 4×2 block in the ward, so these need to
combine into a desk run.
25. Desk, left end
26. Desk, middle run
27. Desk, right end
28. Desk with telemetry monitor, screen ON (the central dashboard)
29. Desk with telemetry monitor, screen showing an ALARM state (red)
30. Desk with keyboard / notes

**Rooms and doors (6 tiles)**
31. Drug room door, closed (should read as secure/locked - a keypad or a green cross)
32. Drug room floor / interior hint
33. Ward entrance doors, double, closed
34. Ward entrance doors, double, open
35. Store cupboard
36. Sluice / dirty utility door

*Optional but valuable:* a few decorative tiles - a wall clock, a noticeboard,
a potted plant, a hand-hygiene poster. These do nothing mechanically but make
the ward feel inhabited.

---

## 3. Characters - `characters.png`

**Sprite size: 16 × 24 px** (taller than a tile - the head overlaps the tile
above, as in Pokémon). Anchored bottom-centre.

Each character needs **4 directions × 3 frames** = 12 sprites, laid out as
4 rows (down, left, right, up) × 3 columns (left-step, idle, right-step).
The idle frame is the middle column and is used when standing still.

Each role gets an **iconic attribute** that carries the identity, plus a colour
that reinforces it. The attribute column is the requirement; the colour column
is a suggestion you may adjust for palette harmony.

| # | Character | ICONIC ATTRIBUTE (required) | Colour |
|---|---|---|---|
| 1 | **Ward nurse - the player** | **White nurse's cap with a coloured cross**, worn high so it breaks the head silhouette. Plus a visible **lanyard/ID badge** on the chest. This is the one character the viewer follows for twelve hours - it must be unmistakable from anywhere on screen. | Navy tunic, white cap |
| 2 | Healthcare assistant | **No cap, short sleeves, and a carried item** - a clipboard or a small tray held in front. Bare-headed is itself the cue. | Pale teal tunic |
| 3 | Staff nurse | **White cap, but plain - no cross.** Same silhouette family as the player, deliberately one step down in prominence. | Mid blue tunic |
| 4 | Doctor | **Knee-length open white coat** - the outline flares below the waist, unlike every tunic. Plus a **stethoscope round the neck** rendered as a distinct dark loop. | White coat, dark trousers |
| 5 | Surgeon | **Scrub cap covering all hair, plus a face mask.** Head reads as a smooth solid block with a lighter band across the lower face. | Teal scrubs |
| 6 | Diabetes specialist nurse | **Cap plus a shoulder-slung bag** (the visiting-specialist cue) - the bag strap crosses the torso diagonally, which nothing else has. | Purple tunic |
| 7 | Patient, walking | **Hospital gown with an open back**, bare lower legs, and a **drip stand held in one hand** for at least one variant. Stooped, shorter stance than staff. | Pale gown, individual blanket-colour trim (see below) |

**Sanity check before delivery:** lay all seven front-facing sprites in a row at
1×. If you cannot name each role without the labels, the set is not done.

**Skin tones:** each character needs **5 skin-tone variants**. Two acceptable
formats - the first is strongly preferred:

- **Preferred - one indexed sheet.** A single `characters.png` of
  **7 characters × 4 directions × 3 frames = 84 sprites**, laid out as 28 rows
  (character 1 down/left/right/up, then character 2, and so on) × 3 columns.
  Skin occupies **dedicated palette indices** listed in the accompanying index
  file, so the code can swap them at load time. Supply the five tones as a
  palette table, not as extra sprites.
- **Alternative - pre-rendered variants.** The same sheet repeated once per
  skin tone: **7 × 4 × 3 × 5 = 420 sprites**, as five separate files
  `characters_skin1.png` … `characters_skin5.png`, each 28 rows × 3 columns.

Suggested tones: `#F0CDB2`, `#DEB08C`, `#BE8C68`, `#966848`, `#6C4A34`. Hair
colour should vary too; two or three options per character is plenty, and may be
baked in rather than indexed.

### Patients must be individually identifiable - `patients_in_bed.png`

16 × 16, overlaying bed tile 15 exactly: a patient lying under a blanket, head
on the pillow.

This is not decorative. The simulation has up to 32 patients on screen at once
and a viewer needs to follow **one specific patient** across a shift - the one
whose glucose is falling, the one waiting for discharge. Bed numbers are printed
but are tiny; **blanket colour is the primary identifier.**

So the patient-in-bed sprite needs **two independently indexed regions**:

- **Skin** - 5 tones, as above (head on the pillow is the only skin visible)
- **Blanket** - **8 distinct colours**, swapped per patient at load time

Supply the sprite **once**, with skin and blanket on their own dedicated palette
indices listed in the index file. Do not pre-render 40 combinations.

The eight blanket colours must be distinguishable from each other **and** from
every alarm overlay colour, since they sit under the alarms. Avoid saturated
red, amber and yellow entirely - those are reserved for clinical state. Suggested
family: dusty blue, sage, lilac, teal, warm grey, soft pink, sand, pale mint.

A patient walking (character 7) should carry a **matching trim colour** on the
gown, so a patient who gets out of bed is still recognisably the same person.

---

## 4. Status overlays

These carry the actual clinical meaning in the simulation, so **readability at a
glance beats prettiness**. They come in two sheets because they anchor
differently.

### 4a. Bed overlays - `overlays_bed.png`

16 × 16 each. Composited **on top of a bed tile**, aligned to the same 16 × 16
grid cell, drawn above the bed and above the patient-in-bed sprite.

1. **Sensor attached, working** - a small green pip or a stylised CGM patch,
   top-right corner of the tile
2. **Sensor attached, signal lost** - same shape, grey, with a question mark or
   a broken-signal motif
3. **Alarm: hypoglycaemia** - amber/orange border treatment or a corner badge
4. **Alarm: severe hypoglycaemia** - red, visually louder than #3
5. **Alarm: hyperglycaemia** - yellow
6. **Alarm: rapid fall** - orange with a downward arrow
7. **Alarm: rapid rise** - yellow with an upward arrow
8. **Point-of-care test in progress** - a small glucometer or droplet icon
9. **Treatment given** - a tick or a small syringe/gel icon
10. **Patient ready for discharge** - a suitcase or an open-door icon
11. **Selected/adjacent bed indicator** - a subtle frame showing which bed the
    player can currently interact with

Alarm overlays 3–7 should be designed to **animate by alternating with an empty
frame** (a simple two-state blink), so please keep them as a single tile each;
the code handles the blink.

**Colour is reserved.** Red, amber, orange and yellow belong to clinical state
and nothing else in the set may use them as a dominant colour - not a blanket,
not a tunic, not a chair. When a bed goes red on this ward it must mean one
thing. Green is similarly reserved for "sensor working" and "treated".

Each alarm overlay must also carry a **shape** cue, not just a colour: a
downward wedge for a fall, an upward one for a rise, a filled versus hollow
badge for severe versus mild. Someone with deuteranopia must still be able to
tell a severe hypoglycaemia alarm from a hyperglycaemia one.

### 4b. Effect overlays - `overlays_effect.png`

Anchored to a **character**, not a bed, and drawn **underneath** the character
sprite so it reads as something the character is standing on or in.

1. **Player highlight ring** - 16 × 8 px, an ellipse/glow anchored to the
   player's feet (bottom-centre of their tile) so they are never lost on a busy
   ward. Must read clearly against both the pale floor and the darker corridor.
2. **Staff busy indicator** - 16 × 16, a small marker shown near a colleague
   who has been asked for help and is occupied. *Optional.*

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

- `tiles.png` - 36 tiles, packed in the order above (plus any optional extras,
  appended after 36 and listed in the index)
- `characters.png` - 84 sprites (28 rows × 3 columns) with an indexed skin
  palette, **or** five `characters_skinN.png` files of the same layout
- `patients_in_bed.png` - ONE 16 × 16 sprite aligned to bed tile 15, with skin
  and blanket on separate indexed palette regions (5 skin tones × 8 blanket
  colours supplied as palette tables, not as 40 pre-rendered tiles)
- `overlays_bed.png` - 11 tiles, bed-anchored
- `overlays_effect.png` - 1 required (16 × 8 player ring) + 1 optional
- A plain-text or JSON index listing what is at each tile position, **and the
  palette indices used for skin, blanket and any other swappable region**
- *Optional:* the source file (Aseprite `.ase` strongly preferred, or `.pyxel`,
  or layered `.psd`)

**Licensing - non-negotiable.** All artwork must be **original**. It must not be
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
partial delivery is fine - tiles can land before characters, and anything not yet
supplied keeps its placeholder.

The simulator is an academic model of inpatient glucose monitoring. It is not a
game being sold, and it is not clinical software. Nothing in the art should
imply a real hospital, a real product, or real clinical guidance.
