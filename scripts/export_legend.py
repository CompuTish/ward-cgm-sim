#!/usr/bin/env python3
"""Export a legend of the ward artwork as standalone PNGs.

The simulator draws its own HUD inside the WebAssembly canvas, which the
browser then stretches - so that text can only ever be as crisp as the canvas
scaling allows. Anything explanatory therefore belongs in the page around the
demo, as real HTML, where it stays sharp at any zoom and can be read by a
screen reader.

This writes one crop per sprite plus a manifest, so the project page can show a
key of what each character and marker on the ward actually means.

    python scripts/export_legend.py --out ../site_isabelsmith.me/public/projects/ward-sim/legend

Development only: needs Pillow, and nothing here ships to the browser bundle.
"""

import argparse
import json
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "ward_cgm_sim" / "render" / "assets"

# Displayed at 4x so the pixel grid stays hard on a high-resolution screen.
SCALE = 4

# (file, entry name in the manifest, output slug, label, what it means)
PEOPLE = [
    ("ward_nurse_player", "you", "You",
     "The nurse you control. The tall white cap with the raised centre is unique to you."),
    ("staff_nurse", "staff-nurse", "Staff nurse",
     "A colleague on the ward. Plain white cap - the same family as yours, one step down."),
    ("healthcare_assistant", "hca", "Healthcare assistant",
     "No cap, and carrying a tray. Helps with checks, point-of-care tests and discharge prep."),
    ("doctor", "doctor", "Doctor",
     "Long white coat and a stethoscope. Reviews, prescriptions and medical escalation."),
    ("surgeon", "surgeon", "Surgeon",
     "Green scrub cap and mask. Reviews and discharge decisions for surgical patients."),
    ("diabetes_specialist_nurse", "dsn", "Diabetes specialist nurse",
     "Cap plus a shoulder bag strap. The specialist escalation route for glycaemic events."),
    ("patient_walking", "patient-walking", "Patient walking",
     "A patient out of bed - arriving, transferring or going home - with a drip stand."),
]

TILES = [
    ("hospital_bed_made", "bed-empty", "Empty bed",
     "Made up and free. A free bed can take the next admission from the queue."),
    ("hospital_bed_disturbed", "bed-vacated", "Bed left unmade",
     "The patient is off the ward or walking. The bed is still theirs, so it is not free."),
    ("desk_monitor_on", "station", "Nurse station",
     "The telemetry board. Standing here refreshes it for free; from elsewhere it costs a step."),
    ("desk_monitor_alarm", "station-alarm", "Board alarming",
     "The monitor reddens while any alarm is unresolved."),
    ("drug_room_door_closed", "drug-room", "Drug room", "Where treatments are drawn up."),
    ("ward_entrance_double_open", "entrance", "Ward doors",
     "Admissions arrive here and discharges leave through it."),
]

OVERLAYS = [
    ("sensor_working", "sensor-ok", "On telemetry",
     "This patient is enrolled and their sensor is reporting."),
    ("sensor_signal_lost", "sensor-lost", "Signal lost",
     "Enrolled, but the sensor has dropped out. It does NOT alarm - noticing is the point."),
    ("alarm_hypoglycaemia", "alarm-hypo", "Hypoglycaemia",
     "Glucose below 3.9 mmol/L."),
    ("alarm_severe_hypoglycaemia", "alarm-severe-hypo", "Severe hypoglycaemia",
     "Below 3.0 mmol/L. Left untreated this is what ends a shift badly."),
    ("alarm_hyperglycaemia", "alarm-hyper", "Hyperglycaemia",
     "Above 14.0 mmol/L, or above this patient's own threshold if one is set."),
    ("alarm_rapid_fall", "alarm-fall", "Falling fast",
     "A rapid downward trend, before any threshold is crossed."),
    ("alarm_rapid_rise", "alarm-rise", "Rising fast", "A rapid upward trend."),
    ("point_of_care_test", "poc", "Point-of-care test",
     "A finger-prick reading. It is trusted over the sensor."),
    ("treatment_given", "treated", "Treatment given",
     "Treatment is in progress; glucose responds over the following half hour."),
    ("ready_for_discharge", "discharge", "Ready for discharge",
     "Shown only once you have established it - the ward knows before you do."),
    ("selected_adjacent_bed", "selected", "Bed in reach",
     "The bed a patient-directed key would act on."),
]


def load_manifest() -> dict:
    return json.loads((ASSETS / "assets-index.json").read_text(encoding="utf-8"))


def crop(sheet: Image.Image, item: dict) -> Image.Image:
    box = (item["x"], item["y"], item["x"] + item["width"], item["y"] + item["height"])
    piece = sheet.crop(box).convert("RGBA")
    return piece.resize((piece.width * SCALE, piece.height * SCALE), Image.NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    entries = []

    with Image.open(ASSETS / "characters.png") as characters:
        by_key = {
            (i["character"], i["direction"], i["frame"]): i
            for i in manifest["sheets"]["characters.png"]["items"]
        }
        for character, slug, label, meaning in PEOPLE:
            item = by_key[(character, "down", "idle")]
            crop(characters, item).save(out / f"{slug}.png")
            entries.append({"slug": slug, "label": label, "meaning": meaning,
                            "group": "people"})

    with Image.open(ASSETS / "tiles.png") as tiles:
        by_name = {i["name"]: i for i in manifest["sheets"]["tiles.png"]["items"]}
        bed = by_name["hospital_bed_made"]
        for name, slug, label, meaning in TILES:
            crop(tiles, by_name[name]).save(out / f"{slug}.png")
            entries.append({"slug": slug, "label": label, "meaning": meaning,
                            "group": "places"})

        # An occupied bed is a composite, so build it the way the renderer does.
        with Image.open(ASSETS / "patients_in_bed.png") as occupant:
            base = crop(tiles, bed)
            item = manifest["sheets"]["patients_in_bed.png"]["items"][0]
            base.alpha_composite(crop(occupant, item))
            base.save(out / "bed-occupied.png")
            entries.append({
                "slug": "bed-occupied", "label": "Occupied bed", "group": "places",
                "meaning": "Each patient keeps one blanket colour for the whole "
                           "shift, so you can follow them from bed to door.",
            })

    with Image.open(ASSETS / "overlays_bed.png") as overlays:
        by_name = {i["name"]: i for i in manifest["sheets"]["overlays_bed.png"]["items"]}
        for name, slug, label, meaning in OVERLAYS:
            # Overlays are drawn on top of a bed, so show them that way.
            with Image.open(ASSETS / "tiles.png") as tiles:
                bed_item = [i for i in manifest["sheets"]["tiles.png"]["items"]
                            if i["name"] == "hospital_bed_made"][0]
                base = crop(tiles, bed_item)
            base.alpha_composite(crop(overlays, by_name[name]))
            base.save(out / f"{slug}.png")
            entries.append({"slug": slug, "label": label, "meaning": meaning,
                            "group": "markers"})

    (out / "legend.json").write_text(
        json.dumps({"scale": SCALE, "entries": entries}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(entries)} legend images to {out}")


if __name__ == "__main__":
    main()
