# Apple's raw decoder on a Mac

On macOS 12 and later LightCraft can develop raws with Apple's raw decoder, Core Image's `CIRAWFilter`. Apple keeps
decoders for hundreds of cameras up to date with the system, so on a Mac the raws LightCraft's own decoders can't
develop yet develop instead of opening from their embedded JPEG, when Apple's decoder knows them: compressed Olympus
ORF and Nikon High Efficiency NEF do (measured below); Samsung's compressed SRW (NX500) does not and stays a preview.
LightCraft's own decoders stay the default for everything they can develop, and the fallback for everything else.
Elsewhere (and before macOS 12) nothing changes.

*Status (2026-10-11):* measured on an Apple M5 with macOS 26.6.2 (fork issue #2), see
[Measured on a Mac](#measured-on-a-mac); other raws only Apple develops (Nikon "lossy after split" NEF, unverified Canon
CRX variants) are untested.

## Modes

| Mode | What develops raws |
|---|---|
| `auto` (default) | LightCraft's decoders; Apple's for the raws they can't develop (those they would show from the embedded JPEG) |
| `apple` | Apple's first for every raw but DNG (LightCraft's DNG support is complete: opcodes, profiles, semantic masks, gain maps), LightCraft's when Apple's fails |
| `builtin` | LightCraft's only |

Choose it in **Settings ▸ Performance ▸ Raw decoder** (shown on a Mac), with `app.rawDecoder {mode}` (CLI `run`, MCP
`cmd_app_rawDecoder`, the control channel), or for a whole process with `LIGHTCRAFT_RAW_DECODER=auto|apple|builtin`
(which wins over the setting). `app.rawDecoder` returns `{mode, effective, available, reason, overriddenBy, identity,
lastFallback, skipped}`: `effective` is `builtin` when this Mac has no Apple decoder or it was turned off, and
`reason` says why; `lastFallback` is the latest raw Apple's decoder did not develop, and why.

## What Apple's decoder is asked to do

LightCraft's pipeline takes scene-linear Rec.2020, so Apple's picture is made like the built-in decoders' (see
`crates/coreimage/src/lib.rs` and `crates/engine/src/apple_raw.rs`):

- **Scene-linear.** Everything Apple would do to the look is off: no tone curve ("boost") or shadow boost, no local
  tone mapping, contrast, sharpening, luminance/colour noise or moiré reduction, no gamut mapping. The output is
  extended (unclipped) floats in linear Rec.2020, read back in 16 MiB strips. The camera's baseline exposure is applied
  by LightCraft as a linear gain (Core Image's own gives the same pixels on macOS 26; this way a later Core Image
  can't change it). Highlight recovery stays Apple's. The tone curve, noise reduction and sharpening are then
  LightCraft's, as for any raw.
- **Upright, in the file's own frame.** Core Image applies the file's orientation. When LightCraft's decoder can read
  the file's headers (as for unverified CRX variants), Apple's picture is centre-cropped to the headers' default crop
  (it may differ by up to 5 %; a larger difference, or a picture turned 90°, is refused and the built-in decoder's
  fallback is used), so crops, masks and region renders mean the same with either decoder; the headers' lens
  corrections are then LightCraft's (*Enable Profile Corrections*). Without readable headers (compressed ORF) Apple's
  own lens corrections apply, and the import takes the size from Apple.
- **White balance on the catalog's scale.** For the formats whose white balance the catalog keeps relative (ARW, NEF,
  RW2, RAF, CR3, CR2, PEF, SRW) the as-shot white is the relative scale's 6500 K / 0; for the others it is Apple's
  as-shot white in Kelvin and tint. White-balance changes are Bradford adaptations of Apple's picture (the built-in
  decoder re-evaluates a DNG's colour model instead). No camera-JPEG look is fitted (that corrects the built-in
  decoders' neutral matrices; Apple has its own camera colour), so Apple's pictures get LightCraft's default raw tone
  curve.
- **Small sizes are fast.** For thumbnails and previews Core Image scales the raw before it demosaics (draft mode at
  512 px and below).

## Raws imported before, and moving a library

- In `auto` and `apple` mode, an import asks Apple about a raw the built-in decoder can't read: if Apple can develop
  it, it imports as a raw that develops (with Apple's size and as-shot white), not as *preview only*.
- A raw imported as *preview only* keeps showing its embedded JPEG — its white-balance scale and defaults are a
  rendered file's until then — until **Photo ▸ Reload from Disk**, which re-reads it and makes it a raw that develops.
  The *Preview only* notice says so on a Mac.
- A library opened where there is no Apple decoder (another platform, `builtin` mode) shows those raws from their
  embedded JPEG again; Reload from Disk there marks them *preview only*.

## What stays with LightCraft's decoders

- **AI Denoise** needs the Bayer mosaic: a photo with a denoised picture is always developed by the built-in decoder.
- **Export as DNG, Convert to DNG, HDR and panorama merge, camera-profile calibration** (`lightcraft-cli calibrate`)
  read the raw data with LightCraft's decoders: a raw only Apple can develop is handled there as before (its embedded
  JPEG, or the built-in decoder's error).
- In `apple` mode, **DNG**.

## Safety

Core Image runs inside LightCraft (through the pure-Rust `objc2` bindings; `crates/coreimage` is one of the four
crates allowed `unsafe`, every block with its `SAFETY:` reasoning, behind a safe API that bounds sizes and checks what
Core Image returns). An Objective-C exception or a crash inside Core Image can't be caught in-process, so:

- every decode runs on its own thread with a time limit (20 s for a thumbnail, 40 s for a preview, 120 s at full size;
  20 s to describe a raw for import), at most two at a time; a decode over time is an error (the built-in decoder's
  fallback is used) and keeps its slot until it finishes;
- after three decodes in a row over time, or when every slot is held by one that never finished, Apple's decoder is
  turned off for the session (`app.rawDecoder` → `reason`);
- the desktop app leaves a marker in `<config>/apple-raw/` while Apple develops a raw. A marker still there at the next
  launch means Apple's decoder stopped LightCraft on that file: the file goes on `<config>/apple-raw/skip.txt` and is
  left to the built-in decoder from then on (`app.rawDecoder` → `skipped`; `{clearSkipped: true}` gives them another
  try). So a crash can happen once per file, not every time the grid shows it. No markers with `LIGHTCRAFT_NO_PREFS`,
  in a `--memory` session, the CLI or MCP.

## Caches

Renders of a raw Apple's decoder may develop are keyed by the decoder too (mode, this crate's revision and the macOS
version, since Apple updates its decoders with the system): switching modes or updating macOS re-renders those
thumbnails, and a source decoded before a switch is not taken into the cache. JPEGs, raws shown from their embedded
preview, and every photo in `builtin` mode (so every platform but macOS) keep their keys.

## Comparing the decoders

```sh
cargo run --release -p lightcraft-engine --example raw_decoders -- --out /tmp/raw-cmp --timing --sweep FILE...
```

For each file: both decoders' size and time; Apple's exposure against the built-in decoder's (`gainEv`); the tone
response by stop below white (`tone`: Apple / built-in at matched exposure, 1 everywhere means the same, linear,
response); how much goes above white (`appleOver1`, `appleMax`); a fitted 3×3 matrix (identity: the same colour) and
the Oklab ΔE at matched exposure; where Apple's picture sits on the built-in one (`frame`: `scale` above 1 means
Apple's frame is tighter, `dx`/`dy` a shift in fractions of the width and height, `match` the correlation there);
side-by-side renders with the default raw settings in `--out`. `--sweep` repeats the comparison with Core Image's
tuning changed one setting at a time (Apple's shadow bias, no highlight recovery, Apple's lens corrections). A file
the built-in decoder shows from its embedded preview (`"raw": false`, `previewOnly` says why) has no colour to
compare: only Apple develops it, and `previewFrame` places Apple's picture on the camera's JPEG. `--self-check` shows
what identical looks like, and a frame zoomed 3 %, on any platform.

## Testing on a Mac

```sh
cargo test -p lightcraft-coreimage                                   # strips read back in order, from any extent
LC_APPLE_RAW=/path/to/file.CR3 cargo test --release -p lightcraft-coreimage --lib real_raw -- --ignored --nocapture
cargo test -p lightcraft-engine apple_raw                            # the engine's routing, guard and caches
LIGHTCRAFT_RAW_DECODER=apple cargo run --release -p lightcraft-cli -- render FILE -o apple.jpg
LIGHTCRAFT_RAW_DECODER=builtin cargo run --release -p lightcraft-cli -- render FILE -o builtin.jpg
```

## Measured on a Mac

Apple M5 (10 cores, 32 GB), macOS 26.6.2 (`ciraw1:26.6.2`), 2026-10-11: `raw_decoders --timing --sweep` on CC0 raws
from raw.pixls.us and a portrait ARW (33 MP, 7008 × 4672), plus the tests above, `lightcraft-cli render` in both modes
and headless imports.

- **What only Apple develops.** Compressed ORF (E-M1 Mark II, two files): the built-in decoder shows the embedded
  3200 × 2400 JPEG, Apple develops all 5184 × 3888 pixels. Nikon High Efficiency NEF (Z 8 HE and HE\*, Z 9 HE): the
  built-in decoder shows the embedded JPEG, Apple develops the 8256 × 5504 sensor. In `auto` they import as raws that
  develop (not *preview only*). Samsung's compressed SRW (NX500) Apple can't develop either: it imports as *preview
  only*, its reason naming both decoders', and `app.rawDecoder` reports it as `lastFallback`. Every Canon CR3 tried
  develops with the built-in decoder, so `auto` leaves them to it: R6 RAW and C-RAW, R8 (Apple develops these too,
  ΔE below), and on Linux R3, R5 Mark II RAW and C-RAW, R6 Mark III, R7, R10, R50 V, R100 and M50 Mark II.
- **Linear, unclipped, upright, the same frame.** At matched exposure Apple's tone response is within 0.94–1.17 of the
  built-in decoder's from 10 stops below white to white (the A7 III: 0.875 one stop below white). Highlights above
  white are kept (Apple's brightest values 2.2–2.9 × white on the files that have them). Orientation matches, portrait
  included. Apple's picture shows exactly the built-in decoder's frame (scale 1.000, no shift, correlation
  0.994–0.9996), and on the raws only Apple develops, the camera JPEG's frame (correlation 0.98–0.99). With Apple's
  lens corrections on, the portrait ARW's frame would be 1.8 % tighter: they stay off when the headers are readable.
- **Exposure.** A DNG: the same (−0.003 EV; both apply its `BaselineExposure`). Other raws: Apple's are 0.23–0.38 EV
  brighter, because Apple applies its own baseline exposure for the camera (R6 +0.5 EV, E-M1 Mark II +0.9 EV) and the
  built-in decoder none (its fit to the camera's JPEG sets the default look's brightness).
- **Colour** at matched exposure (Oklab ΔE × 100, mean / 95th percentile; a fitted 3×3 matrix's residual): DNG (5D Mark
  III) 0.54 / 1.17; R8 0.97 / 2.60; D7500 1.37 / 4.13; portrait ARW 1.39 / 3.27; R6 1.41 / 2.86 (C-RAW the same);
  A7 III 1.68 / 3.97; residual 0.002–0.019. Rendered with the default raw settings, Apple's pictures look darker and
  more contrasty and saturated than the built-in decoder's, which take the look of the camera's JPEG (a portrait's sky
  that clips there keeps its clouds with Apple's).
- **Speed** (ms, built-in / Apple; one decode after the first, which takes Core Image 1.1–1.6 s to start):

  | File | 512 px | 2,560 px | Full size |
  |---|---|---|---|
  | 5D Mark III DNG, 22 MP | 60 / 94 | 136 / 128 | 540 / 247 |
  | A7 III ARW, 24 MP | 42 / 196 | 86 / 237 | 529 / 352 |
  | R8 CR3, 24 MP | 118 / 222 | 130 / 286 | 414 / 391 |
  | D7500 NEF, 21 MP | 132 / 208 | 177 / 236 | 503 / 324 |
  | R6 CR3, 20 MP | 198 / 184 | 230 / 210 | 529 / 301 |
  | R6 C-RAW CR3, 20 MP | 237 / 201 | 263 / 226 | 575 / 312 |
  | portrait ARW, 33 MP | 104 / 233 | 187 / 363 | 829 / 631 |
  | E-M1 Mark II ORF, 20 MP | 15 / 224 (built-in: the JPEG) | 40 / 250 | 25 / 338 |

  Apple takes about 200 ms for any decode, so the built-in decoder is faster for thumbnails and previews; at full size
  Apple is faster (1.1–2.2×).
- **Tuning.** Core Image's shadow boost and EDR amount change no pixel once its tone curve is off, and its baseline
  exposure gives the same pixels as LightCraft's gain: none of them is a setting any more. Shadow bias stays 0:
  Apple's default for each camera darkens the deepest shadows to 1–15 % of the built-in decoder's at 9 stops below
  white, and makes ΔE worse on four of seven files (DNG 0.54 → 1.13, R8 0.97 → 1.21; clearly better only on the R6,
  1.42 → 1.29). Highlight recovery stays on (without it the brightest values go from 2.6 to 3.3 × white, and the A7 III's
  tone response at white drops from 0.94 to 0.72).
