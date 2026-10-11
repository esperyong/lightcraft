# Apple's raw decoder on a Mac

On macOS 12 and later LightCraft can develop raws with Apple's raw decoder, Core Image's `CIRAWFilter`. Apple keeps
decoders for hundreds of cameras up to date with the system, so on a Mac the raws LightCraft's own decoders can't
develop yet (unverified Canon CRX variants, compressed Olympus ORF and Samsung SRW, Nikon "lossy after split" NEF…)
develop instead of opening from their embedded JPEG. LightCraft's own decoders stay the default for everything they
can develop, and the fallback for everything else. Elsewhere (and before macOS 12) nothing changes.

*Status (2026-10-11):* implemented and tested on Linux with a stand-in for Core Image; measured on real files on an
Apple Silicon Mac is pending (fork issue #2). Numbers go in [Measured on a Mac](#measured-on-a-mac).

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
  by LightCraft as a linear gain (Core Image would clip the highlights it pushes past white). Highlight recovery stays
  Apple's. The tone curve, noise reduction and sharpening are then LightCraft's, as for any raw.
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
the Oklab ΔE at matched exposure; side-by-side renders with the default raw settings in `--out`. `--sweep` repeats the
comparison with Core Image's tuning changed one setting at a time. A file the built-in decoder shows from its embedded
preview (`"raw": false`) has nothing to compare against: only Apple develops it. `--self-check` shows what identical
looks like, on any platform.

## Testing on a Mac

```sh
cargo test -p lightcraft-coreimage                                   # strips read back in order, from any extent
LC_APPLE_RAW=/path/to/file.CR3 cargo test --release -p lightcraft-coreimage --lib real_raw -- --ignored --nocapture
cargo test -p lightcraft-engine apple_raw                            # the engine's routing, guard and caches
LIGHTCRAFT_RAW_DECODER=apple cargo run --release -p lightcraft-cli -- render FILE -o apple.jpg
LIGHTCRAFT_RAW_DECODER=builtin cargo run --release -p lightcraft-cli -- render FILE -o builtin.jpg
```

## Measured on a Mac

Pending: decode times (thumbnail, preview, 24 MP full size) against the built-in decoder, the tone response and
colour difference on the corpus' raws, and CR3 / compressed ORF samples the built-in decoders can't develop.
