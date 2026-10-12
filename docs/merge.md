# Photo Merge (HDR, Panorama, HDR Panorama, Focus Stack)

`crates/merge` (layer L3, no UI dependencies) implements the algorithms; `crates/engine/src/merge.rs`
plans, runs and finishes merges in the library; the desktop UI shows a dialog with a live preview.
All algorithms are our own implementations from published papers (cited in the module docs).

## Using it

- **App:** select the photos, then *Photo Merge ▸ HDR…* (`Ctrl+H`), *Panorama…* (`Ctrl+M`),
  *HDR Panorama…* or *Focus Stack…* (`Ctrl+Alt+F`) from the photo context menu (or the
  `dialog.merge*` commands). The dialog previews the merge at ≤ 1024 px and re-runs the preview when
  an option changes (slider drags after a short pause; previews already made come back at once);
  *Merge* runs the full merge in the background (progress in a toast). The result is written next to
  the first photo as `<name>-HDR.dng`, `-Pano.dng`, `-HDR-Pano.dng` or `-Focus.dng`, imported and
  selected.
- **Commands** (CLI, MCP, control channel):
  - `merge.hdr {ids?, align=true, deghost=none|low|medium|high, autoSettings=true, stack=false, preview=false, showOverlay=false, previewPath?}`
  - `merge.panorama {ids?, projection=auto|spherical|cylindrical|perspective, boundaryWarp=0..100, autoCrop=false, fillEdges=false, autoSettings=true, maxMegapixels=40, preview, previewPath?}`
  - `merge.hdrPanorama {ids?, bracket=0 (from EXIF), …both sets of options}`
  - `merge.focusStack {ids?, method=depth|average|pyramid (or b|a|c), radius=1..30 (8), smoothing=0..10 (4), align=true, alignShift, alignRotation, alignScale, alignBrightness, order=capture|selection|reverse|auto, every=1, reference=middle|first|sharpest, crop=common|reference, maxMegapixels=0, depthPath?, autoSettings=true, stack=false, preview, showDepth, previewPath?}`
- **CLI:** `lightcraft-cli merge hdr|panorama|hdr-panorama|focus [options] FILES…`;
  `lightcraft-cli synth-merge hdr|panorama|focus -o DIR` writes synthetic inputs (procedural scenes).

## What it does

**HDR** (`hdr.rs`): frames ordered by EXIF exposure (or median luminance); the middle one is the
reference. Auto Align: features on exposure-normalised luminance, RANSAC translation / similarity /
homography (the simplest model that explains ≥ 97 % of the matches). Exposure ratios are measured
(median log ratio of well-exposed pixels between brightness neighbours), EXIF is the fallback.
Radiance is a weighted mean in linear light, weight `k · (1 − smoothstep(0.8·clip, 0.96·clip, max
channel))`. Deghost: a per-pixel consistency test against the reference radiance (with a noise
floor), dilated and feathered masks; the union is the deghost overlay. Output: 16-bit float DNG.

**Panorama** (`pano/`): MOPS-style features, pairwise RANSAC homographies with Brown & Lowe's
verification, focal from homographies (or EXIF 35 mm), rotations chained along a maximum spanning
tree, Levenberg–Marquardt bundle adjustment of rotations + focal lengths, straightening.
Projections: spherical, cylindrical, perspective; *Auto* picks perspective for compact fields of
view < 110°, cylindrical when the vertical field is < 90°, else spherical. Exposure: EXIF, then
gain compensation from the overlaps. Seams: each pixel goes to the image with the highest
centre-weighted feather weight; multi-band blending (Laplacian pyramids) in log space. Boundary
Warp: a separable stretch of the smoothed valid-area edges to the canvas edges. Auto Crop: the
largest rectangle of image data (set as the develop crop, so it stays adjustable). Fill Edges:
push–pull diffusion. Output: 16-bit integer DNG (16-bit float for HDR panoramas).

## Focus stack

`crates/merge/src/focus/` combines a focus bracket — photos of one scene focused at different
distances, 2 to a few hundred — into one photo sharp throughout, plus a depth map. The three methods
follow what photographers know from Helicon Focus:

- **A — weighted average** (`average.rs`): every frame contributes with weight `fm²` (its focus
  measure squared), the weights Gaussian-smoothed (σ = 2 × Smoothing). Smooth transitions and true
  colours; detail a little softer. Flat areas average the frames, which also averages their noise.
- **B — depth map** (`depth.rs`, the default): each pixel from the frame where it is sharpest. A
  parabola through the log focus measures of the winner and its neighbours gives a sub-frame depth;
  a confidence (contrast above noise, how peaked the focus profile is, and a halo term — weaker than
  half the strongest focus within 4 × Radius) decides which depths are trusted; the rest are filled
  from around (push–pull), and the depth map is smoothed by a confidence-weighted guided filter with
  the winners' grey image as guide (radius 2 × Smoothing). Colours blend the two frames nearest the
  depth; when smoothing moves a pixel's depth beyond the frames the stacker kept for it, a second
  *gather* pass fetches those colours. Crisp detail and texture.
- **C — pyramid** (`pyramid.rs`): Laplacian pyramids of √RGB, each band taken from the frame with the
  most local energy there (window Radius / 2^level, smoothed by Smoothing / 2^level), one winner for
  all three channels; the coarsest level is the mean of the frames (no contrast lift). Best for
  overlapping fine structures (hairs, bristles) where no single frame is sharp for every layer.

*Radius* (1–30, pixels at full size) is the window the focus measure — the sum-modified Laplacian
(Nayar & Nakagawa 1994) on √luminance — is averaged over; *Smoothing* (0–10) how much the choice is
smoothed. Previews scale both with the frame size.

**Streaming.** A `FocusStacker` asks for frames pass by pass and drops each frame once it has been
pushed, so memory doesn't depend on the number of photos (per output pixel: A 20 bytes, B 46, C ≈ 24;
`state_bytes()` is tested to be the same for 10 and 40 frames). The engine decodes the next 1–4
frames (as memory allows) while the stacker works on the current ones.

1. *Order* (only `order=auto`): ≤ 512 px frames; each frame's sharpness grid (64 cells across) becomes a
   focus profile (per cell, the log ratio to the cell's mean over the frames); frames are ordered by
   the Fiedler vector of the similarity graph (spectral seriation, Atkins et al. 1998), polished by
   2-opt. In capture order, a warning suggests `auto` when > 20 % of the frames' most similar frame
   isn't a near neighbour.
2. *Align* (≤ 2048 px; raws are binned from the mosaic, not demosaiced): each frame registered to the
   previous one by inverse-compositional Lucas–Kanade (Baker & Matthews 2004) over a grey pyramid —
   shift, rotation and magnification (focus breathing) each optional, or affine — with gain/offset
   normalisation, Huber weights and a focus-change weight (pixels much sharper in one frame than the
   other show the focus change, not motion; this cut the chained error on synthetic brackets from 0.57
   to 0.21 px). A poor fit (NCC < 0.6) tries a features + RANSAC seed; a frame that matches neither
   (NCC < 0.5) is left out and reported. Pairs are chained to the middle frame (the reference: its
   geometry and colour tags are the result's), brightness per channel likewise (median log ratio of
   mid-tones). The result is cropped to the area every frame covers.
3. *Fuse* (full size): each frame resampled (Catmull–Rom) into the reference geometry, divided by its
   gain, its focus measured, folded into the method's state.
4. *Gather* (method B, only when needed).

**Output:** a 16-bit linear DNG `<first photo in focus order>-Focus.dng` with the reference frame's colour
tags (raw sources stay camera RGB, so white balance and the rest stay editable). The depth map goes
into the DNG as a DNG 1.6 semantic mask (`urn:app:lightcraft:focus:depth`, 16-bit, ≤ 4096 px; 0 = the
first photo of the focus order) unless `depthMap: false`: Masking ▸ Depth (a Depth Range mask) selects
on it, e.g. to darken or blur what lies behind the subject. Optionally also as a 16-bit grey PNG
(`depthPath`). `info` reports the method, focus order, used and
skipped photos, the reference, alignment ranges (shift, rotation, scale, lowest NCC), warnings and the
seconds each pass took (`LIGHTCRAFT_PROFILE=1` also prints them).

**Many brackets.** `merge.detectFocusBrackets {ids?, gapSecs=2, minFrames=3, sameExposure=true, stack}`
finds focus brackets (runs of photos at most `gapSecs` apart with the same camera, lens, focal length
and exposure; merge results are left out) in the given photos, the selection or the photos shown, and
optionally stacks each. `merge.focusStackBatch {groups? | the same scope, …merge.focusStack options}`
stacks them one after the other (a bracket that fails is reported, the others still run). In the app,
Photo Merge ▸ Focus Stack Each Bracket queues them in the background with the last focus-stack options.

**For agents:** `merge.focusStack {ids, preview: true, previewPath, every: 4}` for a quick look (every 4th
photo, ≤ 1024 px), compare `method: "a"|"b"|"c"`, `showDepth: true` to see the depth map, then run it
without `preview`. `order: "auto"` when the files aren't in focus order.

**DNG output** (`output.rs`): LinearRaw, 3 samples, Deflate tiles; values scaled below 1.0 with the
scale in `BaselineExposure`; raw sources keep their DNG colour tags (camera RGB), other sources get
`ColorMatrix1`/`ForwardMatrix1` of their RGB space.

## Measured (synthetic tests, `cargo test -p lightcraft-merge`)

- Focus stack speed on Helicon Focus's terms (100 frames of 5616 × 3744 in memory, MPixel/s):
  `cargo run --release -p lightcraft-merge --example focus_bench` (or `cargo xtask bench --focus`,
  which records it in `target/bench/history.jsonl`; `SIZE=1920x1280 FRAMES=30` for a small machine).
  Every frame-sized buffer is reused from frame to frame (fresh ones cost more in page faults than
  the arithmetic). Numbers that count are from an Apple M5 (#18); Helicon's own table: M5 753 MPixel/s
  for method B.
- Focus stack, 12 frames 640 × 420 (defocus 1.2 px per frame, a front disc as a depth step, breathing,
  drift, noise): best single frame 25.7 dB PSNR; A 32.6 dB, B 34.7 dB, C 35.2 dB; B's depth error
  0.1 frames where confident; alignment ≤ 0.21 px with 0.4 %/frame breathing, 1.5 px drift, 0.15°
  rotation; C's brightness within 1.3 %; the halo around the depth step within 1.5 × that of an oracle
  taking each pixel from the frame focused at its true depth.

- HDR, 3 brackets ±2 EV with 14-bit noise and handheld shifts: alignment error < 0.3 px, exposure
  error < 0.02 EV, radiance median log error < 0.03 (noise-limited, bias < 0.01), recovered
  highlights < 0.05; moving object: ghost error 0.2–2.0 without deghosting, < 0.06 with High.
- Panorama, 5 views over 182° with a 1.3× exposure drift: relative rotations < 0.1°, focal < 1 %,
  BA RMS ≈ 0.35 px; median log error vs ground truth < 0.02; no seam steps.

## Limits (known gaps)

- No lens-distortion model in the panorama bundle adjustment (rectilinear lenses assumed); no
  360° wrap-around; flat (translating-camera) scans are fitted with the rotation model.
- No camera response calibration for JPEG brackets (sRGB decoding is taken as linear light).
- Boundary Warp is a separable stretch, not a content-preserving mesh warp; Fill Edges is diffusion,
  not patch synthesis.
- *Create Stack*: the result goes on top of a collapsed stack of its sources (`Catalog::stack_with_ops`); its
  caption also lists them.
- Panorama output is limited to `maxMegapixels` (default 40) — blending memory is ~25 bytes/pixel.
- The browser build has no file access for merges yet.
- Focus stack: CPU only so far (GPU kernels to come); its speed against Helicon Focus has to be measured
  on a Mac (`#18`). No retouching between methods/frames after the merge yet.
