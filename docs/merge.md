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
  selected. Every merge makes a new photo, also when the same photos are merged again with the same
  options (each DNG has its own `RawDataUniqueID`).
- **Brackets as stacks** (#24): *Photo ▸ Stack ▸ Auto-Stack Focus Brackets…* (or *Settings ▸ Import ▸
  Stack focus brackets on import*) makes each focus bracket a collapsed stack, marked on its badge. A
  selected collapsed stack stands for all its photos: one click on its cell and *Focus Stack…* merges
  the whole bracket ("12 photos (from a stack)"), several selected stacks are focus-stacked (or
  HDR-merged) one by one ("3 stacks, merged one by one"), and the result goes on top of its stack.
  Adjust the bracket first (Sync, Paste, presets and Auto Sync also reach the whole collapsed stack):
  the result takes the settings (below, *Adjust first, then merge*).
- **Commands** (CLI, MCP, control channel):
  - `merge.hdr {ids?, align=true, deghost=none|low|medium|high, autoSettings, sourceSettings=true, stack, preview=false, showOverlay=false, previewPath?}`
  - `merge.panorama {ids?, projection=auto|spherical|cylindrical|perspective, boundaryWarp=0..100, autoCrop=false, fillEdges=false, autoSettings, sourceSettings=true, stack, maxMegapixels=40, preview, previewPath?}`
  - `merge.hdrPanorama {ids?, bracket=0 (from EXIF), …both sets of options}`
  - `merge.focusStack {ids?, method=depth|average|pyramid (or b|a|c), radius=1..30 (8), smoothing=0..10 (4), align=true, alignShift, alignRotation, alignScale, alignBrightness, order=capture|selection|reverse|auto, every=1, reference=middle|first|sharpest, crop=common|reference, maxMegapixels=0, depthPath?, autoSettings, sourceSettings=true, stack, preview, showDepth, previewPath?}`
  - Without `ids` they take the selection, a photo of a collapsed stack standing for the stack's
    photos (merge results, virtual copies and videos left out, so a stack topped by an earlier merge
    is merged again from its frames); `expandStacks: true` does the same for explicit `ids`. Two or
    more collapsed stacks and nothing else: `merge.focusStack` and `merge.hdr` merge each on its own
    and answer `{results, failed, count, perStack: true}` (a preview shows the first, `stacks: n`).
  - `stack` defaults to on when the photos are all in one stack: the result goes on its top, the
    stack's order kept. `autoSettings` defaults to on unless the result takes edited photos'
    settings; `sourceSettings: false` gives the old as-shot + Auto look.
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
  half the strongest focus within 4 × Radius) decides which depths are trusted. Weakly trusted
  depths give way to a depth spread from the trusted ones along the picture and stopped by its edges
  (a normalised convolution with the domain transform, on a quarter-size grid, upsampled
  edge-aware): a smooth subject — a plastic face, a petal — measures little more than noise, so its
  per-pixel winners are random frames; it takes the depth of its own details and outline instead,
  not an average of random frames or of what lies behind it. Then the depth map is smoothed by a
  confidence-weighted guided filter with the winners' grey image as guide (radius 2 × Smoothing). Colours blend the two frames nearest the
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
   gain, its focus measured, folded into the method's state. With a GPU, methods A and B do this on
   the device (`crates/gpu/src/focus.rs`), their state staying there until the end of the pass; a
   device error repeats the pass on the CPU. `info.device` says where it ran.
4. *Gather* (method B, only when needed): colours for the pixels whose smoothed depth left the frames
   the fuse pass kept (`info.gathered`: their share), from frames at half size. Those are the
   low-confidence pixels, with little detail of their own, and a raw at half size is binned instead of
   demosaiced. Real brackets gather most pixels (88–91 % on two 60-frame ARW brackets); there full
   size looked the same at 100 % and took 2.4–2.6 × as long (below). Synthetic bracket: B 34.76 →
   34.65 dB, halo rings unchanged within 2 %.

**Output:** a 16-bit linear DNG `<first photo in focus order>-Focus.dng` with the reference frame's colour
tags (raw sources stay camera RGB, so white balance and the rest stay editable). The depth map goes
into the DNG as a DNG 1.6 semantic mask (`urn:app:lightcraft:focus:depth`, 16-bit, ≤ 4096 px; 0 = the
first photo of the focus order) unless `depthMap: false`: Masking ▸ Depth (a Depth Range mask) selects
on it, e.g. to darken or blur what lies behind the subject. Optionally also as a 16-bit grey PNG
(`depthPath`). `info` reports the method, focus order, used and
skipped photos, the reference, alignment ranges (shift, rotation, scale, lowest NCC), warnings, where
the fuse pass ran (`device`), the share of gathered pixels and the seconds each pass took
(`LIGHTCRAFT_PROFILE=1` also prints them).

**Many brackets.** `merge.detectFocusBrackets {ids?, gapSecs=2, minFrames=3, sameExposure=true, stack}`
finds focus brackets (runs of photos at most `gapSecs` apart with the same camera, lens, focal length
and exposure; merge results and virtual copies are left out) in the given photos, the selection (two or
more cells) or the photos shown — the hidden members of collapsed stacks included — and with
`stack: true` makes each a collapsed stack of its own (`kind: focusBracket`, catalog format 5): its
photos leave the stacks they were in (a capture-time stack holding two brackets splits in two), a
stack that already is the bracket (merge results on top allowed) is kept, and the whole run is one
undo step (`{groups, count, photos, stacked, unchanged}`). `library.import {stackFocusBrackets}` (by
default the library preference) does it for the photos an import brings in.
`merge.focusStackBatch {groups? | …}` stacks one bracket after the other (a bracket that fails is
reported, the others still run): without `groups`, each selected collapsed stack as it is — with no
selection, each focus-bracket stack shown — and the brackets found among the other photos. In the app,
Photo Merge ▸ Focus Stack Each Bracket queues them in the background with the last focus-stack options.

**Retouching.** Every stack saves what it was made from beside its DNG, `<dng>.lcfocus.json`: the frames
(paths, focus order), the options, each frame's map onto the result and its gain, the crop. Photo Merge ▸
Retouch Focus Stack… (or `merge.focusRetouch`) paints over the stack with pixels from one aligned frame
(*Frame*), from the stack redone with method A, B or C, or from the stack as merged (an eraser), with a
round brush (size as a share of the long edge, hardness, opacity). The dialog previews at ≤ 1600 px
with the photo's settings (a method's source is restacked at that size with the saved geometry — no new
alignment — the first time a stroke uses it); Apply redoes it at full size in the background and writes
the DNG over the stack's own file, keeping the un-retouched one as `<dng>.lcfocus-base`, so strokes can be
changed or removed later (`strokes: []` restores the stack as merged). `merge.focusProject` lists the
frames and strokes. Painting happens on the whole picture (no zoom yet).

**For agents:** `merge.focusStack {ids, preview: true, previewPath, every: 4}` for a quick look (every 4th
photo, ≤ 1024 px), compare `method: "a"|"b"|"c"`, `showDepth: true` to see the depth map, then run it
without `preview`. `order: "auto"` when the files aren't in focus order.

**DNG output** (`output.rs`): LinearRaw, 3 samples, Deflate tiles. HDR results are scaled below 1.0
with the scale in `BaselineExposure`; 16-bit results (panoramas, focus stacks) keep the sources' white
level, so clipped highlights are rebuilt as in the sources. Raw sources keep their DNG colour tags
(camera RGB); a raw without colour matrices (Sony ARW, Nikon NEF, Fujifilm RAF…) also gets its as-shot
white balance (`AsShotNeutral`) and the generic camera model LightCraft falls back on
(`ColorMatrix1`/`ForwardMatrix1`, `frame::camera_color`), and the look LightCraft fits to its camera JPEG
travels in `DNGPrivateData` (`LightCraft`, JSON: matrix, tone and chroma curve, hue/saturation table),
so the merge opens looking like its reference photo. Other sources get `ColorMatrix1`/`ForwardMatrix1`
of their RGB space. The reference photo's embedded lens corrections (`WarpRectilinear`,
`FixVignetteRadial` in `OpcodeList3`) go into HDR and focus-stack DNGs (below), and every merged DNG
gets a `RawDataUniqueID` of its own.

## Adjust first, then merge (#24)

When the photos were edited, the result starts from the settings of its reference photo — the HDR's
middle exposure, the focus stack's alignment reference, the panorama's first photo; when that one is
unedited, the first edited photo — instead of the as-shot white balance and Auto Settings
(`crates/engine/src/merge_look.rs`; `info.settingsFrom` names the photo; the preview shows them too):

- **Taken:** profile, treatment, white balance, light, tone curve, colour, colour mixer and point
  colour, colour grading, effects, vignette, grain, detail (sharpening and noise reduction), optics
  (lens corrections, chromatic aberration, defringe, manual distortion and vignetting) and calibration,
  with the on/off state of their panel sections. A panorama keeps its own lens profile, distortion and
  vignetting (its picture is a projection of all the frames).
- **Left behind:** crop and orientation, Upright and transforms, masks, spot removal, red eye and lens
  blur (the result is aligned and cut differently), and Enhance — AI denoise, raw details, super
  resolution work on raw sensor data the LinearRaw DNG no longer has. Nothing is applied to the frames
  before merging.
- **White balance:** a DNG source's absolute white balance is copied (the result carries the reference
  frame's colour tags). Most vendor raws (ARW, NEF, RW2, RAF, CR2/CR3, PEF, SRW) and rendered files keep
  white balance relative to their as-shot white, while the merged DNG has an absolute one; such a
  setting becomes the Custom white that makes the same correction from the result's as-shot white
  (`lightcraft_color::transfer_white`: cone responses LMS(result as-shot) · LMS(set) / LMS(as-shot),
  the same matrix where both render with Bradford adaptation; on the ARW corpus the adjusted result
  renders within 3/255 of its source). As Shot stays the result's own as-shot.
- **Lens corrections:** the reference photo's embedded corrections are written into the DNG — an HDR
  as they are (same geometry), a focus stack moved with its cut of the reference frame
  (`pipeline::optics::crop_lens`, `files::lens_opcodes`: the inverse of reading them, coefficients
  re-scaled to the opcodes' normalisation radius) — so the result opens with Lens Corrections on like
  its sources, and retouching keeps them. A panorama gets none.
- Auto Settings is off by default then (on top it would undo the exposure you chose); unedited photos
  give the old look (as-shot white balance, Auto Settings). `sourceSettings: false` asks for that too.

## Measured (synthetic tests, `cargo test -p lightcraft-merge`)

- Focus stack speed on Helicon Focus's terms (100 frames of 5616 × 3744 in memory, MPixel/s):
  `cargo run --release -p lightcraft-engine --example focus_bench` (or `cargo xtask bench --focus`,
  which records it in `target/bench/history.jsonl`; `SIZE=1920x1280 FRAMES=30` for a small machine).
  Every frame-sized buffer is reused from frame to frame (fresh ones cost more in page faults than
  the arithmetic). Numbers that count are from an Apple M5 (below); Helicon's own table: M5 753 MPixel/s
  for method B.
- Focus stack, 12 frames 640 × 420 (defocus 1.2 px per frame, a front disc as a depth step, breathing,
  drift, noise): best single frame 25.7 dB PSNR; A 32.6 dB, B 34.7 dB, C 35.2 dB; B's depth error
  0.1 frames where confident; alignment ≤ 0.21 px with 0.4 %/frame breathing, 1.5 px drift, 0.15°
  rotation; C's brightness within 1.3 %; the halo around the depth step within 1.5 × that of an oracle
  taking each pixel from the frame focused at its true depth.
- Focus stack, a smooth, faintly grainy face in focus in the first of 20 frames, a textured background
  behind it: B puts the face at 0.03 frames (it was 4.4, 85 % of it more than 1.5 frames off, with
  worm-like seams), its error 1.06 × that of the sharp frame (was 2.7 ×).

- HDR, 3 brackets ±2 EV with 14-bit noise and handheld shifts: alignment error < 0.3 px, exposure
  error < 0.02 EV, radiance median log error < 0.03 (noise-limited, bias < 0.01), recovered
  highlights < 0.05; moving object: ghost error 0.2–2.0 without deghosting, < 0.06 with High.
- Panorama, 5 views over 182° with a 1.3× exposure drift: relative rotations < 0.1°, focal < 1 %,
  BA RMS ≈ 0.35 px; median log error vs ground truth < 0.02; no seam steps.

## Measured on an Apple M5 (#18)

- `focus_bench`, MPixel/s end to end / stacking only: B on the GPU 369 / 588 (Helicon Focus: 753 for
  method B, so 78 %), B on the CPU 224 / 291, A on the GPU 339 / 478, A on the CPU 189 / 225, C 138 / 156.
- Two brackets of 60 ARW frames, 7008 × 4672 (Sony A7C II, 100 mm macro, a figurine; focus steps 4 and
  1), from the raw files: B 33.0–34.5 s (align 3.8–4.0 s, fuse 19.3–20.9 s on the GPU, gather
  8.0–8.2 s for 88–91 % of the pixels), 8.5–9.7 GB peak; C 34.9 s; A 24.4 s. Decoding the raws takes
  most of the fuse pass (the same work in memory, 100 frames of 21 MP, takes 3.6 s).
- Gathering at full size instead: 19.6–20.8 s, the whole stack 44.8–46.1 s, +1.7 GB. Rendered at
  100 %, face grain, petal texture and edges looked the same as with half-size frames; the 8-bit
  renders differ by 2.4 on average, as much in the flat background as in the petals.
- The figurine's smooth plastic face comes out as sharp as in its sharpest frame, without seams, in
  B and C alike (B was soft with worm-like seams before the edge-aware fill). What no frame has
  sharp stays soft: the step-4 bracket missed the flowers, the step-1 bracket has them.

## Limits (known gaps)

- No lens-distortion model in the panorama bundle adjustment (rectilinear lenses assumed); no
  360° wrap-around; flat (translating-camera) scans are fitted with the rotation model.
- No camera response calibration for JPEG brackets (sRGB decoding is taken as linear light).
- Boundary Warp is a separable stretch, not a content-preserving mesh warp; Fill Edges is diffusion,
  not patch synthesis.
- *Create Stack*: the result goes on top of the stack its sources share, the stack's order and
  collapsed state kept (`Catalog::push_top_ops`), else on top of a new collapsed stack of its sources
  (`Catalog::stack_with_ops`, which merges the stacks they were in); its caption also lists them.
- A white balance carried over from rendered photos (JPEG, PNG, TIFF: relative) to their merge, which
  re-balances in its own colour space rather than with Bradford, is close but not exact.
- Panorama output is limited to `maxMegapixels` (default 40) — blending memory is ~25 bytes/pixel.
- The browser build has no file access for merges yet.
- Focus stack: method C, the depth refinement and the gather pass run on the CPU, and raws are
  demosaiced on the CPU, which bounds a stack from raw files more than the stacking does. Retouching
  paints on a ≤ 1600 px view of the whole picture (no zoom); a retouched stack's depth map is the
  merged one.
