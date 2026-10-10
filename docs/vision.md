# Apple Vision: subject masks, labels and aesthetics (macOS)

On a Mac, LightCraft uses Apple's **Vision** framework for three things: Select Subject / Background masks, labels
for what each photo shows (searchable), and an aesthetics score for culling. The models are part of macOS and run
on the Mac's Neural Engine: nothing to download, no weights for LightCraft to license, and the photos never leave
the machine.

| Feature | Vision request | Needs |
|---|---|---|
| Select Subject / Background | `VNGenerateForegroundInstanceMaskRequest` (every foreground instance) | macOS 14 |
| Labels (auto keywords for search) | `VNClassifyImageRequest` (~1,300 scene and object classes) | macOS 10.15 |
| Aesthetics score, utility images | `VNCalculateImageAestheticsScoresRequest` | macOS 15 |

Elsewhere (Linux, Windows, the web, older macOS) nothing changes: Subject and Background masks are estimated as
before (the photo's own mattes, else a heuristic) and the `ai.*` commands are disabled with the reason
(`ai.status` → `available: false`). Everything Vision produced is plain data in the library, so a library analyzed
on a Mac searches, sorts and renders the same on any platform.

**Status:** tested with stand-in models on Linux and checked on an Apple M5 (macOS 26.6, 2026-10-10) with 57 real
photos (JPEG, NEF, ARW, PNG, screenshots); numbers in [Measured on a Mac](#measured-on-a-mac).

## Select Subject / Background

Masking panel → **Subject** or **Background** (or `mask.add {kind: subject|background}`, `mask.addComponent`). Vision
looks at the photo as developed, uncropped and without masks at 1008 px (the frame mask coordinates live in, as for
SAM 3). Its matte is averaged onto a 512 × 512 grid over the uncropped photo and stored with the shape as logits
(`MaskShape::Subject { seg }`; `Background` stores the inverse), so the mask:

- renders the same in the preview and the export, at any size, and follows crop, rotation and geometry;
- stays what the user saw when they made it (it is not recomputed when the photo is edited later);
- renders on every platform (a mask made on a Mac shows the same subject on Linux).

In the app the mask appears at once with the estimate and Vision's subject replaces it when it answers (one undo step
brings the estimate back). From the CLI or MCP the command waits. When Vision finds no subject, or fails, the
estimate stays and the result (or a toast) says why. Sky has no Vision request and stays as it was.

## Labels and aesthetics: `ai.analyze`

Photo → **Analyze with Apple Vision** (or `ai.analyze`) analyzes the selected photos, else every photo in view;
photos analyzed before are skipped unless `force: true`. It runs on worker threads (half the cores, at most four);
the app shows "Analyzing photos: n of m" with Cancel and a toast at the end. Each photo is seen as the grid shows it
(its cached thumbnail render, edits and crop included; an unedited raw's embedded preview, which is much faster than
decoding the raw), upright and at 512 px.

What is stored per photo (`Photo.ai`; catalog format 4):

- **labels**: Vision's class identifiers (English, `_` between words: `dog`, `beach`, `sports_car`) with their
  confidence, most confident first, at most 16 from Vision. Only labels at Vision's high-precision operating point are
  kept (about 9 in 10 right; recall ≥ 1 % at 90 % precision), so an unusual photo can have none.
- **aesthetic**: Vision's overall score mapped from −1..1 to **0..100** (composition, exposure, focus, subject…).
- **utility**: a utility image (screenshot, receipt, document…) rather than a memorable photo. Vision gives utility
  images a score of 0, so they all land on **50**: rank photos with `utility:no`.

The results are LightCraft's own bookkeeping: not an undo step, not written to XMP sidecars, and they don't touch the
keyword list. To turn labels into real keywords (written to XMP as `dc:subject`, shown in the Keywords panel, read by
other apps), use Photo → **Add AI Labels as Keywords** (`ai.keywords`): the confident labels go under an `AI`
parent (`AI|dog`), one undo step. Re-run `ai.analyze` with `force` after big edits if the score should follow them.

### Finding photos by them

| Where | Syntax |
|---|---|
| Search (library filter text) | `ai:dog`, `ai:sports_car`; a bare word that is a whole label (`dog`, not `do`); `aesthetic:>70`, `aesthetic:<30`; `utility:yes`; `analyzed:no` |
| Smart albums / rule editor | Aesthetic Score (AI, 0–100) — number; What It Shows (AI labels) — like keywords; Utility Image — yes/no |
| Sort | View → Sort → Aesthetic Score (`library.sort {key: aesthetic}`; unscored photos last when best first) |
| `catalog.query` | each photo carries `aesthetic` and its top `aiLabels` |

## Commands

| Command | Params → result |
|---|---|
| `ai.analyze` | `{ids?, force?, wait?, budgetMs?}` → `{queued, skipped, done, total, pending, failed: [{id, error}], photos?: [{id, labels, aesthetic, utility}]}`. Waits by default outside the app (at most `budgetMs`; what is left keeps going and the next `ai.*` call collects it). |
| `ai.keywords` | `{ids?, minConfidence? (0.5), max? (5 per photo), parent? ("AI"; "" = top level), replace?}` → `{changed, added}`: labels become keywords, one undo step |
| `ai.status` | `{}` → progress plus `available`, `features: {labels, aesthetics, subject}` and the photos finished since the last report |
| `ai.cancel` | `{}` → `{cancelled}`: drops the photos still waiting (those being analyzed finish) |

## Agents: score, pick, edit, export

With MCP (`lightcraft-cli mcp --library ~/lc-lib`, or `--connect` to drive the app) every command is a tool:

```text
→ tools/call cmd_ai_analyze {}                                            # every photo in view; waits
← {"done": 412, "total": 412, "pending": 0, "failed": [], "photos": [{"id": 3, "labels": [{"name": "beach", …}], "aesthetic": 81.4, …}, …]}
→ tools/call cmd_catalog_query {"filter": {"text": "aesthetic:>0 utility:no"}, "sort": {"key": "aesthetic"}, "limit": 82}
← {"total": 405, "photos": [{"id": 3, "aesthetic": 81.4, "aiLabels": ["beach", "sky", …]}, …]}   # the top 20 %
→ tools/call cmd_preset_apply {"id": "lc.warm-glow", "ids": [3, 17, …]}
→ tools/call export {"ids": [3, 17, …], "dir": "/Users/me/Desktop/best"}
```

The same from a shell, in one process:

```sh
lightcraft-cli run --library ~/lc-lib ai.analyze budgetMs=600000 \
    catalog.query '{"filter": {"text": "ai:dog aesthetic:>60"}, "sort": {"key": "aesthetic"}, "limit": 20}'
```

For a large library, call `ai.analyze` with a `budgetMs` and repeat until `pending` is 0 (each call reports the photos
finished since the previous one).

## Faces and Vision

LightCraft's face features (`crates/faces`, [faces.md](faces.md)) stay as they are, on every platform: YuNet finds
faces and SFace / AuraFace make the embeddings that recognise people. Vision has face detection, landmarks and a face
capture-quality score, but no identity embeddings, so it could replace only the detector, which is neither slow nor
the weak part. Worth adding later: Vision's face capture quality for culling (the best shot of a person in a burst,
closed eyes), and its image feature prints for Find Similar Photos.

## Design

- `crates/vision` (L3) wraps the Objective-C calls in a safe API (`features`, `analyze`, `subject`). It is the third
  crate allowed `unsafe` (after `crates/sysmem` and `crates/mpsgraph`; see AGENTS.md), through the pure-Rust `objc2`
  bindings (no C, no Swift helper process). Objective-C exceptions cannot be caught without a C helper, so images are
  checked against their size before they become a `CGImage`, classes newer than the running macOS are looked up
  before they are created, and Vision's masks are read only after their pixel format, planarity, stride and data size
  are checked, under a read-only lock. Off macOS it is a stub that says so.
- `crates/engine/src/ai.rs`: the queue, the worker threads and the `ai.*` commands; the model is swappable
  (`AiState::analyzer`), which is how the tests run anywhere.
- `crates/engine/src/subject.rs`: Select Subject / Background, the matte-to-grid conversion and the background path.

## Measured on a Mac

Apple M5 (10 cores), macOS 26.6, release build:

| What | Time |
|---|---|
| Labels + aesthetics, one image (512–1,024 px) | 6–8 ms (screenshots 20 ms); the first call also loads the models |
| Subject mask, one image | 9–17 ms |
| `ai.analyze`, 57 photos (40 JPEG, 12 raws, 5 PNG), 4 workers | 1.73 s: **30 ms per photo**, renders included |

- Rankings matched a photographer's eye at both ends (sharp, clean close-ups on top; dark, cluttered night crowds at the
  bottom); `ai:sky` / `sky`, `utility:yes` and `ai.keywords` (105 keywords on 32 photos) worked as documented.
- Subject / Background masks: no fallback notes; exports at 800 and 3,000 px put the edge in the same place (IoU 0.94,
  the difference is resampling); fine feathers kept. In the app the estimate (52 % of the frame) was replaced by
  Vision's subject (37 %) and one undo brought the estimate back.
- Vision's limits seen: a high-key photo of a white peacock filling the frame had no subject; one photo had no label at
  the high-precision bar but a 93 aesthetics score.

## Testing on a Mac

```sh
cargo test -p lightcraft-vision -- --nocapture                         # Vision on a synthetic scene
# a real photo, upright as LightCraft gives it to Vision; a rotated one also prints the score of its stored pixels
LC_VISION_IMAGE=~/Pictures/some.jpg cargo test --release -p lightcraft-vision --lib real_photo -- --ignored --nocapture
cargo run --release -p lightcraft-cli -- run --import ~/Pictures/folder ai.analyze budgetMs=600000
```
