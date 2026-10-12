// Focus stacking's fuse pass (`lightcraft_merge::focus`): resampling a frame into the output
// geometry, the focus measure, and the per-pixel updates of methods A and B, whose state stays on
// the device. Each kernel mirrors its CPU twin in `crates/merge/src/focus/` (named in the
// comment); keep them in step.
// Bindings: src (an input), a, b (planes), flags, fs (float state), cs (word state), frame (the
// frame uploaded: 16-bit samples, two a word, low half first).
// 1-D kernels: P[0] = pixel count.

const VALID: u32 = 1u;
const SATURATED: u32 = 2u;
const NONE: f32 = -1.0;
const CSCALE: f32 = 32768.0;

fn src_rgb(i: u32) -> vec3<f32> {
    return vec3<f32>(src[3u * i], src[3u * i + 1u], src[3u * i + 2u]);
}

fn maxc(c: vec3<f32>) -> f32 {
    return max(max(c.x, c.y), c.z);
}

// `catmull_rom` (measure.rs)
fn cr(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(0.5 * (-t3 + 2.0 * t2 - t), 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0), 0.5 * (-3.0 * t3 + 4.0 * t2 + t), 0.5 * (t3 - t2));
}

// Pixel `i` of the uploaded frame (fwarp's P[20]: the value of one 16-bit step).
fn frame_rgb(i: u32) -> vec3<f32> {
    let k = 3u * i;
    let w0 = frame[k >> 1u];
    let w1 = frame[(k >> 1u) + 1u];
    var q: vec3<u32>;
    if ((k & 1u) == 0u) {
        q = vec3<u32>(w0 & 0xffffu, w0 >> 16u, w1 & 0xffffu);
    } else {
        q = vec3<u32>(w0 >> 16u, w1 & 0xffffu, w1 >> 16u);
    }
    return vec3<f32>(q) * pf(20u);
}

// Pixel (x, y) of the `sw × sh` frame, clamped.
fn at(x: i32, y: i32, sw: i32, sh: i32) -> vec3<f32> {
    return frame_rgb(u32(clamp(y, 0, sh - 1) * sw + clamp(x, 0, sw - 1)));
}

// `sample` (measure.rs): Catmull–Rom or bilinear at continuous coordinates (centres at +0.5).
fn sample_src(x: f32, y: f32, sw: i32, sh: i32, bicubic: bool) -> vec3<f32> {
    let fx = x - 0.5;
    let fy = y - 0.5;
    let x0f = floor(fx);
    let y0f = floor(fy);
    let tx = fx - x0f;
    let ty = fy - y0f;
    let x0 = i32(x0f);
    let y0 = i32(y0f);
    if (!bicubic) {
        let a = at(x0, y0, sw, sh);
        let b = at(x0 + 1, y0, sw, sh);
        let c = at(x0, y0 + 1, sw, sh);
        let e = at(x0 + 1, y0 + 1, sw, sh);
        let top = a + (b - a) * tx;
        let bot = c + (e - c) * tx;
        return top + (bot - top) * ty;
    }
    let wx = cr(tx);
    let wy = cr(ty);
    var acc = vec3<f32>(0.0);
    for (var j = 0; j < 4; j++) {
        let yy = y0 - 1 + j;
        let row = wx.x * at(x0 - 1, yy, sw, sh) + wx.y * at(x0, yy, sw, sh) + wx.z * at(x0 + 1, yy, sw, sh) + wx.w * at(x0 + 2, yy, sw, sh);
        acc += wy[j] * row;
    }
    return max(acc, vec3<f32>(0.0));
}

// `warp` (measure.rs). P: 0 n, 1 ow, 2 ox, 3 oy, 4 sw, 5 sh, 6 bicubic, 7..16 h (f32, output →
// source), 16..19 inverse gains, 19 saturation level, 20 the frame's 16-bit step. frame in; a = rgb
// out; flags out.
@compute @workgroup_size(256)
fn fwarp(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let ow = pu(1u);
    let u = f32(pu(2u) + i % ow) + 0.5;
    let v = f32(pu(3u) + i / ow) + 0.5;
    let sw = i32(pu(4u));
    let sh = i32(pu(5u));
    var wz = pf(13u) * u + pf(14u) * v + pf(15u);
    if (abs(wz) < 1e-30) {
        wz = 1e-30;
    }
    let sx = (pf(7u) * u + pf(8u) * v + pf(9u)) / wz;
    let sy = (pf(10u) * u + pf(11u) * v + pf(12u)) / wz;
    if (!(sx >= 0.0 && sy >= 0.0 && sx <= f32(sw) && sy <= f32(sh))) {
        flags[i] = 0u;
        a[3u * i] = 0.0;
        a[3u * i + 1u] = 0.0;
        a[3u * i + 2u] = 0.0;
        return;
    }
    let p = sample_src(sx, sy, sw, sh, pu(6u) != 0u);
    let nx = min(i32(sx), sw - 1);
    let ny = min(i32(sy), sh - 1);
    flags[i] = select(VALID, VALID | SATURATED, maxc(at(nx, ny, sw, sh)) >= pf(19u));
    let gain = vec3<f32>(pf(16u), pf(17u), pf(18u));
    let o = p * gain;
    a[3u * i] = o.x;
    a[3u * i + 1u] = o.y;
    a[3u * i + 2u] = o.z;
}

// `grey` (measure.rs): √luminance. src = rgb; a = grey out.
@compute @workgroup_size(256)
fn fgrey(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let c = src_rgb(i);
    a[i] = sqrt(max(0.25 * c.x + 0.6 * c.y + 0.15 * c.z, 0.0));
}

// `modified_laplacian_into` (measure.rs). P: 0 n, 1 w, 2 h. src = grey; a = out; flags in.
@compute @workgroup_size(256)
fn fml(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    if (flags[i] != VALID) {
        a[i] = 0.0;
        return;
    }
    let w = pu(1u);
    let h = pu(2u);
    let x = i % w;
    let y = i / w;
    let c = src[i];
    var l = c;
    var r = c;
    var u = c;
    var d = c;
    if (x > 0u && (flags[i - 1u] & VALID) != 0u) {
        l = src[i - 1u];
    }
    if (x + 1u < w && (flags[i + 1u] & VALID) != 0u) {
        r = src[i + 1u];
    }
    if (y > 0u && (flags[i - w] & VALID) != 0u) {
        u = src[i - w];
    }
    if (y + 1u < h && (flags[i + w] & VALID) != 0u) {
        d = src[i + w];
    }
    a[i] = abs(2.0 * c - l - r) + abs(2.0 * c - u - d);
}

// Every `step`-th focus measure, for the frame's noise floor (`noise_floor`, measure.rs).
// P: 0 count, 1 step. src = fm; a = out.
@compute @workgroup_size(256)
fn fsub(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    a[i] = src[i * pu(1u)];
}

// Method A's weights before smoothing: fm² + 1e-10 (`Average::add`). src = fm; a = out.
@compute @workgroup_size(256)
fn favg_w(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let f = src[i];
    a[i] = f * f + 1e-10;
}

// Method A's state: zero. fs = sums (3n) | weights (n) | weighted positions (n).
@compute @workgroup_size(256)
fn favg_init(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    let n = pu(0u);
    if (i >= n) {
        return;
    }
    fs[3u * i] = 0.0;
    fs[3u * i + 1u] = 0.0;
    fs[3u * i + 2u] = 0.0;
    fs[3u * n + i] = 0.0;
    fs[4u * n + i] = 0.0;
}

// `Average::add`. P: 0 n, 1 position (f32). src = rgb; b = smoothed weights; flags in.
@compute @workgroup_size(256)
fn favg_add(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    let n = pu(0u);
    if (i >= n || (flags[i] & VALID) == 0u) {
        return;
    }
    let k = b[i];
    let c = src_rgb(i);
    fs[3u * i] = fs[3u * i] + k * c.x;
    fs[3u * i + 1u] = fs[3u * i + 1u] + k * c.y;
    fs[3u * i + 2u] = fs[3u * i + 2u] + k * c.z;
    fs[3u * n + i] = fs[3u * n + i] + k;
    fs[4u * n + i] = fs[4u * n + i] + k * pf(1u);
}

// Method B's state (`Depth::new`). fs = fm_best | fm_left | fm_right | fm_sum | prev_fm;
// cs = best | col_best | col_left | col_right | prev_col (colours: two words a pixel).
@compute @workgroup_size(256)
fn fdep_init(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    let n = pu(0u);
    if (i >= n) {
        return;
    }
    fs[i] = NONE;
    fs[n + i] = NONE;
    fs[2u * n + i] = NONE;
    fs[3u * n + i] = 0.0;
    fs[4u * n + i] = NONE;
    cs[i] = 0u;
    for (var k = 0u; k < 4u; k++) {
        cs[n + 2u * (k * n + i)] = 0u;
        cs[n + 2u * (k * n + i) + 1u] = 0u;
    }
}

// `enc` (depth.rs): linear 0..2 in 16 bits (round half up: values are ≥ 0).
fn enc1(v: f32) -> u32 {
    return u32(clamp(floor(v * CSCALE + 0.5), 0.0, 65535.0));
}

fn put_col(k: u32, n: u32, i: u32, lo: u32, hi: u32) {
    cs[n + 2u * (k * n + i)] = lo;
    cs[n + 2u * (k * n + i) + 1u] = hi;
}

// `Depth::add`. P: 0 n, 1 position. src = rgb; b = fm; flags in.
@compute @workgroup_size(256)
fn fdep_add(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    let n = pu(0u);
    if (i >= n) {
        return;
    }
    let k = pu(1u);
    if ((flags[i] & VALID) == 0u) {
        fs[4u * n + i] = NONE;
        return;
    }
    let f = b[i];
    let c = src_rgb(i);
    let lo = enc1(c.x) | (enc1(c.y) << 16u);
    let hi = enc1(c.z);
    fs[3u * n + i] = fs[3u * n + i] + f;
    let pfm = fs[4u * n + i];
    if (f > fs[i]) {
        if (pfm >= 0.0) {
            fs[n + i] = pfm;
            // col_left = prev_col
            put_col(1u, n, i, cs[n + 2u * (3u * n + i)], cs[n + 2u * (3u * n + i) + 1u]);
        } else {
            fs[n + i] = NONE;
        }
        cs[i] = k;
        fs[i] = f;
        put_col(0u, n, i, lo, hi);
        fs[2u * n + i] = NONE;
    } else if (k > 0u && cs[i] == k - 1u) {
        fs[2u * n + i] = f;
        put_col(2u, n, i, lo, hi);
    }
    fs[4u * n + i] = f;
    put_col(3u, n, i, lo, hi);
}
