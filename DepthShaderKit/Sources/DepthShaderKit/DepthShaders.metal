#include <metal_stdlib>
using namespace metal;

kernel void flip_horizontal(texture2d<float, access::read> src [[texture(0)]],
                            texture2d<float, access::write> dst [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint w = src.get_width();
    uint h = src.get_height();
    if (gid.x >= w || gid.y >= h) return;
    dst.write(src.read(uint2(w - 1 - gid.x, gid.y)), gid);
}

struct AffineParams {
    float m00, m01, m10, m11;
    float bx, by;
    float invScale, dz, threshold;
    uint ox, oy;  // dispatch origin for tile-region renders (0,0 for full canvas)
};

// Weight snap threshold: inverse-mapped coordinates within this distance of an
// integer are treated as exact pixel hits, keeping right-angle rotations bit-exact.
constant float kSnap = 1e-3f;

kernel void affine_transform(texture2d<float, access::read> src [[texture(0)]],
                             texture2d<float, access::write> dst [[texture(1)]],
                             texture2d<float, access::write> mask [[texture(2)]],
                             constant AffineParams &p [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint tw = dst.get_width();
    uint th = dst.get_height();
    uint2 pos = gid + uint2(p.ox, p.oy);
    if (pos.x >= tw || pos.y >= th) return;
    int sw = int(src.get_width());
    int sh = int(src.get_height());

    float fx = fma(p.m00, float(pos.x), fma(p.m01, float(pos.y), p.bx));
    float fy = fma(p.m10, float(pos.x), fma(p.m11, float(pos.y), p.by));

    bool valid = fx > -0.5f && fx < float(sw) - 0.5f && fy > -0.5f && fy < float(sh) - 0.5f;
    mask.write(valid ? 1.0f : 0.0f, pos);
    if (!valid) {
        dst.write(0.0f, pos);
        return;
    }

    int x0 = clamp(int(floor(fx)), 0, sw - 1);
    int x1 = clamp(x0 + 1, 0, sw - 1);
    int y0 = clamp(int(floor(fy)), 0, sh - 1);
    int y1 = clamp(y0 + 1, 0, sh - 1);

    float a = src.read(uint2(x0, y0)).r;
    float b = src.read(uint2(x1, y0)).r;
    float c = src.read(uint2(x0, y1)).r;
    float d = src.read(uint2(x1, y1)).r;

    float spread = max(max(a, b), max(c, d)) - min(min(a, b), min(c, d));
    float v;
    if (spread > p.threshold) {
        int nx = clamp(int(floor(fx + 0.5f)), 0, sw - 1);
        int ny = clamp(int(floor(fy + 0.5f)), 0, sh - 1);
        v = src.read(uint2(nx, ny)).r;
    } else {
        float wx = fx - floor(fx);
        float wy = fy - floor(fy);
        wx = wx < kSnap ? 0.0f : (wx > 1.0f - kSnap ? 1.0f : wx);
        wy = wy < kSnap ? 0.0f : (wy > 1.0f - kSnap ? 1.0f : wy);
        v = a * (1.0f - wx) * (1.0f - wy) + b * wx * (1.0f - wy)
          + c * (1.0f - wx) * wy + d * wx * wy;
    }
    dst.write(fma(v, p.invScale, p.dz), pos);
}

struct ResizeParams {
    float sx;
    float sy;
};

kernel void resize_bilinear(texture2d<float, access::read> src [[texture(0)]],
                            texture2d<float, access::write> dst [[texture(1)]],
                            constant ResizeParams &params [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint tw = dst.get_width();
    uint th = dst.get_height();
    if (gid.x >= tw || gid.y >= th) return;
    int w = int(src.get_width());
    int h = int(src.get_height());

    float fy = (float(gid.y) + 0.5f) * params.sy - 0.5f;
    int y0 = max(0, int(floor(fy)));
    int y1 = min(h - 1, y0 + 1);
    float wy = fy - float(y0);

    float fx = (float(gid.x) + 0.5f) * params.sx - 0.5f;
    int x0 = max(0, int(floor(fx)));
    int x1 = min(w - 1, x0 + 1);
    float wx = fx - float(x0);

    float a = src.read(uint2(x0, y0)).r;
    float b = src.read(uint2(x1, y0)).r;
    float c = src.read(uint2(x0, y1)).r;
    float d = src.read(uint2(x1, y1)).r;

    dst.write(a * (1.0f - wx) * (1.0f - wy) + b * wx * (1.0f - wy)
              + c * (1.0f - wx) * wy + d * wx * wy, gid);
}

struct FuseParams {
    float blendThreshold;
    float seamThreshold;
    uint ox, oy;  // dispatch origin for tile-region renders
};

// Two-layer z-buffer composite: smaller depth = nearer. Blend band: when both layers are
// valid and |dA - dB| < blendThreshold, the nearer layer gets weight 0.5 + delta/(2T)
// (equal blend at delta=0, continuous into the hard pick at delta=T).
kernel void fuse_layers(texture2d<float, access::read> depthA [[texture(0)]],
                        texture2d<float, access::read> maskA [[texture(1)]],
                        texture2d<float, access::read> depthB [[texture(2)]],
                        texture2d<float, access::read> maskB [[texture(3)]],
                        texture2d<float, access::write> outDepth [[texture(4)]],
                        texture2d<float, access::write> outMask [[texture(5)]],
                        texture2d<float, access::write> outWinner [[texture(6)]],
                        texture2d<float, access::write> outSeam [[texture(7)]],
                        constant FuseParams &p [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint2 pos = gid + uint2(p.ox, p.oy);
    if (pos.x >= outDepth.get_width() || pos.y >= outDepth.get_height()) return;
    bool va = maskA.read(pos).r > 0.5f;
    bool vb = maskB.read(pos).r > 0.5f;
    float d = 0.0f, m = 0.0f, w = -1.0f, s = 0.0f;
    if (va && !vb) {
        d = depthA.read(pos).r; m = 1.0f; w = 0.0f;
    } else if (vb && !va) {
        d = depthB.read(pos).r; m = 1.0f; w = 1.0f;
    } else if (va && vb) {
        float da = depthA.read(pos).r;
        float db = depthB.read(pos).r;
        float delta = abs(da - db);
        bool nearA = da <= db;
        float dn = nearA ? da : db;
        float df = nearA ? db : da;
        m = 1.0f;
        w = nearA ? 0.0f : 1.0f;
        s = delta < p.seamThreshold ? 1.0f : 0.0f;
        d = dn;
        if (p.blendThreshold > 0.0f && delta < p.blendThreshold) {
            float wn = 0.5f + delta / (2.0f * p.blendThreshold);
            d = wn * dn + (1.0f - wn) * df;
        }
    }
    outDepth.write(d, pos);
    outMask.write(m, pos);
    outWinner.write(w, pos);
    outSeam.write(s, pos);
}

struct GuidedParams {
    float epsilon;
    float depthThreshold;
    int radius;
    int useDepthGuide;
    uint ox, oy;  // dispatch origin for tile-region renders
};

// Single-pass guided filter applied only inside the dilated seam band (any seam pixel
// within radius r). Outside the band the input is copied bit-exactly.
kernel void guided_smooth(texture2d<float, access::read> depthIn [[texture(0)]],
                          texture2d<float, access::read> maskIn [[texture(1)]],
                          texture2d<float, access::read> seamIn [[texture(2)]],
                          texture2d<float, access::read> guide [[texture(3)]],
                          texture2d<float, access::write> depthOut [[texture(4)]],
                          constant GuidedParams &p [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    int w = int(depthIn.get_width());
    int h = int(depthIn.get_height());
    uint2 pos = gid + uint2(p.ox, p.oy);
    if (int(pos.x) >= w || int(pos.y) >= h) return;
    float dc = depthIn.read(pos).r;
    if (maskIn.read(pos).r < 0.5f) {
        depthOut.write(dc, pos);
        return;
    }
    int r = p.radius;
    bool inBand = false;
    for (int dy = -r; dy <= r && !inBand; dy++) {
        for (int dx = -r; dx <= r && !inBand; dx++) {
            int2 q = clamp(int2(pos) + int2(dx, dy), int2(0), int2(w - 1, h - 1));
            inBand = seamIn.read(uint2(q)).r > 0.5f;
        }
    }
    if (!inBand) {
        depthOut.write(dc, pos);
        return;
    }
    float4 gc4 = guide.read(pos);
    float gc = p.useDepthGuide != 0 ? dc : dot(gc4.rgb, float3(0.299f, 0.587f, 0.114f));
    float n = 0.0f, sg = 0.0f, sp = 0.0f, sgg = 0.0f, sgp = 0.0f;
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int2 q = clamp(int2(pos) + int2(dx, dy), int2(0), int2(w - 1, h - 1));
            if (maskIn.read(uint2(q)).r < 0.5f) continue;
            float dn = depthIn.read(uint2(q)).r;
            if (abs(dn - dc) > p.depthThreshold) continue;
            float4 gn4 = guide.read(uint2(q));
            float gn = p.useDepthGuide != 0 ? dn : dot(gn4.rgb, float3(0.299f, 0.587f, 0.114f));
            n += 1.0f;
            sg += gn;
            sp += dn;
            sgg += gn * gn;
            sgp += gn * dn;
        }
    }
    if (n == 0.0f) {
        depthOut.write(dc, pos);
        return;
    }
    float meanG = sg / n;
    float meanP = sp / n;
    float varG = sgg / n - meanG * meanG;
    float cov = sgp / n - meanG * meanP;
    float a = cov / (varG + p.epsilon);
    float b = meanP - a * meanG;
    depthOut.write(fma(a, gc, b), pos);
}
