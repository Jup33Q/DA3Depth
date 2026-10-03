import XCTest
@testable import DepthShaderKit

// CPU reference for the reprojection splat passes + hole fill: identical pinhole
// back-projection, yaw/pitch rotation about the pivot plane, adaptive disc splat
// footprint (ps = clamp(ceil(hypot(distR, distD)) , 1, 8); pixel q covered iff
// |q - u'| <= ps/2 euclidean), pass-A nearest-depth visibility, pass-B gaussian
// accumulation gated by dz <= depthBreak (sigma = depthBreak/3) in vertex order,
// pass-C normalize (mask = weight > 1e-3), 8-neighborhood diffusion fill, then
// pull-push pyramid fill (color: weighted-mean pull + bilinear push; depth: max
// pull + depth-layer-gated bilinear push) and depth-gated band smoothing.
// Kept in sync with Sources/depthshader-parity/main.swift.
func cpuReproject(depth src: [Float], width sw: Int, height sh: Int,
                  rgba: [UInt8], canvas cw: Int, _ ch: Int,
                  params: ReprojectParams) -> (depth: [Float], mask: [Float], rgba: [UInt8]) {
    let pivotZ = params.pivotZ ?? src.reduce(0, +) / Float(src.count)
    let zMax = max((src.max() ?? 1) * 2, pivotZ * 4, 1e-3)
    let f = params.focal ?? Float(max(cw, ch))
    let cx = Float(cw - 1) / 2, cy = Float(ch - 1) / 2
    let yaw = params.yawDeg * .pi / 180, pitch = params.pitchDeg * .pi / 180
    let cyw = cos(yaw), syw = sin(yaw), cpt = cos(pitch), spt = sin(pitch)
    let depthBreak = 0.05 * pivotZ
    let splatMax: Float = 8

    // (u', v', d') or nil when culled (behind camera / beyond clip far)
    func project(_ u: Int, _ v: Int, _ d: Float) -> (Float, Float, Float)? {
        let X = (Float(u) - cx) * d / f
        let Y = -(Float(v) - cy) * d / f
        let Zc = pivotZ - d
        let X1 = X * cyw + Zc * syw
        let Z1 = -X * syw + Zc * cyw
        let Y1 = Y * cpt - Z1 * spt
        let Z2 = Y * spt + Z1 * cpt
        let dp = pivotZ - Z2
        guard dp > 1e-6, dp / zMax < 1 else { return nil }
        return (f * X1 / dp + cx, cy - f * Y1 / dp, dp)
    }

    var splatU = [Float](repeating: 0, count: cw * ch)
    var splatV = [Float](repeating: 0, count: cw * ch)
    var splatD = [Float](repeating: 0, count: cw * ch)
    var splatPS = [Float](repeating: 0, count: cw * ch)
    var splatOK = [Bool](repeating: false, count: cw * ch)

    for v in 0..<ch {
        for u in 0..<cw {
            let su = min(Int((Float(u) + 0.5) * Float(sw) / Float(cw)), sw - 1)
            let sv = min(Int((Float(v) + 0.5) * Float(sh) / Float(ch)), sh - 1)
            let d = src[sv * sw + su]
            guard let (up, vp, dp) = project(u, v, d) else { continue }
            var distR: Float = 0, distD: Float = 0
            if u + 1 < cw {
                let nsu = min(Int((Float(u + 1) + 0.5) * Float(sw) / Float(cw)), sw - 1)
                let dn = src[sv * sw + nsu]
                if abs(dn - d) <= depthBreak, let q = project(u + 1, v, dn) {
                    distR = hypot(q.0 - up, q.1 - vp)
                }
            }
            if v + 1 < ch {
                let nsv = min(Int((Float(v + 1) + 0.5) * Float(sh) / Float(ch)), sh - 1)
                let dn = src[nsv * sw + su]
                if abs(dn - d) <= depthBreak, let q = project(u, v + 1, dn) {
                    distD = hypot(q.0 - up, q.1 - vp)
                }
            }
            let ps = min(max(ceil(hypot(distR, distD) - 1e-3), 1), splatMax)
            let i = v * cw + u
            splatU[i] = up; splatV[i] = vp; splatD[i] = dp; splatPS[i] = ps
            splatOK[i] = true
        }
    }

    // pass A: nearest depth per pixel over the disc footprints (hardware z-test)
    var za = [Float](repeating: .infinity, count: cw * ch)
    for v in 0..<ch {
        for u in 0..<cw {
            let i = v * cw + u
            guard splatOK[i] else { continue }
            let up = splatU[i], vp = splatV[i], dp = splatD[i], ps = splatPS[i]
            let half = ps / 2
            let qx0 = max(0, Int(ceil(up - half))), qx1 = min(cw - 1, Int(floor(up + half)))
            let qy0 = max(0, Int(ceil(vp - half))), qy1 = min(ch - 1, Int(floor(vp + half)))
            guard qx0 <= qx1, qy0 <= qy1 else { continue }
            for py in qy0...qy1 {
                for px in qx0...qx1 {
                    let r2 = (Float(px) - up) * (Float(px) - up)
                           + (Float(py) - vp) * (Float(py) - vp)
                    guard r2 <= half * half else { continue }
                    let j = py * cw + px
                    if dp < za[j] { za[j] = dp }
                }
            }
        }
    }

    // pass B: gated gaussian accumulation in vertex order (API-ordered blending)
    var sumW = [Float](repeating: 0, count: cw * ch)
    var sumD = [Float](repeating: 0, count: cw * ch)
    var sumC = [Float](repeating: 0, count: cw * ch * 3)
    let sigma = depthBreak / 3
    for v in 0..<ch {
        for u in 0..<cw {
            let i = v * cw + u
            guard splatOK[i] else { continue }
            let up = splatU[i], vp = splatV[i], dp = splatD[i], ps = splatPS[i]
            let half = ps / 2
            let qx0 = max(0, Int(ceil(up - half))), qx1 = min(cw - 1, Int(floor(up + half)))
            let qy0 = max(0, Int(ceil(vp - half))), qy1 = min(ch - 1, Int(floor(vp + half)))
            guard qx0 <= qx1, qy0 <= qy1 else { continue }
            let s = (v * cw + u) * 4
            for py in qy0...qy1 {
                for px in qx0...qx1 {
                    let r2 = (Float(px) - up) * (Float(px) - up)
                           + (Float(py) - vp) * (Float(py) - vp)
                    guard r2 <= half * half else { continue }
                    let j = py * cw + px
                    let dz = max(dp - za[j], 0)
                    guard dz <= depthBreak else { continue }
                    let w = exp(-4 * r2 / (half * half)) * exp(-0.5 * dz * dz / (sigma * sigma))
                    sumW[j] += w
                    sumD[j] += dp * w
                    sumC[j * 3] += Float(rgba[s]) * w
                    sumC[j * 3 + 1] += Float(rgba[s + 1]) * w
                    sumC[j * 3 + 2] += Float(rgba[s + 2]) * w
                }
            }
        }
    }

    // pass C: normalize
    var depth = [Float](repeating: 0, count: cw * ch)
    var mask = [Float](repeating: 0, count: cw * ch)
    var color = [UInt8](repeating: 0, count: cw * ch * 4)
    for i in 0..<cw * ch where sumW[i] > 1e-3 {
        mask[i] = 1
        depth[i] = sumD[i] / sumW[i]
        color[i * 4] = UInt8(clamping: Int((sumC[i * 3] / sumW[i]).rounded()))
        color[i * 4 + 1] = UInt8(clamping: Int((sumC[i * 3 + 1] / sumW[i]).rounded()))
        color[i * 4 + 2] = UInt8(clamping: Int((sumC[i * 3 + 2] / sumW[i]).rounded()))
        color[i * 4 + 3] = 255
    }

    for _ in 0..<max(0, params.fillRadius) {
        var nd = depth, nm = mask, nc = color
        for y in 0..<ch {
            for x in 0..<cw {
                let i = y * cw + x
                if mask[i] > 0.5 { continue }
                var ds: Float = 0, r: Float = 0, g: Float = 0, b: Float = 0, n: Float = 0
                for dy in -1...1 {
                    for dx in -1...1 where !(dx == 0 && dy == 0) {
                        let qx = min(max(x + dx, 0), cw - 1), qy = min(max(y + dy, 0), ch - 1)
                        let j = qy * cw + qx
                        if mask[j] > 0.5 {
                            ds += depth[j]; n += 1
                            r += Float(color[j * 4]); g += Float(color[j * 4 + 1]); b += Float(color[j * 4 + 2])
                        }
                    }
                }
                if n > 0 {
                    nd[i] = ds / n
                    nm[i] = 1
                    nc[i * 4] = UInt8(clamping: Int((r / n).rounded()))
                    nc[i * 4 + 1] = UInt8(clamping: Int((g / n).rounded()))
                    nc[i * 4 + 2] = UInt8(clamping: Int((b / n).rounded()))
                    nc[i * 4 + 3] = 255
                }
            }
        }
        depth = nd; mask = nm; color = nc
    }

    // pull-push pyramid fill over mask=0 pixels (mask itself untouched)
    var dims = [(cw, ch)]
    while dims.last!.0 > 1 || dims.last!.1 > 1 {
        let (w, h) = dims.last!
        dims.append((max(1, (w + 1) / 2), max(1, (h + 1) / 2)))
    }
    if dims.count > 1 {
        var dL = [depth], wL = [mask]
        var cL = [[Float]](repeating: [], count: 1)
        cL[0] = (0..<cw * ch).flatMap { i in
            [Float(color[i * 4]), Float(color[i * 4 + 1]), Float(color[i * 4 + 2])]
        }
        for l in 1..<dims.count {
            let (iw, ih) = dims[l - 1]
            let (ow, oh) = dims[l]
            var dO = [Float](repeating: 0, count: ow * oh)
            var wO = [Float](repeating: 0, count: ow * oh)
            var cO = [Float](repeating: 0, count: ow * oh * 3)
            for y in 0..<oh {
                for x in 0..<ow {
                    var r: Float = 0, g: Float = 0, b: Float = 0, dmax: Float = 0, n: Float = 0
                    for dy in 0..<2 {
                        for dx in 0..<2 {
                            let qx = min(x * 2 + dx, iw - 1), qy = min(y * 2 + dy, ih - 1)
                            let j = qy * iw + qx
                            if wL[l - 1][j] > 0.5 {
                                r += cL[l - 1][j * 3]; g += cL[l - 1][j * 3 + 1]; b += cL[l - 1][j * 3 + 2]
                                dmax = max(dmax, dL[l - 1][j])
                                n += 1
                            }
                        }
                    }
                    let i = y * ow + x
                    if n > 0 {
                        dO[i] = dmax; wO[i] = 1
                        cO[i * 3] = r / n; cO[i * 3 + 1] = g / n; cO[i * 3 + 2] = b / n
                    }
                }
            }
            dL.append(dO); wL.append(wO); cL.append(cO)
        }
        var fillD = dL[dims.count - 1], fillC = cL[dims.count - 1]
        for l in stride(from: dims.count - 2, through: 0, by: -1) {
            let (wf, hf) = dims[l]
            let (wc, hc) = dims[l + 1]
            var dO = dL[l]
            var cO = cL[l]
            for y in 0..<hf {
                for x in 0..<wf {
                    let i = y * wf + x
                    if wL[l][i] > 0.5 { continue }
                    // depth-layer-gated bilinear from the coarser filled level:
                    // only the farthest layer's taps participate (renormalized)
                    let fx = (Float(x) + 0.5) * Float(wc) / Float(wf) - 0.5
                    let fy = (Float(y) + 0.5) * Float(hc) / Float(hf) - 0.5
                    let x0 = Int(floor(fx)), y0 = Int(floor(fy))
                    let wx = fx - floor(fx), wy = fy - floor(fy)
                    let q00 = (min(max(y0, 0), hc - 1)) * wc + (min(max(x0, 0), wc - 1))
                    let q10 = (min(max(y0, 0), hc - 1)) * wc + (min(max(x0 + 1, 0), wc - 1))
                    let q01 = (min(max(y0 + 1, 0), hc - 1)) * wc + (min(max(x0, 0), wc - 1))
                    let q11 = (min(max(y0 + 1, 0), hc - 1)) * wc + (min(max(x0 + 1, 0), wc - 1))
                    let d00 = fillD[q00], d10 = fillD[q10], d01 = fillD[q01], d11 = fillD[q11]
                    let dmax = max(max(d00, d10), max(d01, d11))
                    var b00 = (1 - wx) * (1 - wy), b10 = wx * (1 - wy)
                    var b01 = (1 - wx) * wy, b11 = wx * wy
                    b00 *= d00 >= dmax - depthBreak ? 1 : 0
                    b10 *= d10 >= dmax - depthBreak ? 1 : 0
                    b01 *= d01 >= dmax - depthBreak ? 1 : 0
                    b11 *= d11 >= dmax - depthBreak ? 1 : 0
                    let wsum = b00 + b10 + b01 + b11
                    if wsum <= 0 {
                        dO[i] = dmax
                        cO[i * 3] = fillC[q00 * 3]; cO[i * 3 + 1] = fillC[q00 * 3 + 1]
                        cO[i * 3 + 2] = fillC[q00 * 3 + 2]
                        continue
                    }
                    let inv = 1 / wsum
                    dO[i] = (b00 * d00 + b10 * d10 + b01 * d01 + b11 * d11) * inv
                    for c in 0..<3 {
                        cO[i * 3 + c] = (b00 * fillC[q00 * 3 + c] + b10 * fillC[q10 * 3 + c]
                                       + b01 * fillC[q01 * 3 + c] + b11 * fillC[q11 * 3 + c]) * inv
                    }
                }
            }
            fillD = dO; fillC = cO
        }
        depth = fillD
        for i in 0..<cw * ch where mask[i] < 0.5 {
            color[i * 4] = UInt8(clamping: Int(fillC[i * 3].rounded()))
            color[i * 4 + 1] = UInt8(clamping: Int(fillC[i * 3 + 1].rounded()))
            color[i * 4 + 2] = UInt8(clamping: Int(fillC[i * 3 + 2].rounded()))
            color[i * 4 + 3] = 255
        }
    }

    // inpainted-band smoothing: 2 iterations, mask=0 pixels only; depth via gated
    // 5x5 plane fit (depth-gradient = first-order normal information), color via
    // gated 3x3 gaussian
    for _ in 0..<2 where dims.count > 1 {
        var nd = depth, nc = color
        for y in 0..<ch {
            for x in 0..<cw {
                let i = y * cw + x
                if mask[i] > 0.5 { continue }
                let dc = depth[i]
                var r: Float = 0, g: Float = 0, b: Float = 0, cwsum: Float = 0
                for dy in -1...1 {
                    for dx in -1...1 {
                        let qx = min(max(x + dx, 0), cw - 1), qy = min(max(y + dy, 0), ch - 1)
                        let j = qy * cw + qx
                        let dn = depth[j]
                        if abs(dn - dc) > depthBreak { continue }
                        let wgt = Float((2 - abs(dx)) * (2 - abs(dy)))
                        cwsum += wgt
                        r += Float(color[j * 4]) * wgt
                        g += Float(color[j * 4 + 1]) * wgt
                        b += Float(color[j * 4 + 2]) * wgt
                    }
                }
                var sxx: Float = 0, sxy: Float = 0, sx: Float = 0
                var syy: Float = 0, sy: Float = 0, n: Float = 0
                var sxd: Float = 0, syd: Float = 0, sd: Float = 0
                var dmin = Float.greatestFiniteMagnitude, dmax = -Float.greatestFiniteMagnitude
                for dy in -2...2 {
                    for dx in -2...2 {
                        let qx = min(max(x + dx, 0), cw - 1), qy = min(max(y + dy, 0), ch - 1)
                        let dn = depth[qy * cw + qx]
                        if abs(dn - dc) > depthBreak { continue }
                        let wgt = Float((3 - abs(dx)) * (3 - abs(dy)))
                        let fx = Float(dx), fy = Float(dy)
                        sxx += wgt * fx * fx; sxy += wgt * fx * fy; sx += wgt * fx
                        syy += wgt * fy * fy; sy += wgt * fy; n += wgt
                        sxd += wgt * fx * dn; syd += wgt * fy * dn; sd += wgt * dn
                        dmin = min(dmin, dn); dmax = max(dmax, dn)
                    }
                }
                let m00 = sxx, m01 = sxy, m02 = sx
                let m11 = syy, m12 = sy, m22 = n
                let c01 = m01 * m22 - m02 * m12
                let c02 = m01 * m12 - m02 * m11
                let det = m00 * (m11 * m22 - m12 * m12) - m01 * c01 + m02 * c02
                if n >= 4 && abs(det) > 1e-3 * n * n * n {
                    let det3 = m00 * (m11 * sd - m12 * syd)
                             - m01 * (m01 * sd - m02 * syd)
                             + sxd * (m01 * m12 - m02 * m11)
                    nd[i] = min(max(det3 / det, dmin), dmax)
                } else if n > 0 {
                    nd[i] = sd / n
                }
                if cwsum > 0 {
                    nc[i * 4] = UInt8(clamping: Int((r / cwsum).rounded()))
                    nc[i * 4 + 1] = UInt8(clamping: Int((g / cwsum).rounded()))
                    nc[i * 4 + 2] = UInt8(clamping: Int((b / cwsum).rounded()))
                }
            }
        }
        depth = nd; color = nc
    }
    return (depth, mask, color)
}

/// Per-pixel comparison allowing the splat boundary to land ±1px off between GPU
/// rasterization and the CPU reference: a GPU pixel passes if its mask/depth match
/// the CPU value at the same pixel or any 8-neighbor. Returns the mismatch fraction
/// and the max same-pixel depth diff over pixels valid in both.
func compareReproject(gpu: (depth: [Float], mask: [Float]),
                      cpu: (depth: [Float], mask: [Float]),
                      width w: Int, height h: Int, depthTol: Float)
    -> (mismatchFraction: Double, maxDepthDiff: Double) {
    var mismatches = 0
    var maxDiff: Float = 0
    for y in 0..<h {
        for x in 0..<w {
            let i = y * w + x
            let gv = gpu.mask[i] > 0.5
            let cv = cpu.mask[i] > 0.5
            var ok = cv == gv && (!gv || abs(gpu.depth[i] - cpu.depth[i]) <= depthTol)
            if ok, gv { maxDiff = max(maxDiff, abs(gpu.depth[i] - cpu.depth[i])) }
            if !ok {
                // splat boundary allowance: a matching CPU pixel within ±1px
                for dy in -1...1 {
                    for dx in -1...1 where !(dx == 0 && dy == 0) {
                        let qx = x + dx, qy = y + dy
                        guard qx >= 0, qx < w, qy >= 0, qy < h else { continue }
                        let j = qy * w + qx
                        let cj = cpu.mask[j] > 0.5
                        if cj == gv && (!gv || abs(gpu.depth[i] - cpu.depth[j]) <= depthTol) { ok = true }
                    }
                }
            }
            if !ok { mismatches += 1 }
        }
    }
    return (Double(mismatches) / Double(w * h), Double(maxDiff))
}

func colorPattern(width w: Int, height h: Int) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: w * h * 4)
    for y in 0..<h {
        for x in 0..<w {
            let i = (y * w + x) * 4
            out[i] = UInt8(x * 255 / max(w - 1, 1))
            out[i + 1] = UInt8(y * 255 / max(h - 1, 1))
            out[i + 2] = UInt8((x + y) % 256)
            out[i + 3] = 255
        }
    }
    return out
}

final class ReprojectTests: XCTestCase {
    private let ops = DepthOps()

    // Smooth ramp depth, canvas 2x the depth resolution (exercises nearest upsample),
    // yaw 5°: GPU vs CPU per-pixel parity with ±1px splat boundary allowance.
    func testReprojectParityYaw5() throws {
        let (sw, sh, cw, ch) = (48, 32, 96, 64)
        var depth = [Float](repeating: 0, count: sw * sh)
        for y in 0..<sh {
            for x in 0..<sw {
                depth[y * sw + x] = 2 + Float(x) * 0.05 + sin(Float(y) * 0.2)
            }
        }
        let p = ReprojectParams(yawDeg: 5, fillRadius: 0)
        let gpu = try ops.reproject(values: depth, width: sw, height: sh,
                                    colorRGBA: colorPattern(width: cw, height: ch),
                                    canvasWidth: cw, canvasHeight: ch, params: p)
        let cpu = cpuReproject(depth: depth, width: sw, height: sh,
                               rgba: colorPattern(width: cw, height: ch),
                               canvas: cw, ch, params: p)
        let r = compareReproject(gpu: (gpu.depth, gpu.mask), cpu: (cpu.depth, cpu.mask),
                                 width: cw, height: ch, depthTol: 1e-3)
        // beyond the ±1px splat boundary, point-sprite rasterization tie noise at
        // silhouettes can leave a handful of unmatched pixels (15 of 6144 observed)
        XCTAssertLessThanOrEqual(r.mismatchFraction, 5e-3, "mask/depth alignment mismatch fraction")
        XCTAssertLessThanOrEqual(r.maxDepthDiff, 1e-3, "max depth diff on matched pixels")
        XCTAssertEqual(gpu.rgba.count, cw * ch * 4)
    }

    // Two depth planes (near left / far right) with yaw +5 and pitch -2, fill on:
    // parity plus occlusion sanity. The no-blend assertion runs on a fill-free pass:
    // with fill enabled the disocclusion band legitimately holds diffusion averages
    // of the two plane depths.
    func testReprojectParityTwoPlanes() throws {
        let (sw, sh) = (64, 48)
        var depth = [Float](repeating: 6, count: sw * sh)
        for y in 0..<sh {
            for x in 0..<(sw / 2) { depth[y * sw + x] = 2 }
        }
        let p = ReprojectParams(yawDeg: 5, pitchDeg: -2, fillRadius: 3)
        let gpu = try ops.reproject(values: depth, width: sw, height: sh,
                                    colorRGBA: colorPattern(width: sw, height: sh),
                                    canvasWidth: sw, canvasHeight: sh, params: p)
        let cpu = cpuReproject(depth: depth, width: sw, height: sh,
                               rgba: colorPattern(width: sw, height: sh),
                               canvas: sw, sh, params: p)
        let r = compareReproject(gpu: (gpu.depth, gpu.mask), cpu: (cpu.depth, cpu.mask),
                                 width: sw, height: sh, depthTol: 2e-2)
        XCTAssertLessThanOrEqual(r.mismatchFraction, 2e-2)

        let raw = try ops.reproject(values: depth, width: sw, height: sh,
                                    colorRGBA: colorPattern(width: sw, height: sh),
                                    canvasWidth: sw, canvasHeight: sh,
                                    params: ReprojectParams(yawDeg: 5, pitchDeg: -2, fillRadius: 0))
        for i in 0..<sw * sh where raw.mask[i] > 0.5 {
            let d = raw.depth[i]
            XCTAssert(min(abs(d - 2), abs(d - 6)) < 0.6,
                      "pixel \(i) depth \(d) blends front/back planes")
        }
    }

    // yaw = pitch = 0 reproduces the input (one splat per pixel, z-tie free).
    // Depth passes through (d - pivotZ) + pivotZ, so allow float round-trip error.
    func testReprojectIdentity() throws {
        let (w, h) = (40, 24)
        var depth = [Float](repeating: 0, count: w * h)
        for i in 0..<w * h { depth[i] = 1 + Float(i % 17) * 0.25 }
        let gpu = try ops.reproject(values: depth, width: w, height: h,
                                    colorRGBA: colorPattern(width: w, height: h),
                                    canvasWidth: w, canvasHeight: h,
                                    params: ReprojectParams(yawDeg: 0, fillRadius: 0))
        for i in 0..<w * h {
            XCTAssertEqual(gpu.depth[i], depth[i], accuracy: 1e-4)
        }
        XCTAssertTrue(gpu.mask.allSatisfy { $0 > 0.5 })
    }

    // Small holes (<= fill radius) get filled and marked valid; big holes keep
    // mask 0 at their core but are still inpainted by the pull-push fill (flat
    // source depth 3 -> the filled core reads exactly 3, color non-black).
    // Holes are carved with depth 0: Z' = 0 is culled by the vertex guard, so
    // those splats never land.
    func testHoleFillSmallVsBig() throws {
        let (w, h) = (64, 64)
        var depth = [Float](repeating: 3, count: w * h)
        for y in 30..<32 { for x in 30..<32 { depth[y * w + x] = 0 } }      // small 2x2 hole
        for y in 40..<60 { for x in 40..<60 { depth[y * w + x] = 0 } }      // big 20x20 hole
        let gpu = try ops.reproject(values: depth, width: w, height: h,
                                    colorRGBA: colorPattern(width: w, height: h),
                                    canvasWidth: w, canvasHeight: h,
                                    params: ReprojectParams(yawDeg: 0, fillRadius: 3))
        XCTAssertEqual(gpu.mask[31 * w + 31], 1, "small hole should be filled")
        XCTAssertEqual(gpu.mask[50 * w + 50], 0, "big hole core should stay masked out")
        XCTAssertEqual(gpu.mask[0], 1)
        XCTAssertEqual(gpu.depth[50 * w + 50], 3, accuracy: 1e-5,
                       "big hole core inpainted with surrounding background depth")
        XCTAssertGreaterThan(gpu.rgba[(50 * w + 50) * 4], 0,
                             "big hole core inpainted with non-black color")
    }

    // Pull-push fill: no zero-depth / pure-black pixels anywhere, mask=0 only marks
    // the disocclusion band; the filled depth matches the CPU reference bit-exactly
    // (fill propagates original values via max/nearest copies only).
    func testPullPushFillNoZeroHoles() throws {
        let (sw, sh, cw, ch) = (48, 32, 96, 64)
        var depth = [Float](repeating: 0, count: sw * sh)
        for y in 0..<sh {
            for x in 0..<sw {
                depth[y * sw + x] = 2 + Float(x) * 0.05 + sin(Float(y) * 0.2)
            }
        }
        let p = ReprojectParams(yawDeg: 5, fillRadius: 4)
        let gpu = try ops.reproject(values: depth, width: sw, height: sh,
                                    colorRGBA: colorPattern(width: cw, height: ch),
                                    canvasWidth: cw, canvasHeight: ch, params: p)
        for i in 0..<cw * ch {
            XCTAssertGreaterThan(gpu.depth[i], 0, "depth hole at \(i)")
        }
        var black = 0
        for i in 0..<cw * ch
        where gpu.rgba[i * 4] == 0 && gpu.rgba[i * 4 + 1] == 0 && gpu.rgba[i * 4 + 2] == 0 {
            black += 1
        }
        // the pattern itself contains a few genuine blacks; only the count is gated
        XCTAssertLessThanOrEqual(black, 32, "pull-push must not leave black holes")

        // exact fill-depth parity on the carved-hole identity case
        let (w, h) = (64, 64)
        var flat = [Float](repeating: 3, count: w * h)
        for y in 30..<32 { for x in 30..<32 { flat[y * w + x] = 0 } }
        for y in 40..<60 { for x in 40..<60 { flat[y * w + x] = 0 } }
        let fp = ReprojectParams(yawDeg: 0, fillRadius: 3, softenEdges: false)
        let gpuFlat = try ops.reproject(values: flat, width: w, height: h,
                                        colorRGBA: colorPattern(width: w, height: h),
                                        canvasWidth: w, canvasHeight: h, params: fp)
        let cpuFlat = cpuReproject(depth: flat, width: w, height: h,
                                   rgba: colorPattern(width: w, height: h),
                                   canvas: w, h, params: fp)
        // gaussian accumulation cancels to exactly 3 mathematically; allow ulp noise
        for i in 0..<w * h {
            XCTAssertEqual(gpuFlat.depth[i], cpuFlat.depth[i], accuracy: 1e-5,
                           "pull-push depth mismatch at \(i)")
        }
        XCTAssertEqual(gpuFlat.mask, cpuFlat.mask)
    }

    // reprojectView: requests beyond the 1-degree micro-step limit are applied
    // recursively as equal sub-steps; per-axis travel is clamped to ±30 degrees.
    func testReprojectViewRecursionAndClamp() throws {
        let (w, h) = (48, 32)
        var depth = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w { depth[y * w + x] = 2 + Float(x) * 0.05 + sin(Float(y) * 0.2) }
        }
        let tex = try DepthTexture(values: depth, width: w, height: h)
        let color = colorPattern(width: w, height: h)

        let r12 = try ops.reprojectView(depth: tex, colorRGBA: color, canvasWidth: w,
                                        canvasHeight: h, params: ReprojectParams(yawDeg: 12))
        XCTAssertEqual(r12.steps, 12)  // 12° -> 12 x 1°
        XCTAssertEqual(r12.appliedYawDeg, 12, accuracy: 1e-4)
        XCTAssertGreaterThan(r12.coverage, 0.8)

        let r45 = try ops.reprojectView(depth: tex, colorRGBA: color, canvasWidth: w,
                                        canvasHeight: h, params: ReprojectParams(yawDeg: 45))
        XCTAssertEqual(r45.appliedYawDeg, 30)  // clamped
        XCTAssertEqual(r45.steps, 30)

        let r08 = try ops.reprojectView(depth: tex, colorRGBA: color, canvasWidth: w,
                                        canvasHeight: h,
                                        params: ReprojectParams(yawDeg: 0.8, fillRadius: 0,
                                                                softenEdges: false))
        XCTAssertEqual(r08.steps, 1)
        // one step through reprojectView must match the plain single-step path
        let single = try ops.reproject(depth: tex, colorRGBA: color, canvasWidth: w,
                                       canvasHeight: h,
                                       params: ReprojectParams(yawDeg: 0.8, fillRadius: 0,
                                                               softenEdges: false))
        XCTAssertEqual(r08.depth.readback(), single.depth.readback())
    }
}
