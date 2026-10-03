import XCTest
@testable import DepthShaderKit

// CPU reference for the reprojection splat pass + hole fill: identical pinhole
// back-projection, yaw/pitch rotation about the pivot plane, splat rounding
// (pixel = floor(u' + 0.5)), strict-less z-buffer (first-drawn wins ties, matching
// the GPU compare-less depth test), and 8-neighborhood diffusion fill.
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

    var zbuf = [Float](repeating: .infinity, count: cw * ch)
    var depth = [Float](repeating: 0, count: cw * ch)
    var mask = [Float](repeating: 0, count: cw * ch)
    var color = [UInt8](repeating: 0, count: cw * ch * 4)

    for v in 0..<ch {
        for u in 0..<cw {
            let su = min(Int((Float(u) + 0.5) * Float(sw) / Float(cw)), sw - 1)
            let sv = min(Int((Float(v) + 0.5) * Float(sh) / Float(ch)), sh - 1)
            let d = src[sv * sw + su]
            // X right, Y up, Z out of screen (camera looks along -Z, so Z = -d)
            let X = (Float(u) - cx) * d / f
            let Y = -(Float(v) - cy) * d / f
            let Zc = pivotZ - d
            let X1 = X * cyw + Zc * syw
            let Z1 = -X * syw + Zc * cyw
            let Y1 = Y * cpt - Z1 * spt
            let Z2 = Y * spt + Z1 * cpt
            let dp = pivotZ - Z2           // depth after rotation
            guard dp > 1e-6, dp / zMax < 1 else { continue }
            let up = f * X1 / dp + cx
            let vp = cy - f * Y1 / dp
            let px = Int(floor(up + 0.5)), py = Int(floor(vp + 0.5))
            guard px >= 0, px < cw, py >= 0, py < ch else { continue }
            let i = py * cw + px
            if dp < zbuf[i] {
                zbuf[i] = dp
                depth[i] = dp
                mask[i] = 1
                let s = (v * cw + u) * 4
                color[i * 4] = rgba[s]
                color[i * 4 + 1] = rgba[s + 1]
                color[i * 4 + 2] = rgba[s + 2]
                color[i * 4 + 3] = 255
            }
        }
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
    // mask 0 at their core. Holes are carved with depth 0: Z' = 0 is culled by the
    // vertex guard, so those splats never land.
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
    }

    // reprojectView: requests beyond the 5-degree micro-step limit are applied
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
        XCTAssertEqual(r12.steps, 3)  // 12° -> 3 x 4°
        XCTAssertEqual(r12.appliedYawDeg, 12, accuracy: 1e-4)
        XCTAssertGreaterThan(r12.coverage, 0.8)

        let r45 = try ops.reprojectView(depth: tex, colorRGBA: color, canvasWidth: w,
                                        canvasHeight: h, params: ReprojectParams(yawDeg: 45))
        XCTAssertEqual(r45.appliedYawDeg, 30)  // clamped
        XCTAssertEqual(r45.steps, 6)

        let r3 = try ops.reprojectView(depth: tex, colorRGBA: color, canvasWidth: w,
                                       canvasHeight: h,
                                       params: ReprojectParams(yawDeg: 3, fillRadius: 0,
                                                               softenEdges: false))
        XCTAssertEqual(r3.steps, 1)
        // one step through reprojectView must match the plain single-step path
        let single = try ops.reproject(depth: tex, colorRGBA: color, canvasWidth: w,
                                       canvasHeight: h,
                                       params: ReprojectParams(yawDeg: 3, fillRadius: 0,
                                                               softenEdges: false))
        XCTAssertEqual(r3.depth.readback(), single.depth.readback())
    }
}
