import XCTest
@testable import DepthShaderKit

// CPU reference for fuse_layers: identical hard pick + depth-weighted blend band.
func cpuFuse(depthA: [Float], maskA: [Float], depthB: [Float], maskB: [Float],
             blendThreshold: Float, seamThreshold: Float)
    -> (depth: [Float], mask: [Float], winner: [Float], seam: [Float]) {
    let n = depthA.count
    var depth = [Float](repeating: 0, count: n)
    var mask = [Float](repeating: 0, count: n)
    var winner = [Float](repeating: -1, count: n)
    var seam = [Float](repeating: 0, count: n)
    for i in 0..<n {
        let va = maskA[i] > 0.5, vb = maskB[i] > 0.5
        if va && !vb {
            depth[i] = depthA[i]; mask[i] = 1; winner[i] = 0
        } else if vb && !va {
            depth[i] = depthB[i]; mask[i] = 1; winner[i] = 1
        } else if va && vb {
            let da = depthA[i], db = depthB[i]
            let delta = abs(da - db)
            let nearA = da <= db
            let dn = nearA ? da : db
            let df = nearA ? db : da
            mask[i] = 1
            winner[i] = nearA ? 0 : 1
            seam[i] = delta < seamThreshold ? 1 : 0
            depth[i] = dn
            if blendThreshold > 0 && delta < blendThreshold {
                let wn = 0.5 + delta / (2 * blendThreshold)
                depth[i] = wn * dn + (1 - wn) * df
            }
        }
    }
    return (depth, mask, winner, seam)
}

// CPU reference for guided_smooth (single-pass guided filter, clamp-to-edge windows).
func cpuGuided(depth: [Float], mask: [Float], seam: [Float], guide: [Float]?,
               width w: Int, height h: Int, params: GuidedFilterParams) -> [Float] {
    var out = depth
    let r = params.radius
    func lum(_ i: Int) -> Float { guide?[i] ?? depth[i] }
    for y in 0..<h {
        for x in 0..<w {
            let i = y * w + x
            if mask[i] < 0.5 { continue }
            var inBand = false
            for dy in -r...r where !inBand {
                for dx in -r...r where !inBand {
                    let qx = min(max(x + dx, 0), w - 1), qy = min(max(y + dy, 0), h - 1)
                    inBand = seam[qy * w + qx] > 0.5
                }
            }
            if !inBand { continue }
            let dc = depth[i]
            var n: Float = 0, sg: Float = 0, sp: Float = 0, sgg: Float = 0, sgp: Float = 0
            for dy in -r...r {
                for dx in -r...r {
                    let qx = min(max(x + dx, 0), w - 1), qy = min(max(y + dy, 0), h - 1)
                    let j = qy * w + qx
                    if mask[j] < 0.5 { continue }
                    let dn = depth[j]
                    if abs(dn - dc) > params.depthThreshold { continue }
                    let gn = lum(j)
                    n += 1; sg += gn; sp += dn; sgg += gn * gn; sgp += gn * dn
                }
            }
            if n == 0 { continue }
            let meanG = sg / n, meanP = sp / n
            let varG = sgg / n - meanG * meanG
            let cov = sgp / n - meanG * meanP
            let a = cov / (varG + params.epsilon)
            let b = meanP - a * meanG
            out[i] = a * lum(i) + b
        }
    }
    return out
}

final class FuseTests: XCTestCase {
    let ops = DepthOps()
    let w = 48, h = 32

    /// Two layers on a common canvas: A = smooth ramp, B = shifted variant, each with a
    /// circular-ish valid mask so all overlap combinations occur.
    private func makeLayers() throws -> (a: DepthLayer, b: DepthLayer,
                                         da: [Float], ma: [Float], db: [Float], mb: [Float]) {
        var da = [Float](repeating: 0, count: w * h)
        var db = [Float](repeating: 0, count: w * h)
        var ma = [Float](repeating: 0, count: w * h)
        var mb = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                let i = y * w + x
                da[i] = 5 + Float(x) * 0.05 + sin(Float(y) * 0.1)
                db[i] = 5 + Float(x) * 0.05 + sin(Float(y) * 0.1) + (x < w / 2 ? -0.02 : 0.6)
                let ax = Float(x - 20), ay = Float(y - 16)
                ma[i] = ax * ax + ay * ay < 15 * 15 ? 1 : 0
                let bx = Float(x - 28), by = Float(y - 16)
                mb[i] = bx * bx + by * by < 15 * 15 ? 1 : 0
            }
        }
        return (DepthLayer(depth: try DepthTexture(values: da, width: w, height: h),
                           mask: try DepthTexture(values: ma, width: w, height: h)),
                DepthLayer(depth: try DepthTexture(values: db, width: w, height: h),
                           mask: try DepthTexture(values: mb, width: w, height: h)),
                da, ma, db, mb)
    }

    func testFuseHardPickBitExact() throws {
        let (a, b, da, ma, db, mb) = try makeLayers()
        let result = try ops.fuse(a, b, params: FuseParams(blendThreshold: 0, seamThreshold: 0.1))
        let cpu = cpuFuse(depthA: da, maskA: ma, depthB: db, maskB: mb,
                          blendThreshold: 0, seamThreshold: 0.1)
        XCTAssertEqual(result.depth.readback(), cpu.depth)
        XCTAssertEqual(result.mask.readback(), cpu.mask)
        XCTAssertEqual(result.winner.readback(), cpu.winner)
        XCTAssertEqual(result.seam.readback(), cpu.seam)
    }

    func testFuseOutsideBandBitExact() throws {
        // acceptance: pixels outside the transition band equal the winning layer's input
        let (a, b, da, ma, db, mb) = try makeLayers()
        let T: Float = 0.1
        let result = try ops.fuse(a, b, params: FuseParams(blendThreshold: T, seamThreshold: T))
        let gd = result.depth.readback(), gm = result.mask.readback(), gw = result.winner.readback()
        for i in 0..<w * h where gm[i] == 1 {
            let delta = (ma[i] == 1 && mb[i] == 1) ? abs(da[i] - db[i]) : .infinity
            if delta >= T {
                let expected = gw[i] == 0 ? da[i] : db[i]
                XCTAssertEqual(gd[i], expected, "pixel \(i) outside band must be the winner's input")
            }
        }
    }

    func testFuseBlendBandParity() throws {
        let (a, b, da, ma, db, mb) = try makeLayers()
        let T: Float = 0.1
        let result = try ops.fuse(a, b, params: FuseParams(blendThreshold: T, seamThreshold: T))
        let cpu = cpuFuse(depthA: da, maskA: ma, depthB: db, maskB: mb,
                          blendThreshold: T, seamThreshold: T)
        XCTAssertEqual(result.mask.readback(), cpu.mask)
        XCTAssertEqual(result.winner.readback(), cpu.winner)
        XCTAssertEqual(result.seam.readback(), cpu.seam)
        let maxDiff = zip(result.depth.readback(), cpu.depth).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(maxDiff, 1e-4, "blend band max abs diff \(maxDiff)")
        XCTAssertTrue(cpu.seam.contains(1), "test data must actually contain seam pixels")
    }

    func testFuseEmptyAndSingleSided() throws {
        let zeros = [Float](repeating: 0, count: w * h)
        let ones = [Float](repeating: 1, count: w * h)
        let depth = [Float](repeating: 3.5, count: w * h)
        let emptyMask = try DepthTexture(values: zeros, width: w, height: h)
        let fullMask = try DepthTexture(values: ones, width: w, height: h)
        let dtex = try DepthTexture(values: depth, width: w, height: h)
        let empty = DepthLayer(depth: dtex, mask: emptyMask)
        let full = DepthLayer(depth: dtex, mask: fullMask)

        let none = try ops.fuse(empty, empty)
        XCTAssertEqual(none.mask.readback(), zeros)
        XCTAssertEqual(none.depth.readback(), zeros)
        XCTAssertEqual(none.winner.readback(), [Float](repeating: -1, count: w * h))

        let aOnly = try ops.fuse(full, empty)
        XCTAssertEqual(aOnly.depth.readback(), depth)
        XCTAssertEqual(aOnly.winner.readback(), zeros)
        let bOnly = try ops.fuse(empty, full)
        XCTAssertEqual(bOnly.depth.readback(), depth)
        XCTAssertEqual(bOnly.winner.readback(), ones)
    }

    func testGuidedOutsideBandUntouched() throws {
        let (a, b, _, _, _, _) = try makeLayers()
        let fused = try ops.fuse(a, b, params: FuseParams(blendThreshold: 0.1, seamThreshold: 0.1))
        let smoothed = try ops.guidedSmooth(fused, params: GuidedFilterParams(radius: 2))
        let src = fused.depth.readback()
        let dst = smoothed.readback()
        let seam = fused.seam.readback()
        let mask = fused.mask.readback()
        let r = 2
        for y in 0..<h {
            for x in 0..<w {
                var inBand = false
                for dy in -r...r where !inBand {
                    for dx in -r...r where !inBand {
                        let qx = min(max(x + dx, 0), w - 1), qy = min(max(y + dy, 0), h - 1)
                        inBand = seam[qy * w + qx] > 0.5
                    }
                }
                if !inBand || mask[y * w + x] < 0.5 {
                    XCTAssertEqual(dst[y * w + x], src[y * w + x], "pixel (\(x),\(y)) outside band changed")
                }
            }
        }
    }

    func testGuidedParity() throws {
        let (a, b, da, ma, db, mb) = try makeLayers()
        let params = GuidedFilterParams(radius: 2, epsilon: 1e-3, depthThreshold: 2.0)
        let fused = try ops.fuse(a, b, params: FuseParams(blendThreshold: 0.1, seamThreshold: 0.1))
        let gpu = try ops.guidedSmooth(fused, params: params).readback()
        let cpu = cpuGuided(depth: fused.depth.readback(), mask: fused.mask.readback(),
                            seam: fused.seam.readback(), guide: nil,
                            width: w, height: h, params: params)
        let maxDiff = zip(gpu, cpu).map { abs($0 - $1) }.max() ?? 0
        // single-pass box guided filter; fast-math sum reordering on GPU vs CPU
        XCTAssertLessThan(maxDiff, 1e-3, "guided parity max abs diff \(maxDiff)")
        _ = (da, ma, db, mb)
    }

    func testGuidedRespectsDepthDiscontinuity() throws {
        // step edge fused content with a seam band running along the step: the guided pass
        // must not pull left-side (0) pixels toward the right side (10)
        let (w2, h2) = (32, 16)
        var depth = [Float](repeating: 0, count: w2 * h2)
        var mask = [Float](repeating: 1, count: w2 * h2)
        var seam = [Float](repeating: 0, count: w2 * h2)
        for y in 0..<h2 {
            for x in 0..<w2 {
                depth[y * w2 + x] = x < w2 / 2 ? 0 : 10
                seam[y * w2 + x] = (x == w2 / 2 || x == w2 / 2 - 1) ? 1 : 0
            }
        }
        let fused = FuseResult(depth: try DepthTexture(values: depth, width: w2, height: h2),
                               mask: try DepthTexture(values: mask, width: w2, height: h2),
                               winner: try DepthTexture(values: mask, width: w2, height: h2),
                               seam: try DepthTexture(values: seam, width: w2, height: h2))
        let out = try ops.guidedSmooth(fused, params: GuidedFilterParams(radius: 2, depthThreshold: 5))
            .readback()
        for y in 0..<h2 {
            for x in 0..<w2 {
                let v = out[y * w2 + x]
                if x < w2 / 2 {
                    XCTAssertLessThan(v, 5, "left side bled across the step at (\(x),\(y)): \(v)")
                } else {
                    XCTAssertGreaterThan(v, 5, "right side bled across the step at (\(x),\(y)): \(v)")
                }
            }
        }
    }
}
