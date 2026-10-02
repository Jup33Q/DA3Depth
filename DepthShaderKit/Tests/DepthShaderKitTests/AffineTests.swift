import XCTest
@testable import DepthShaderKit

// CPU reference for the affine kernel: identical inverse map + depth-aware sampling.
// Coordinate math uses addingProduct in the same order as the kernel's explicit fma,
// so floor bins, mask decisions and nearest picks are bit-identical to the GPU.
func cpuAffine(_ values: [Float], width sw: Int, height sh: Int,
               transform: DepthAffineTransform, canvas tw: Int, _ th: Int) -> (depth: [Float], mask: [Float]) {
    let p = transform.params(sourceWidth: sw, sourceHeight: sh, canvasWidth: tw, canvasHeight: th)
    var depth = [Float](repeating: 0, count: tw * th)
    var mask = [Float](repeating: 0, count: tw * th)
    for y in 0..<th {
        for x in 0..<tw {
            let fx = p.bx.addingProduct(p.m01, Float(y)).addingProduct(p.m00, Float(x))
            let fy = p.by.addingProduct(p.m11, Float(y)).addingProduct(p.m10, Float(x))
            let i = y * tw + x
            guard fx > -0.5, fx < Float(sw) - 0.5, fy > -0.5, fy < Float(sh) - 0.5 else { continue }
            mask[i] = 1
            let x0 = min(max(Int(floor(fx)), 0), sw - 1)
            let x1 = min(max(x0 + 1, 0), sw - 1)
            let y0 = min(max(Int(floor(fy)), 0), sh - 1)
            let y1 = min(max(y0 + 1, 0), sh - 1)
            let a = values[y0 * sw + x0], b = values[y0 * sw + x1]
            let c = values[y1 * sw + x0], d = values[y1 * sw + x1]
            let spread = max(max(a, b), max(c, d)) - min(min(a, b), min(c, d))
            let v: Float
            if spread > p.threshold {
                let nx = min(max(Int(floor(fx + 0.5)), 0), sw - 1)
                let ny = min(max(Int(floor(fy + 0.5)), 0), sh - 1)
                v = values[ny * sw + nx]
            } else {
                var wx = fx - floor(fx)
                var wy = fy - floor(fy)
                if wx < 1e-3 { wx = 0 } else if wx > 1 - 1e-3 { wx = 1 }
                if wy < 1e-3 { wy = 0 } else if wy > 1 - 1e-3 { wy = 1 }
                v = a * (1 - wx) * (1 - wy) + b * wx * (1 - wy)
                  + c * (1 - wx) * wy + d * wx * wy
            }
            depth[i] = p.dz.addingProduct(v, p.invScale)
        }
    }
    return (depth, mask)
}

/// Smooth pattern (bounded gradient ~0.7/px) for 1e-4-tolerance parity.
func smoothPattern(width: Int, height: Int) -> [Float] {
    var out = [Float](repeating: 0, count: width * height)
    for y in 0..<height {
        for x in 0..<width {
            out[y * width + x] = sin(Float(x) * 0.05) * cos(Float(y) * 0.03) * 10
                               + Float(x + y) * 0.1
        }
    }
    return out
}

final class AffineTests: XCTestCase {
    let ops = DepthOps()

    private func runAffine(_ values: [Float], _ w: Int, _ h: Int,
                           _ t: DepthAffineTransform, _ tw: Int, _ th: Int) throws
        -> (gpu: (depth: [Float], mask: [Float]), cpu: (depth: [Float], mask: [Float])) {
        let gpu = try ops.affine(values: values, width: w, height: h,
                                 transform: t, canvasWidth: tw, canvasHeight: th)
        let cpu = cpuAffine(values, width: w, height: h, transform: t, canvas: tw, th)
        return (gpu, cpu)
    }

    func testRightAngleRotationsBitExact() throws {
        for (w, h) in [(5, 3), (16, 10), (7, 7)] {
            let values = pattern(width: w, height: h)  // noisy LCG pattern from M1 tests
            for (quarter, name) in [(1, "90"), (2, "180"), (3, "270")] {
                let rotated = quarter % 2 == 1
                let (tw, th) = rotated ? (h, w) : (w, h)
                for threshold in [Float.infinity, 0.5] {
                    let t = DepthAffineTransform(rotation: Float(quarter) * .pi / 2,
                                            scale: 1, zShift: 0, threshold: threshold)
                    let (gpu, cpu) = try runAffine(values, w, h, t, tw, th)
                    XCTAssertEqual(gpu.mask, cpu.mask,
                                   "\(name)° mask mismatch \(w)x\(h) threshold=\(threshold)")
                    XCTAssertEqual(gpu.depth, cpu.depth,
                                   "\(name)° depth mismatch \(w)x\(h) threshold=\(threshold)")
                    XCTAssertTrue(gpu.mask.allSatisfy { $0 == 1 },
                                  "\(name)° rotation should fully cover the canvas")
                }
            }
        }
    }

    func testRotationSemantics() throws {
        // 5x3, 90° CCW -> 3x5 canvas: out[y][x] == in[h-1-x][y]
        let (w, h) = (5, 3)
        let values = (0..<w * h).map { Float($0) }
        let t = DepthAffineTransform(rotation: .pi / 2, threshold: 0)
        let (gpu, _) = try runAffine(values, w, h, t, h, w)
        for y in 0..<w {
            for x in 0..<h {
                XCTAssertEqual(gpu.depth[y * h + x], values[(h - 1 - x) * w + y],
                               "90° mapping wrong at (\(x),\(y))")
            }
        }
    }

    func testGeneralParity() throws {
        let cases: [(DepthAffineTransform, Int, Int)] = [
            (DepthAffineTransform(rotation: 15 * .pi / 180, tx: 7, ty: -4, scale: 1.3,
                             zShift: 0.25, threshold: .infinity), 150, 140),
            (DepthAffineTransform(rotation: -33 * .pi / 180, tx: -10, ty: 5, scale: 0.7,
                             zShift: -0.1, threshold: 0.5), 100, 100),
            (DepthAffineTransform(rotation: 8 * .pi / 180, tx: 3, ty: 3, scale: 2.0,
                             zShift: 0, threshold: 0.02), 200, 160),
        ]
        let values = smoothPattern(width: 120, height: 80)
        for (t, tw, th) in cases {
            let (gpu, cpu) = try runAffine(values, 120, 80, t, tw, th)
            XCTAssertEqual(gpu.mask, cpu.mask, "mask mismatch")
            let maxDiff = zip(gpu.depth, cpu.depth).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThan(maxDiff, 1e-4, "max abs diff \(maxDiff)")
        }
    }

    func testMaskAndInvalidRegion() throws {
        let values = smoothPattern(width: 120, height: 80)
        let t = DepthAffineTransform(tx: 80, ty: -30, scale: 1, threshold: .infinity)
        let (gpu, cpu) = try runAffine(values, 120, 80, t, 150, 140)
        XCTAssertEqual(gpu.mask, cpu.mask)
        for i in 0..<gpu.mask.count where gpu.mask[i] == 0 {
            XCTAssertEqual(gpu.depth[i], 0, "invalid pixel must be 0 at \(i)")
        }
        let coverage = gpu.mask.reduce(0, +) / Float(gpu.mask.count)
        XCTAssertGreaterThan(coverage, 0.1)
        XCTAssertLessThan(coverage, 0.9)
    }

    func testDepthCorrectionBitExact() throws {
        // identity mapping: out = fma(v, 1.0, 0.5) = v + 0.5 exactly
        let values = pattern(width: 40, height: 30)
        let identity = DepthAffineTransform(scale: 1, zShift: 0.5, threshold: .infinity)
        let (gpu1, cpu1) = try runAffine(values, 40, 30, identity, 40, 30)
        XCTAssertEqual(gpu1.depth, cpu1.depth)
        for (g, v) in zip(gpu1.depth, values) {
            XCTAssertEqual(g, v + 0.5)
        }
        // scale 0.5 on odd dims: inverse map factor 2.0, src = 2*dst exactly,
        // out = fma(v, 2.0, 0) = values[2y][2x] * 2 exactly
        let odd = pattern(width: 41, height: 31)
        let shrink = DepthAffineTransform(scale: 0.5, zShift: 0, threshold: .infinity)
        let (gpu2, cpu2) = try runAffine(odd, 41, 31, shrink, 21, 16)
        XCTAssertEqual(gpu2.depth, cpu2.depth)
        for y in 0..<16 {
            for x in 0..<21 {
                XCTAssertEqual(gpu2.depth[y * 21 + x], odd[(2 * y) * 41 + 2 * x] * 2,
                               "at (\(x),\(y))")
            }
        }
    }

    func testNearestOnDiscontinuity() throws {
        // step edge: left half 0, right half 10; threshold 1 forces nearest at the edge
        let (w, h) = (60, 40)
        var values = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in w / 2..<w {
                values[y * w + x] = 10
            }
        }
        let t = DepthAffineTransform(rotation: 5 * .pi / 180, scale: 1.2, threshold: 1.0)
        let (gpu, _) = try runAffine(values, w, h, t, 80, 60)
        let hi = 10 * Float(1.0 / 1.2)  // nearest pick + metric correction depth/s
        for i in 0..<gpu.depth.count where gpu.mask[i] == 1 {
            XCTAssertTrue(gpu.depth[i] == 0 || gpu.depth[i] == hi,
                          "nearest fallback must pick a source value, got \(gpu.depth[i])")
        }
    }
}
