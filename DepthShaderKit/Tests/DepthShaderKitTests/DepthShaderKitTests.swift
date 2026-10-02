import XCTest
@testable import DepthShaderKit

// CPU reference implementations copied verbatim from DA3Depth/DepthMap.swift.
func cpuFlip(_ values: [Float], width: Int, height: Int) -> [Float] {
    var out = [Float](repeating: 0, count: values.count)
    for y in 0..<height {
        for x in 0..<width {
            out[y * width + x] = values[y * width + (width - 1 - x)]
        }
    }
    return out
}

func cpuResize(_ values: [Float], width: Int, height: Int, to tw: Int, _ th: Int) -> [Float] {
    guard tw != width || th != height else { return values }
    var out = [Float](repeating: 0, count: tw * th)
    let sx = Double(width) / Double(tw), sy = Double(height) / Double(th)
    for y in 0..<th {
        let fy = (Double(y) + 0.5) * sy - 0.5
        let y0 = max(0, Int(fy.rounded(.down)))
        let y1 = min(height - 1, y0 + 1)
        let wy = Float(fy - Double(y0))
        for x in 0..<tw {
            let fx = (Double(x) + 0.5) * sx - 0.5
            let x0 = max(0, Int(fx.rounded(.down)))
            let x1 = min(width - 1, x0 + 1)
            let wx = Float(fx - Double(x0))
            let a = values[y0 * width + x0], b = values[y0 * width + x1]
            let c = values[y1 * width + x0], d = values[y1 * width + x1]
            out[y * tw + x] = a * (1 - wx) * (1 - wy) + b * wx * (1 - wy)
                            + c * (1 - wx) * wy + d * wx * wy
        }
    }
    return out
}

struct LCG {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float((state >> 33) & 0xFFFFFF) / Float(0xFFFFFF)
    }
}

func pattern(width: Int, height: Int, seed: UInt64 = 42) -> [Float] {
    var rng = LCG(seed: seed)
    var out = [Float](repeating: 0, count: width * height)
    for y in 0..<height {
        for x in 0..<width {
            let base = sin(Float(x) * 0.05) * cos(Float(y) * 0.03) + Float(x + y) * 0.01
            out[y * width + x] = base * 10 + rng.next()
        }
    }
    return out
}

final class DepthShaderKitTests: XCTestCase {
    let ops = DepthOps()

    func testRoundtripLossless() throws {
        for (w, h) in [(1, 1), (3, 2), (64, 64), (257, 131), (1920, 1080)] {
            let values = pattern(width: w, height: h)
            let tex = try DepthTexture(values: values, width: w, height: h)
            XCTAssertEqual(tex.readback(), values, "roundtrip mismatch at \(w)x\(h)")
        }
    }

    func testFlipParityExact() throws {
        for (w, h) in [(1, 1), (2, 1), (1, 7), (3, 5), (16, 16), (31, 17), (256, 128)] {
            let values = pattern(width: w, height: h)
            let gpu = try ops.flippedHorizontal(values: values, width: w, height: h)
            XCTAssertEqual(gpu, cpuFlip(values, width: w, height: h), "flip mismatch at \(w)x\(h)")
        }
    }

    func testResizeParity() throws {
        let cases: [(Int, Int, Int, Int)] = [
            (64, 64, 128, 128),      // upscale
            (128, 128, 64, 64),      // downscale
            (100, 50, 240, 180),     // non-uniform upscale
            (240, 180, 37, 53),      // non-uniform downscale
            (33, 33, 100, 7),        // extreme aspect
            (1, 1, 5, 5),            // tiny source
            (1920, 1080, 640, 480),
        ]
        for (w, h, tw, th) in cases {
            let values = pattern(width: w, height: h)
            let gpu = try ops.resized(values: values, width: w, height: h, to: tw, th)
            let cpu = cpuResize(values, width: w, height: h, to: tw, th)
            let maxDiff = zip(gpu, cpu).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThan(maxDiff, 1e-4, "resize \(w)x\(h)->\(tw)x\(th) max abs diff \(maxDiff)")
        }
    }

    func testIdentityResize() throws {
        let values = pattern(width: 50, height: 40)
        let gpu = try ops.resized(values: values, width: 50, height: 40, to: 50, 40)
        let maxDiff = zip(gpu, values).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(maxDiff, 1e-4)
    }
}
