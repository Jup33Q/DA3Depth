import DepthShaderKit
import Foundation

// CPU reference implementations, kept in sync with Tests/DepthShaderKitTests.
// (Roadmap discipline: CPU references live outside the library main path.)

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

func cpuGuided(depth: [Float], mask: [Float], seam: [Float],
               width w: Int, height h: Int, params: GuidedFilterParams) -> [Float] {
    var out = depth
    let r = params.radius
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
                    n += 1; sg += dn; sp += dn; sgg += dn * dn; sgp += dn * dn
                }
            }
            if n == 0 { continue }
            let meanG = sg / n, meanP = sp / n
            let varG = sgg / n - meanG * meanG
            let cov = sgp / n - meanG * meanP
            let a = cov / (varG + params.epsilon)
            let b = meanP - a * meanG
            out[i] = a * dc + b
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

// ---- parity case runner ----

struct ParityCase: Codable {
    let name: String
    let size: String
    let max_abs_diff: Double
    let tolerance: Double
    let pass: Bool
}

struct ParityReport: Codable {
    let cases: [ParityCase]
    let all_pass: Bool
}

let ops = DepthOps()
var cases: [ParityCase] = []

func check(_ name: String, _ size: String, _ gpu: [Float], _ cpu: [Float], tolerance: Double) {
    let maxDiff = zip(gpu, cpu).map { abs($0 - $1) }.max() ?? 0
    cases.append(ParityCase(name: name, size: size, max_abs_diff: Double(maxDiff),
                            tolerance: tolerance, pass: Double(maxDiff) <= tolerance))
}

do {
    // roundtrip lossless
    let rt = pattern(width: 1920, height: 1080)
    let rtTex = try DepthTexture(values: rt, width: 1920, height: 1080)
    check("roundtrip", "1920x1080", rtTex.readback(), rt, tolerance: 0)

    // flip exact, odd/even and non-square
    for (w, h) in [(256, 128), (31, 17)] {
        let v = pattern(width: w, height: h)
        let gpu = try ops.flippedHorizontal(values: v, width: w, height: h)
        check("flip", "\(w)x\(h)", gpu, cpuFlip(v, width: w, height: h), tolerance: 0)
    }

    // resize parity
    for (w, h, tw, th) in [(128, 128, 64, 64), (100, 50, 240, 180)] {
        let v = pattern(width: w, height: h)
        let gpu = try ops.resized(values: v, width: w, height: h, to: tw, th)
        check("resize", "\(w)x\(h)->\(tw)x\(th)", gpu,
              cpuResize(v, width: w, height: h, to: tw, th), tolerance: 1e-4)
    }

    // affine 90-degree rotation, bit-exact
    do {
        let (w, h) = (16, 10)
        let v = pattern(width: w, height: h)
        let t = DepthAffineTransform(rotation: .pi / 2, scale: 1, zShift: 0, threshold: 0.5)
        let gpu = try ops.affine(values: v, width: w, height: h, transform: t, canvasWidth: h, canvasHeight: w)
        let cpu = cpuAffine(v, width: w, height: h, transform: t, canvas: h, w)
        check("affine rot90 depth", "\(w)x\(h)->\(h)x\(w)", gpu.depth, cpu.depth, tolerance: 0)
        check("affine rot90 mask", "\(w)x\(h)->\(h)x\(w)", gpu.mask, cpu.mask, tolerance: 0)
    }

    // affine general case
    do {
        let v = smoothPattern(width: 120, height: 80)
        let t = DepthAffineTransform(rotation: 15 * .pi / 180, tx: 7, ty: -4, scale: 1.3,
                                     zShift: 0.25, threshold: .infinity)
        let gpu = try ops.affine(values: v, width: 120, height: 80, transform: t,
                                 canvasWidth: 150, canvasHeight: 140)
        let cpu = cpuAffine(v, width: 120, height: 80, transform: t, canvas: 150, 140)
        check("affine general", "120x80->150x140 rot15 s1.3", gpu.depth, cpu.depth, tolerance: 1e-4)
        check("affine general mask", "120x80->150x140", gpu.mask, cpu.mask, tolerance: 0)
    }

    // fuse: hard pick exact + blend band tolerance, on overlapping circular masks
    let (fw, fh) = (48, 32)
    var da = [Float](repeating: 0, count: fw * fh)
    var db = [Float](repeating: 0, count: fw * fh)
    var ma = [Float](repeating: 0, count: fw * fh)
    var mb = [Float](repeating: 0, count: fw * fh)
    for y in 0..<fh {
        for x in 0..<fw {
            let i = y * fw + x
            da[i] = 5 + Float(x) * 0.05 + sin(Float(y) * 0.1)
            db[i] = da[i] + (x < fw / 2 ? -0.02 : 0.6)
            let ax = Float(x - 20), ay = Float(y - 16)
            ma[i] = ax * ax + ay * ay < 225 ? 1 : 0
            let bx = Float(x - 28), by = Float(y - 16)
            mb[i] = bx * bx + by * by < 225 ? 1 : 0
        }
    }
    func layer(_ d: [Float], _ m: [Float]) throws -> DepthLayer {
        DepthLayer(depth: try DepthTexture(values: d, width: fw, height: fh),
                   mask: try DepthTexture(values: m, width: fw, height: fh))
    }
    let la = try layer(da, ma), lb = try layer(db, mb)

    let fusedHard = try ops.fuse(la, lb, params: FuseParams(blendThreshold: 0, seamThreshold: 0.1))
    let cpuHard = cpuFuse(depthA: da, maskA: ma, depthB: db, maskB: mb,
                          blendThreshold: 0, seamThreshold: 0.1)
    check("fuse hard pick", "\(fw)x\(fh)", fusedHard.depth.readback(), cpuHard.depth, tolerance: 0)

    let fusedBlend = try ops.fuse(la, lb, params: FuseParams(blendThreshold: 0.1, seamThreshold: 0.1))
    let cpuBlend = cpuFuse(depthA: da, maskA: ma, depthB: db, maskB: mb,
                           blendThreshold: 0.1, seamThreshold: 0.1)
    check("fuse blend band", "\(fw)x\(fh)", fusedBlend.depth.readback(), cpuBlend.depth, tolerance: 1e-4)
    check("fuse seam mask", "\(fw)x\(fh)", fusedBlend.seam.readback(), cpuBlend.seam, tolerance: 0)

    let gpuGuided = try ops.guidedSmooth(fusedBlend,
                                         params: GuidedFilterParams(radius: 2, depthThreshold: 2.0))
    let cpuGuidedOut = cpuGuided(depth: fusedBlend.depth.readback(), mask: fusedBlend.mask.readback(),
                                 seam: fusedBlend.seam.readback(),
                                 width: fw, height: fh,
                                 params: GuidedFilterParams(radius: 2, depthThreshold: 2.0))
    check("guided smooth", "\(fw)x\(fh) r2", gpuGuided.readback(), cpuGuidedOut, tolerance: 1e-3)

    // edit stack: dirty-tile incremental render must equal full re-render bit-exactly
    let (cw, ch) = (300, 200)
    let base = EditLayer(source: try DepthTexture(values: pattern(width: cw, height: ch, seed: 1),
                                                  width: cw, height: ch))
    let top = EditLayer(source: try DepthTexture(values: pattern(width: 120, height: 90, seed: 2),
                                                 width: 120, height: 90),
                        transform: DepthAffineTransform(rotation: 20 * .pi / 180, tx: 30, ty: 10,
                                                        scale: 1.1, zShift: -0.5, threshold: 0.05))
    let stack = try DepthEditStack(canvasWidth: cw, canvasHeight: ch, layers: [base, top],
                                   guided: GuidedFilterParams(radius: 2, depthThreshold: 1.0))
    try stack.updateLayer(1, transform: DepthAffineTransform(rotation: 47 * .pi / 180, tx: -20,
                                                             ty: 40, scale: 1.3, threshold: 0.05))
    let incremental = stack.canvas.readback()
    let full = try stack.render().readback()
    check("stack dirty vs full", "\(cw)x\(ch) 2 layers", incremental, full, tolerance: 0)
}

let report = ParityReport(cases: cases, all_pass: cases.allSatisfy(\.pass))

if CommandLine.arguments.contains("--json") {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    print(String(data: try encoder.encode(report), encoding: .utf8)!)
} else {
    for c in cases {
        print(String(format: "%-22@ %-24@ max diff %10.3e  tol %8.1e  %@",
                     c.name as NSString, c.size as NSString, c.max_abs_diff, c.tolerance,
                     (c.pass ? "PASS" : "FAIL") as NSString))
    }
    print(report.all_pass ? "ALL PASS (\(cases.count) cases)" : "FAILURES: \(cases.filter { !$0.pass }.count) of \(cases.count)")
}

exit(report.all_pass ? 0 : 1)
