import DepthShaderKit
import Foundation
import QuartzCore

// usage: depthshader-bench [--json] [--size WxH]   (default 3840x2160, human output)
var jsonMode = false
var width = 3840, height = 2160
var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--json":
        jsonMode = true
    case "--size":
        let spec = args.popFirst()!.split(separator: "x")
        width = Int(spec[0])!; height = Int(spec[1])!
    default:
        FileHandle.standardError.write("unknown argument: \(arg)\n".data(using: .utf8)!)
        exit(2)
    }
}

let iterations = 20
let megapixels = Double(width * height) / 1e6

struct KernelResult: Codable {
    let name: String
    let wall_ms: Double
    let mpix_s: Double
    let gpu_ms: Double?
}

struct DragSimResult: Codable {
    let steps: Int
    let step_deg: Double
    let mean_ms: Double
    let p95_ms: Double
    let max_ms: Double
}

struct BenchReport: Codable {
    let width: Int
    let height: Int
    let iterations: Int
    let kernels: [KernelResult]
    let drag_sim: DragSimResult
}

var values = [Float](repeating: 0, count: width * height)
for y in 0..<height {
    let fy = Float(y)
    for x in 0..<width {
        values[y * width + x] = (fy + Float(x)) * 1e-4
    }
}

let ops = DepthOps()

func bench(_ label: String, gpu: Bool, _ body: () throws -> Void) -> KernelResult {
    var bestWall = Double.greatestFiniteMagnitude
    var bestGPU = Double.greatestFiniteMagnitude
    for _ in 0..<iterations {
        let start = CACurrentMediaTime()
        try! body()
        bestWall = min(bestWall, (CACurrentMediaTime() - start) * 1000)
        bestGPU = min(bestGPU, ops.context.lastGPUTime * 1000)
    }
    return KernelResult(name: label, wall_ms: bestWall,
                        mpix_s: megapixels / (bestWall / 1000),
                        gpu_ms: gpu ? bestGPU : nil)
}

var kernels: [KernelResult] = []
var tex: DepthTexture!

kernels.append(bench("upload", gpu: false) {
    tex = try DepthTexture(values: values, width: width, height: height)
})

var readback: [Float] = []
kernels.append(bench("readback", gpu: false) {
    readback = tex.readback()
})

kernels.append(bench("flip", gpu: true) {
    _ = try ops.flippedHorizontal(tex)
})

kernels.append(bench("resize", gpu: true) {
    _ = try ops.resized(tex, to: width, height)
})

let transform = DepthAffineTransform(rotation: 17 * .pi / 180, tx: 13, ty: -7,
                                     scale: 1.2, zShift: 0.1, threshold: 0.05)
kernels.append(bench("affine", gpu: true) {
    _ = try ops.affine(tex, transform: transform, canvasWidth: width, canvasHeight: height)
})

let layerA = try ops.affine(tex, transform: DepthAffineTransform(
    rotation: 10 * .pi / 180, scale: 1.1, threshold: 0.05), canvasWidth: width, canvasHeight: height)
let layerB = try ops.affine(tex, transform: DepthAffineTransform(
    rotation: -8 * .pi / 180, tx: 120, ty: -40, scale: 1.05, zShift: -0.3, threshold: 0.05),
    canvasWidth: width, canvasHeight: height)
kernels.append(bench("fuse", gpu: true) {
    _ = try ops.fuse(layerA.layer, layerB.layer, params: FuseParams(blendThreshold: 0.15, seamThreshold: 0.15))
})

let fused = try ops.fuse(layerA.layer, layerB.layer, params: FuseParams(blendThreshold: 0.15, seamThreshold: 0.15))
kernels.append(bench("guided", gpu: true) {
    _ = try ops.guidedSmooth(fused, params: GuidedFilterParams(radius: 2, depthThreshold: 1.0))
})

// M4: simulate a rotation-handle drag — base layer + 1200x800 top layer,
// 60 steps of 0.5 degrees with dirty-tile re-render per step.
let srcW = 1200, srcH = 800
var srcValues = [Float](repeating: 0, count: srcW * srcH)
for i in 0..<srcW * srcH { srcValues[i] = Float(i % 997) * 0.01 }
let baseLayer = EditLayer(source: tex)
let topLayer = EditLayer(source: try DepthTexture(values: srcValues, width: srcW, height: srcH),
                         transform: DepthAffineTransform(scale: 1, threshold: 0.05))
let stack = try DepthEditStack(canvasWidth: width, canvasHeight: height,
                               layers: [baseLayer, topLayer],
                               guided: GuidedFilterParams(radius: 2, depthThreshold: 1.0))
var steps: [Double] = []
for i in 1...60 {
    let t = DepthAffineTransform(rotation: Float(i) * 0.5 * .pi / 180, threshold: 0.05)
    let start = CACurrentMediaTime()
    try stack.updateLayer(1, transform: t)
    steps.append((CACurrentMediaTime() - start) * 1000)
}
let sortedSteps = steps.sorted()
let dragSim = DragSimResult(steps: steps.count, step_deg: 0.5,
                            mean_ms: steps.reduce(0, +) / Double(steps.count),
                            p95_ms: sortedSteps[Int(Double(sortedSteps.count) * 0.95) - 1],
                            max_ms: sortedSteps.last!)

if jsonMode {
    let report = BenchReport(width: width, height: height, iterations: iterations,
                             kernels: kernels, drag_sim: dragSim)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    print(String(data: try encoder.encode(report), encoding: .utf8)!)
} else {
    for k in kernels {
        print(String(format: "%-10@ best %7.3f ms  %8.1f MPix/s", k.name as NSString, k.wall_ms, k.mpix_s))
        if let gpu = k.gpu_ms {
            print(String(format: "%-10@ GPU-only best %7.3f ms", (k.name + " gpu") as NSString, gpu))
        }
    }
    print(String(format: "drag-sim (%d x %.1f deg, %dx%d canvas, dirty tiles): mean %.2f ms  p95 %.2f ms  max %.2f ms  (60fps budget: 16.6 ms)",
                 dragSim.steps, dragSim.step_deg, width, height,
                 dragSim.mean_ms, dragSim.p95_ms, dragSim.max_ms))
    print("sanity: readback[0]=\(readback[0]) (expect 0), flipped row0 last=\(try ops.flippedHorizontal(tex).readback()[width - 1])")
}
