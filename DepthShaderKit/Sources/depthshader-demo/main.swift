import DepthShaderKit
import Foundation

let width = 256, height = 128

var values = [Float](repeating: 0, count: width * height)
for y in 0..<height {
    for x in 0..<width {
        values[y * width + x] = Float(x) / Float(width - 1) + Float(y) / Float(height - 1) * 2
    }
}

func checksum(_ v: [Float]) -> Float {
    v.reduce(0) { $0 + $1 * 0.1 }  // weighted down so large arrays don't explode
}

print("input:      \(width)x\(height)  checksum=\(checksum(values))")

let ops = DepthOps()
let tex = try DepthTexture(values: values, width: width, height: height)

let flipped = try ops.flippedHorizontal(tex).readback()
print("flipped:    \(width)x\(height)  checksum=\(checksum(flipped))")
print("  row0 first: \(flipped[0])  last: \(flipped[width - 1])")

let resized = try ops.resized(tex, to: 384, 200).readback()
print("resized:    384x200  checksum=\(checksum(resized))")
print("  row0 first: \(resized[0])  row0 last: \(resized[383])")

let roundtrip = tex.readback()
let maxDiff = zip(values, roundtrip).map { abs($0 - $1) }.max()!
print("roundtrip max abs diff: \(maxDiff)")

let transform = DepthAffineTransform(rotation: 25 * .pi / 180, tx: 10, ty: -5,
                                scale: 1.5, zShift: 0.25, threshold: 0.05)
let affine = try ops.affine(tex, transform: transform, canvasWidth: 384, canvasHeight: 240)
let affineDepth = affine.depth.readback()
let affineMask = affine.mask.readback()
let coverage = affineMask.reduce(0, +) / Float(affineMask.count) * 100
print(String(format: "affine:     384x240  coverage=%.1f%%  checksum=%.4f", coverage, checksum(affineDepth)))
let masked = zip(affineDepth, affineMask).filter { $0.1 == 1 }.map(\.0)
print("  valid depth range: \(masked.min()!)...\(masked.max()!)")

let canvasW = 384, canvasH = 240
let layerA = try ops.affine(tex, transform: DepthAffineTransform(
    rotation: 15 * .pi / 180, scale: 1.2, threshold: 0.05), canvasWidth: canvasW, canvasHeight: canvasH)
let layerB = try ops.affine(tex, transform: DepthAffineTransform(
    rotation: -10 * .pi / 180, tx: 40, ty: 10, scale: 1.1, zShift: -0.4, threshold: 0.05),
    canvasWidth: canvasW, canvasHeight: canvasH)
let fused = try ops.fuse(layerA.layer, layerB.layer,
                         params: FuseParams(blendThreshold: 0.15, seamThreshold: 0.15))
let fusedMask = fused.mask.readback()
let seamCount = fused.seam.readback().reduce(0, +)
let fusedCoverage = fusedMask.reduce(0, +) / Float(fusedMask.count) * 100
print(String(format: "fuse:       %dx%d  coverage=%.1f%%  seam px=%d  checksum=%.4f",
             canvasW, canvasH, fusedCoverage, Int(seamCount), checksum(fused.depth.readback())))
let smoothed = try ops.guidedSmooth(fused, params: GuidedFilterParams(radius: 2, depthThreshold: 1.0))
let changed = zip(smoothed.readback(), fused.depth.readback()).filter { $0 != $1 }.count
print(String(format: "guided:     %dx%d  changed px=%d  checksum=%.4f",
             canvasW, canvasH, changed, checksum(smoothed.readback())))

let stackBase = EditLayer(source: try DepthTexture(values: values, width: width, height: height))
let stackTop = EditLayer(source: try DepthTexture(values: values, width: width, height: height),
                         transform: DepthAffineTransform(rotation: 15 * .pi / 180, tx: 20, ty: 8,
                                                         scale: 1.1, zShift: -0.3, threshold: 0.05))
let stack = try DepthEditStack(canvasWidth: width, canvasHeight: height,
                               layers: [stackBase, stackTop],
                               guided: GuidedFilterParams(radius: 2, depthThreshold: 1.0))
print(String(format: "stack:      %dx%d  layers=2  checksum=%.4f", width, height,
             checksum(stack.canvas.readback())))
try stack.updateLayer(1, transform: DepthAffineTransform(rotation: 45 * .pi / 180, tx: -30,
                                                         ty: 12, scale: 0.9, threshold: 0.05))
let editedChecksum = checksum(stack.canvas.readback())
try stack.undo()
print(String(format: "stack edit: checksum=%.4f  after undo=%.4f (dirty-tile re-render)",
             editedChecksum, checksum(stack.canvas.readback())))
