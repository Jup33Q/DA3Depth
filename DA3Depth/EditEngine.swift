import Foundation
import DepthShaderKit

/// Owns the DepthEditStack and serializes all GPU edit work off the main actor.
/// Edit semantics: the stack holds a single editable base layer (raw inferred depth,
/// identity transform). transform_depth sets that layer's transform (absolute values,
/// nil = keep current). fuse_depth bakes the current canvas into a new identity base
/// (flatten, enables cumulative edits). edit_reset restores the raw inferred depth.
actor EditEngine {
    private var stack: DepthEditStack?
    private let guided = GuidedFilterParams(radius: 2, depthThreshold: 1.0)
    private let editThreshold: Float = 0.05  // depth-aware resampling threshold (source depth units)

    func load(values: [Float], width: Int, height: Int) throws {
        let src = try DepthTexture(values: values, width: width, height: height)
        stack = try DepthEditStack(canvasWidth: width, canvasHeight: height,
                                   layers: [EditLayer(source: src)], guided: guided)
    }

    func currentTransform() -> DepthAffineTransform? {
        stack?.layers.first?.transform
    }

    func apply(rotateDeg: Float?, tx: Float?, ty: Float?, scale: Float?, zShift: Float?) throws
        -> (values: [Float], width: Int, height: Int, canUndo: Bool, canRedo: Bool) {
        guard let stack else { throw DepthEngine.EngineError.badImage }
        var t = stack.layers[0].transform
        if let rotateDeg { t.rotation = rotateDeg * .pi / 180 }
        if let tx { t.tx = tx }
        if let ty { t.ty = ty }
        if let scale { t.scale = scale }
        if let zShift { t.zShift = zShift }
        t.threshold = editThreshold
        try stack.updateLayer(0, transform: t)
        return (stack.canvas.readback(), stack.canvasWidth, stack.canvasHeight,
                stack.canUndo, stack.canRedo)
    }

    func bake() throws -> (values: [Float], width: Int, height: Int) {
        guard let stack else { throw DepthEngine.EngineError.badImage }
        let values = stack.canvas.readback()
        let src = try DepthTexture(values: values, width: stack.canvasWidth, height: stack.canvasHeight)
        self.stack = try DepthEditStack(canvasWidth: stack.canvasWidth, canvasHeight: stack.canvasHeight,
                                        layers: [EditLayer(source: src)], guided: guided)
        return (values, stack.canvasWidth, stack.canvasHeight)
    }

    func undo() throws -> (values: [Float], canUndo: Bool, canRedo: Bool)? {
        guard let stack, try stack.undo() else { return nil }
        return (stack.canvas.readback(), stack.canUndo, stack.canRedo)
    }

    func redo() throws -> (values: [Float], canUndo: Bool, canRedo: Bool)? {
        guard let stack, try stack.redo() else { return nil }
        return (stack.canvas.readback(), stack.canUndo, stack.canRedo)
    }

    /// 2.5D reprojection (M7). `depthValues` at (depthWidth x depthHeight) is
    /// nearest-upsampled onto the input-resolution canvas; `rgba` is canvas-sized.
    /// Angles are clamped to ±30° and applied recursively in <=5° micro-steps.
    func reproject(depthValues: [Float], depthWidth: Int, depthHeight: Int,
                   rgba: [UInt8], canvasWidth: Int, canvasHeight: Int,
                   yawDeg: Float, pitchDeg: Float, fillRadius: Int, softenEdges: Bool) throws
        -> (rgba: [UInt8], depth: [Float], mask: [Float], coverage: Float,
            steps: Int, appliedYaw: Float, appliedPitch: Float) {
        let src = try DepthTexture(values: depthValues, width: depthWidth, height: depthHeight)
        let result = try DepthOps().reprojectView(
            depth: src, colorRGBA: rgba, canvasWidth: canvasWidth, canvasHeight: canvasHeight,
            params: ReprojectParams(yawDeg: yawDeg, pitchDeg: pitchDeg,
                                    fillRadius: fillRadius, softenEdges: softenEdges))
        return (result.colorRGBA8(), result.depth.readback(), result.mask.readback(),
                result.coverage, result.steps, result.appliedYawDeg, result.appliedPitchDeg)
    }
}
