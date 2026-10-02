import Metal

/// One depth layer on a common canvas: depth + valid mask (as produced by affine()).
public struct DepthLayer {
    public var depth: DepthTexture
    public var mask: DepthTexture
    public init(depth: DepthTexture, mask: DepthTexture) {
        self.depth = depth
        self.mask = mask
    }
}

extension AffineResult {
    public var layer: DepthLayer { DepthLayer(depth: depth, mask: mask) }
}

public struct FuseParams {
    /// |dA - dB| below this -> depth-weighted blend instead of a hard pick. 0 = always hard.
    public var blendThreshold: Float
    /// |dA - dB| below this (both layers valid) -> seam pixel, marks the guided-filter band. 0 = no seam.
    public var seamThreshold: Float
    public init(blendThreshold: Float = 0, seamThreshold: Float = 0) {
        self.blendThreshold = blendThreshold
        self.seamThreshold = seamThreshold
    }
}

public struct FuseResult {
    public let depth: DepthTexture
    public let mask: DepthTexture
    /// -1 = no layer valid, 0 = layer A won, 1 = layer B won (blend still reports the nearer layer).
    public let winner: DepthTexture
    /// 1 where both layers are valid and |dA - dB| < seamThreshold.
    public let seam: DepthTexture
    public var layer: DepthLayer { DepthLayer(depth: depth, mask: mask) }
}

/// Approximation: single-pass guided filter (box mean/variance over a (2r+1)^2 window,
/// no a/b averaging pass). Guide = RGB luminance if a guide texture is given, else the
/// fused depth itself. Neighbors beyond `depthThreshold` from the center pixel are excluded,
/// so smoothing never bleeds across true depth discontinuities.
public struct GuidedFilterParams {
    public var radius: Int
    public var epsilon: Float
    public var depthThreshold: Float
    public init(radius: Int = 2, epsilon: Float = 1e-3, depthThreshold: Float = .infinity) {
        self.radius = radius
        self.epsilon = epsilon
        self.depthThreshold = depthThreshold
    }
}

struct FuseKernelParams {
    var blendThreshold: Float
    var seamThreshold: Float
    var ox: UInt32 = 0, oy: UInt32 = 0
}

struct GuidedKernelParams {
    var epsilon: Float
    var depthThreshold: Float
    var radius: Int32
    var useDepthGuide: Int32
    var ox: UInt32 = 0, oy: UInt32 = 0
}

extension DepthOps {
    /// Two-layer z-buffer composite (smaller depth = nearer). N-layer support in M4 will
    /// chain this over the edit stack; the winner/seam outputs are designed for that.
    public func fuse(_ a: DepthLayer, _ b: DepthLayer,
                     params: FuseParams = FuseParams()) throws -> FuseResult {
        precondition(a.depth.width == b.depth.width && a.depth.height == b.depth.height,
                     "layers must share a common canvas")
        let w = a.depth.width, h = a.depth.height
        let depth = try makeOutput(width: w, height: h)
        let mask = try makeOutput(width: w, height: h)
        let winner = try makeOutput(width: w, height: h)
        let seam = try makeOutput(width: w, height: h)
        let pipeline = try context.pipeline(function: "fuse_layers")
        var kp = FuseKernelParams(blendThreshold: params.blendThreshold,
                                  seamThreshold: params.seamThreshold)
        try context.encode(pipeline, width: w, height: h) { encoder in
            encoder.setTexture(a.depth.texture, index: 0)
            encoder.setTexture(a.mask.texture, index: 1)
            encoder.setTexture(b.depth.texture, index: 2)
            encoder.setTexture(b.mask.texture, index: 3)
            encoder.setTexture(depth, index: 4)
            encoder.setTexture(mask, index: 5)
            encoder.setTexture(winner, index: 6)
            encoder.setTexture(seam, index: 7)
            encoder.setBytes(&kp, length: MemoryLayout<FuseKernelParams>.stride, index: 0)
        }
        return FuseResult(depth: DepthTexture(texture: depth), mask: DepthTexture(texture: mask),
                          winner: DepthTexture(texture: winner), seam: DepthTexture(texture: seam))
    }

    /// Smooths only the dilated seam band of a fused result; all other pixels are copied
    /// bit-exactly. `guide` may be any rgba texture (luminance is used); nil = depth as guide.
    public func guidedSmooth(_ fused: FuseResult, params: GuidedFilterParams = GuidedFilterParams(),
                             guide: MTLTexture? = nil) throws -> DepthTexture {
        let w = fused.depth.width, h = fused.depth.height
        let out = try makeOutput(width: w, height: h)
        let pipeline = try context.pipeline(function: "guided_smooth")
        var kp = GuidedKernelParams(epsilon: params.epsilon, depthThreshold: params.depthThreshold,
                                    radius: Int32(params.radius), useDepthGuide: guide == nil ? 1 : 0)
        try context.encode(pipeline, width: w, height: h) { encoder in
            encoder.setTexture(fused.depth.texture, index: 0)
            encoder.setTexture(fused.mask.texture, index: 1)
            encoder.setTexture(fused.seam.texture, index: 2)
            encoder.setTexture(guide ?? fused.depth.texture, index: 3)
            encoder.setTexture(out, index: 4)
            encoder.setBytes(&kp, length: MemoryLayout<GuidedKernelParams>.stride, index: 0)
        }
        return DepthTexture(texture: out)
    }
}
