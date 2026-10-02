import Metal
import Foundation

/// In-plane affine transform of a depth layer onto an arbitrary output canvas.
/// Forward convention: the source image is scaled by `scale` about its own center,
/// rotated CCW by `rotation` radians, centered on the canvas, then shifted by `translation` pixels.
/// `zShift` is added to sampled depth; sampled depth is also divided by `scale` (metric correction).
/// `threshold` is the depth-discontinuity threshold in absolute source-depth units:
/// if the 2x2 bilinear footprint's (max - min) exceeds it, sampling falls back to nearest.
public struct DepthAffineTransform: Equatable {
    public var rotation: Float
    public var tx: Float
    public var ty: Float
    public var scale: Float
    public var zShift: Float
    public var threshold: Float

    public init(rotation: Float = 0, tx: Float = 0, ty: Float = 0,
                scale: Float = 1, zShift: Float = 0, threshold: Float = .infinity) {
        self.rotation = rotation
        self.tx = tx
        self.ty = ty
        self.scale = scale
        self.zShift = zShift
        self.threshold = threshold
    }

    /// Inverse map src = M * dst + b, computed in double precision and truncated to float.
    /// Layout must match `AffineParams` in DepthShaders.metal (9 floats + 2 origin uints).
    public func params(sourceWidth sw: Int, sourceHeight sh: Int,
                       canvasWidth tw: Int, canvasHeight th: Int) -> AffineParams {
        let csx = Double(sw - 1) / 2, csy = Double(sh - 1) / 2
        let cdx = Double(tw - 1) / 2, cdy = Double(th - 1) / 2
        let c = cos(Double(rotation)), s = sin(Double(rotation))
        let inv = 1.0 / Double(scale)
        let m00 = inv * c, m01 = inv * s
        let m10 = -inv * s, m11 = inv * c
        let px = cdx + Double(tx), py = cdy + Double(ty)
        return AffineParams(m00: Float(m00), m01: Float(m01),
                            m10: Float(m10), m11: Float(m11),
                            bx: Float(csx - (m00 * px + m01 * py)),
                            by: Float(csy - (m10 * px + m11 * py)),
                            invScale: Float(inv), dz: zShift, threshold: threshold)
    }
}

/// Inverse-map constants shared with the `affine_transform` kernel. `ox`/`oy` are the
/// dispatch origin used by the edit stack for tile-region renders (0,0 for full canvas).
public struct AffineParams {
    public var m00: Float, m01: Float, m10: Float, m11: Float
    public var bx: Float, by: Float
    public var invScale: Float, dz: Float, threshold: Float
    public var ox: UInt32 = 0, oy: UInt32 = 0
}

public struct AffineResult {
    public let depth: DepthTexture
    /// r32Float texture, 1 where the inverse-mapped source coordinate is inside the source, else 0.
    public let mask: DepthTexture
}

extension DepthOps {
    public func affine(_ input: DepthTexture, transform: DepthAffineTransform,
                       canvasWidth tw: Int, canvasHeight th: Int) throws -> AffineResult {
        let depth = try makeOutput(width: tw, height: th)
        let mask = try makeOutput(width: tw, height: th)
        let pipeline = try context.pipeline(function: "affine_transform")
        var params = transform.params(sourceWidth: input.width, sourceHeight: input.height,
                                      canvasWidth: tw, canvasHeight: th)
        try context.encode(pipeline, width: tw, height: th) { encoder in
            encoder.setTexture(input.texture, index: 0)
            encoder.setTexture(depth, index: 1)
            encoder.setTexture(mask, index: 2)
            encoder.setBytes(&params, length: MemoryLayout<AffineParams>.stride, index: 0)
        }
        return AffineResult(depth: DepthTexture(texture: depth), mask: DepthTexture(texture: mask))
    }

    public func affine(values: [Float], width: Int, height: Int,
                       transform: DepthAffineTransform,
                       canvasWidth tw: Int, canvasHeight th: Int) throws -> (depth: [Float], mask: [Float]) {
        let result = try affine(DepthTexture(values: values, width: width, height: height),
                                transform: transform, canvasWidth: tw, canvasHeight: th)
        return (result.depth.readback(), result.mask.readback())
    }
}
