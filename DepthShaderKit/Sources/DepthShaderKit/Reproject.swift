import Metal
import Foundation

/// Out-of-plane (2.5D) reprojection: back-project depth to a point cloud, rotate the
/// viewpoint about the vertical axis (yaw, plus optional small pitch) through a pivot
/// depth plane, then splat back to the canvas as point sprites with a hardware z-test
/// (rasterization, no atomics). Color and depth share the same depth test, so the
/// warped RGB and warped depth stay paired. Small disocclusion holes are filled by
/// neighborhood diffusion; big holes keep mask = 0 (LDI/inpainting is out of scope).
///
/// Travel limits: a single step is a micro-move of at most `maxStepDeg`; larger
/// requests are applied recursively as equal sub-steps chained on the previous output
/// (with a fixed pivot from the first frame), and the total per-axis angle is clamped
/// to ±`maxTotalDeg`. Splat stair-step edges are softened by a 3x3 gaussian blend
/// restricted to pixels touching a hole or a depth discontinuity (`softenEdges`).
public struct ReprojectParams {
    /// Viewpoint yaw in degrees about the vertical (Y) axis; positive moves image
    /// content leftward (viewpoint orbits rightward around the pivot).
    public var yawDeg: Float
    /// Viewpoint pitch in degrees; positive moves image content downward.
    public var pitchDeg: Float
    /// Pinhole focal length in canvas pixels. nil -> max(canvasWidth, canvasHeight).
    public var focal: Float?
    /// Rotation pivot depth in source depth units. nil -> mean source depth.
    public var pivotZ: Float?
    /// Small-hole diffusion fill radius in px (0 disables filling).
    public var fillRadius: Int
    /// Blend jagged splat edges with a 3x3 gaussian on the color.
    public var softenEdges: Bool

    public init(yawDeg: Float = 0, pitchDeg: Float = 0, focal: Float? = nil,
                pivotZ: Float? = nil, fillRadius: Int = 4, softenEdges: Bool = true) {
        self.yawDeg = yawDeg
        self.pitchDeg = pitchDeg
        self.focal = focal
        self.pivotZ = pivotZ
        self.fillRadius = fillRadius
        self.softenEdges = softenEdges
    }

    /// Per-step micro-move limit (degrees).
    public static let maxStepDeg: Float = 5
    /// Absolute per-axis travel clamp (degrees).
    public static let maxTotalDeg: Float = 30
}

/// Metal-side constants; layout must match `ReprojectParams` in DepthShaders.metal.
struct ReprojectShaderParams {
    var f: Float
    var cx: Float, cy: Float
    var pivotZ: Float
    var cyw: Float, syw: Float
    var cpt: Float, spt: Float
    var invZMax: Float
    var sw: UInt32, sh: UInt32
    var cw: UInt32, ch: UInt32
}

/// Metal-side constants for the edge-soften pass; matches `SoftenParams`.
struct SoftenShaderParams {
    var depthThreshold: Float
    var mix: Float
}

public struct ReprojectResult {
    /// rgba16Float, alpha 1 where covered (before fill); filled/softened color after.
    public let color: MTLTexture
    public let depth: DepthTexture
    /// 1 where covered or small-hole-filled, 0 in unfilled big holes.
    public let mask: DepthTexture
    public let width: Int
    public let height: Int
    /// Number of recursive micro-steps applied (1 when within the step limit).
    public var steps: Int
    /// Requested angles after the ±maxTotalDeg clamp (what was actually applied).
    public var appliedYawDeg: Float
    public var appliedPitchDeg: Float

    /// Fraction of canvas pixels valid after filling.
    public var coverage: Float {
        let m = mask.readback()
        return m.reduce(0) { $0 + ($1 > 0.5 ? 1 : 0) } / Float(m.count)
    }

    /// Warped color as RGBA8 bytes (row-major).
    public func colorRGBA8() -> [UInt8] {
        var half = [UInt16](repeating: 0, count: width * height * 4)
        half.withUnsafeMutableBytes { ptr in
            color.getBytes(ptr.baseAddress!, bytesPerRow: width * 4 * 2,
                           from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        return half.map { UInt8(clamping: Int((Float(Float16(bitPattern: $0)) * 255).rounded())) }
    }
}

extension DepthOps {
    /// Single micro-step reprojection of `depth` (arbitrary resolution,
    /// nearest-upsampled to the canvas) paired with `colorRGBA` (RGBA8, canvas-sized).
    /// No travel limit is applied here — use `reprojectView` for clamped/recursive moves.
    public func reproject(depth: DepthTexture, colorRGBA: [UInt8],
                          canvasWidth cw: Int, canvasHeight ch: Int,
                          params: ReprojectParams) throws -> ReprojectResult {
        precondition(colorRGBA.count == cw * ch * 4, "colorRGBA must be canvas-sized RGBA8")
        let srcValues = depth.readback()
        let pivotZ = params.pivotZ ?? srcValues.reduce(0, +) / Float(srcValues.count)
        let zMax = max((srcValues.max() ?? 1) * 2, pivotZ * 4, 1e-3)
        let color = try uploadColor(colorRGBA, width: cw, height: ch)
        var r = try reprojectStep(depth: depth.texture, color: color, canvasWidth: cw,
                                  canvasHeight: ch, params: params, pivotZ: pivotZ, zMax: zMax)
        r.steps = 1
        r.appliedYawDeg = params.yawDeg
        r.appliedPitchDeg = params.pitchDeg
        return r
    }

    /// Clamped, recursively micro-stepped reprojection. The requested yaw/pitch are
    /// clamped to ±maxTotalDeg, then split into equal steps of at most maxStepDeg;
    /// each step splats the previous step's warped output (fixed pivot from the
    /// source frame's mean depth).
    public func reprojectView(depth: DepthTexture, colorRGBA: [UInt8],
                              canvasWidth cw: Int, canvasHeight ch: Int,
                              params: ReprojectParams) throws -> ReprojectResult {
        precondition(colorRGBA.count == cw * ch * 4, "colorRGBA must be canvas-sized RGBA8")
        let limit = ReprojectParams.maxTotalDeg
        let yaw = min(max(params.yawDeg, -limit), limit)
        let pitch = min(max(params.pitchDeg, -limit), limit)
        let steps = max(1, Int(ceil(max(abs(yaw), abs(pitch)) / ReprojectParams.maxStepDeg)))

        let srcValues = depth.readback()
        let pivotZ = params.pivotZ ?? srcValues.reduce(0, +) / Float(srcValues.count)
        let zMax = max((srcValues.max() ?? 1) * 2, pivotZ * 4, 1e-3)

        var step = params
        step.yawDeg = yaw / Float(steps)
        step.pitchDeg = pitch / Float(steps)

        var curDepth = depth.texture
        var curColor = try uploadColor(colorRGBA, width: cw, height: ch)
        var result = try reprojectStep(depth: curDepth, color: curColor, canvasWidth: cw,
                                       canvasHeight: ch, params: step, pivotZ: pivotZ, zMax: zMax)
        for _ in 1..<steps {
            curDepth = result.depth.texture
            curColor = result.color
            result = try reprojectStep(depth: curDepth, color: curColor, canvasWidth: cw,
                                       canvasHeight: ch, params: step, pivotZ: pivotZ, zMax: zMax)
        }
        result.steps = steps
        result.appliedYawDeg = yaw
        result.appliedPitchDeg = pitch
        return result
    }

    /// Values-level convenience used by the parity runner (single step).
    public func reproject(values: [Float], width sw: Int, height sh: Int,
                          colorRGBA: [UInt8], canvasWidth cw: Int, canvasHeight ch: Int,
                          params: ReprojectParams) throws
        -> (rgba: [UInt8], depth: [Float], mask: [Float]) {
        let result = try reproject(depth: DepthTexture(values: values, width: sw, height: sh),
                                   colorRGBA: colorRGBA, canvasWidth: cw, canvasHeight: ch,
                                   params: params)
        return (result.colorRGBA8(), result.depth.readback(), result.mask.readback())
    }

    // MARK: - internals

    private func uploadColor(_ rgba: [UInt8], width w: Int, height h: Int) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w,
                                                         height: h, mipmapped: false)
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let t = context.device.makeTexture(descriptor: d) else {
            throw DepthShaderError.textureCreationFailed
        }
        rgba.withUnsafeBytes { ptr in
            t.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                      withBytes: ptr.baseAddress!, bytesPerRow: w * 4)
        }
        return t
    }

    /// One splat + fill + soften frame. `depth`/`color` are canvas- or source-sized
    /// textures; the vertex stage nearest-maps depth to the canvas.
    private func reprojectStep(depth: MTLTexture, color: MTLTexture,
                               canvasWidth cw: Int, canvasHeight ch: Int,
                               params: ReprojectParams, pivotZ: Float, zMax: Float) throws -> ReprojectResult {
        let f = params.focal ?? Float(max(cw, ch))
        let yaw = params.yawDeg * .pi / 180
        let pitch = params.pitchDeg * .pi / 180
        var sp = ReprojectShaderParams(
            f: f, cx: Float(cw - 1) / 2, cy: Float(ch - 1) / 2, pivotZ: pivotZ,
            cyw: cos(yaw), syw: sin(yaw), cpt: cos(pitch), spt: sin(pitch),
            invZMax: 1 / zMax,
            sw: UInt32(depth.width), sh: UInt32(depth.height),
            cw: UInt32(cw), ch: UInt32(ch))
        var soften = SoftenShaderParams(depthThreshold: 0.05 * pivotZ, mix: 0.5)

        let device = context.device
        func makeTexture(_ format: MTLPixelFormat,
                         _ usage: MTLTextureUsage, storage: MTLStorageMode = .shared) throws -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: cw,
                                                             height: ch, mipmapped: false)
            d.usage = usage
            d.storageMode = storage
            guard let t = device.makeTexture(descriptor: d) else {
                throw DepthShaderError.textureCreationFailed
            }
            return t
        }

        var outColor = try makeTexture(.rgba16Float, [.renderTarget, .shaderRead, .shaderWrite])
        var outDepth = try makeTexture(.r32Float, [.renderTarget, .shaderRead, .shaderWrite])
        var outMask = try makeTexture(.r32Float, [.renderTarget, .shaderRead, .shaderWrite])
        let zbuf = try makeTexture(.depth32Float, [.renderTarget], storage: .private)

        let renderPipeline = try context.renderPipeline(
            vertex: "reproject_vertex", fragment: "reproject_fragment",
            colorFormats: [.rgba16Float, .r32Float, .r32Float], depthFormat: .depth32Float)
        let depthStencil = context.depthStencil(compare: .less)
        let fillPipeline = try context.pipeline(function: "hole_fill_step")
        let softenPipeline = try context.pipeline(function: "edge_soften")

        try context.encodeFrame { commandBuffer in
            let rpd = MTLRenderPassDescriptor()
            rpd.colorAttachments[0].texture = outColor
            rpd.colorAttachments[0].loadAction = .clear
            rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            rpd.colorAttachments[0].storeAction = .store
            rpd.colorAttachments[1].texture = outDepth
            rpd.colorAttachments[1].loadAction = .clear
            rpd.colorAttachments[1].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            rpd.colorAttachments[1].storeAction = .store
            rpd.colorAttachments[2].texture = outMask
            rpd.colorAttachments[2].loadAction = .clear
            rpd.colorAttachments[2].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            rpd.colorAttachments[2].storeAction = .store
            rpd.depthAttachment.texture = zbuf
            rpd.depthAttachment.loadAction = .clear
            rpd.depthAttachment.clearDepth = 1.0
            rpd.depthAttachment.storeAction = .dontCare

            guard let enc = commandBuffer.makeRenderCommandEncoder(descriptor: rpd) else {
                throw DepthShaderError.encodingFailed
            }
            enc.setRenderPipelineState(renderPipeline)
            enc.setDepthStencilState(depthStencil)
            enc.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(cw),
                                        height: Double(ch), znear: 0, zfar: 1))
            enc.setVertexTexture(depth, index: 0)
            enc.setVertexBytes(&sp, length: MemoryLayout<ReprojectShaderParams>.stride, index: 0)
            enc.setFragmentTexture(color, index: 0)
            enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: cw * ch)
            enc.endEncoding()

            func fillPass(_ body: (MTLComputeCommandEncoder) -> Void) throws {
                guard let enc = commandBuffer.makeComputeCommandEncoder() else {
                    throw DepthShaderError.encodingFailed
                }
                body(enc)
                let threads = MTLSize(width: 16, height: 16, depth: 1)
                enc.dispatchThreadgroups(MTLSize(width: (cw + 15) / 16, height: (ch + 15) / 16, depth: 1),
                                         threadsPerThreadgroup: threads)
                enc.endEncoding()
            }

            // small-hole diffusion fill, ping-pong between two reusable scratch sets
            if params.fillRadius > 0 {
                let scratchColor = try makeTexture(.rgba16Float, [.shaderRead, .shaderWrite])
                let scratchDepth = try makeTexture(.r32Float, [.shaderRead, .shaderWrite])
                let scratchMask = try makeTexture(.r32Float, [.shaderRead, .shaderWrite])
                for i in 0..<params.fillRadius {
                    let (dIn, mIn, cIn): (MTLTexture, MTLTexture, MTLTexture)
                    if i % 2 == 0 {
                        dIn = outDepth; mIn = outMask; cIn = outColor
                    } else {
                        dIn = scratchDepth; mIn = scratchMask; cIn = scratchColor
                    }
                    let (dOut, mOut, cOut): (MTLTexture, MTLTexture, MTLTexture)
                    if i % 2 == 0 {
                        dOut = scratchDepth; mOut = scratchMask; cOut = scratchColor
                    } else {
                        dOut = outDepth; mOut = outMask; cOut = outColor
                    }
                    try fillPass { enc in
                        enc.setComputePipelineState(fillPipeline)
                        enc.setTexture(dIn, index: 0)
                        enc.setTexture(mIn, index: 1)
                        enc.setTexture(cIn, index: 2)
                        enc.setTexture(dOut, index: 3)
                        enc.setTexture(mOut, index: 4)
                        enc.setTexture(cOut, index: 5)
                    }
                }
                if params.fillRadius % 2 == 1 {
                    outColor = scratchColor
                    outDepth = scratchDepth
                    outMask = scratchMask
                }
            }

            // soften jagged splat edges on the color (depth keeps its true values)
            if params.softenEdges {
                let softColor = try makeTexture(.rgba16Float, [.shaderRead, .shaderWrite])
                try fillPass { enc in
                    enc.setComputePipelineState(softenPipeline)
                    enc.setTexture(outDepth, index: 0)
                    enc.setTexture(outMask, index: 1)
                    enc.setTexture(outColor, index: 2)
                    enc.setTexture(softColor, index: 3)
                    enc.setBytes(&soften, length: MemoryLayout<SoftenShaderParams>.stride, index: 0)
                }
                outColor = softColor
            }
        }

        return ReprojectResult(color: outColor,
                               depth: DepthTexture(texture: outDepth),
                               mask: DepthTexture(texture: outMask),
                               width: cw, height: ch,
                               steps: 1, appliedYawDeg: params.yawDeg,
                               appliedPitchDeg: params.pitchDeg)
    }
}
