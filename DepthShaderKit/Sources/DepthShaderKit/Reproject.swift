import Metal
import Foundation

/// Out-of-plane (2.5D) reprojection: back-project depth to a point cloud, rotate the
/// viewpoint about the vertical axis (yaw, plus optional small pitch) through a pivot
/// depth plane, then splat back to the canvas as soft gaussian disc sprites
/// (two-pass visibility splatting: hardware z-test prepass, then gated additive
/// accumulation — rasterization and fixed-function blending, no atomics). Color and
/// depth accumulate with the same weights, so the warped RGB and warped depth stay
/// paired and share one occlusion order. Splat footprints adapt to the projected
/// neighbor spacing (capped, suppressed across depth discontinuities), which keeps
/// the splat coverage free of single-pixel holes. Remaining holes are filled in two
/// stages: small-hole diffusion (promoted to mask=1), then a pull-push pyramid fills
/// every leftover mask=0 pixel (color + depth; farthest-depth pull discipline so
/// foreground never bleeds into the background reveal), and the inpainted band is
/// smoothed with a depth-layer-gated 3x3 gaussian. Filled pixels keep mask=0 to
/// mark them as inpainted, but color/depth are never left black/zero.
///
/// Travel limits: a single step is a micro-move of at most `maxStepDeg`; larger
/// requests are applied recursively as equal sub-steps chained on the previous output
/// (with a fixed pivot from the first frame), and the total per-axis angle is clamped
/// to ±`maxTotalDeg`. Chained steps re-splat only mask=1 content; each step re-fills
/// its own holes. Splat stair-step edges are softened by a 3x3 gaussian blend
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
    public static let maxStepDeg: Float = 1
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
    var hasMask: UInt32
    var depthBreak: Float
    var splatMax: Float
}

/// Metal-side constants for the edge-soften pass; matches `SoftenParams`.
struct SoftenShaderParams {
    var depthThreshold: Float
    var mix: Float
}

/// Metal-side constants for the inpainted-band smoothing pass; matches `HoleSmoothParams`.
struct HoleSmoothShaderParams {
    var depthThreshold: Float
}

public struct ReprojectResult {
    /// rgba16Float, alpha 1 where covered (before fill); filled/softened color after.
    /// Pull-push guarantees no black holes: mask=0 pixels carry inpainted color.
    public let color: MTLTexture
    /// Pull-push-filled: no zero holes; mask=0 pixels carry inpainted background depth.
    public let depth: DepthTexture
    /// 1 where covered or small-hole-filled, 0 on pull-push-filled disocclusion.
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
                                       canvasHeight: ch, params: step, pivotZ: pivotZ,
                                       zMax: zMax, fullFill: steps == 1)
        for i in 1..<steps {
            curDepth = result.depth.texture
            curColor = result.color
            // only real (mask=1) content is re-splatted; each step re-fills its holes.
            // Intermediate steps skip the pull-push/band-smoothing fill: inpainted
            // (mask=0) pixels are culled from the next splat anyway, so only the
            // final frame needs the full-quality fill (bit-identical result).
            result = try reprojectStep(depth: curDepth, color: curColor, mask: result.mask.texture,
                                       canvasWidth: cw, canvasHeight: ch, params: step,
                                       pivotZ: pivotZ, zMax: zMax, fullFill: i == steps - 1)
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
    /// textures; the vertex stage nearest-maps depth to the canvas. `mask` is passed
    /// by chained micro-steps so inpainted (mask 0) pixels are not re-splatted.
    /// `fullFill` runs the pull-push + band-smoothing stages; intermediate chain
    /// steps skip them (mask=0 pixels are culled from the next splat regardless).
    private func reprojectStep(depth: MTLTexture, color: MTLTexture, mask: MTLTexture? = nil,
                               canvasWidth cw: Int, canvasHeight ch: Int,
                               params: ReprojectParams, pivotZ: Float, zMax: Float,
                               fullFill: Bool = true) throws -> ReprojectResult {
        let f = params.focal ?? Float(max(cw, ch))
        let yaw = params.yawDeg * .pi / 180
        let pitch = params.pitchDeg * .pi / 180
        var sp = ReprojectShaderParams(
            f: f, cx: Float(cw - 1) / 2, cy: Float(ch - 1) / 2, pivotZ: pivotZ,
            cyw: cos(yaw), syw: sin(yaw), cpt: cos(pitch), spt: sin(pitch),
            invZMax: 1 / zMax,
            sw: UInt32(depth.width), sh: UInt32(depth.height),
            cw: UInt32(cw), ch: UInt32(ch),
            hasMask: mask != nil ? 1 : 0,
            depthBreak: 0.05 * pivotZ, splatMax: 8)
        var soften = SoftenShaderParams(depthThreshold: 0.05 * pivotZ, mix: 0.5)
        var holeSmooth = HoleSmoothShaderParams(depthThreshold: 0.05 * pivotZ)

        let device = context.device
        func makeTexture(_ format: MTLPixelFormat,
                         _ usage: MTLTextureUsage, storage: MTLStorageMode = .shared,
                         width: Int? = nil, height: Int? = nil) throws -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format,
                                                             width: width ?? cw,
                                                             height: height ?? ch,
                                                             mipmapped: false)
            d.usage = usage
            d.storageMode = storage
            guard let t = device.makeTexture(descriptor: d) else {
                throw DepthShaderError.textureCreationFailed
            }
            return t
        }

        let visDepth = try makeTexture(.r32Float, [.renderTarget, .shaderRead])
        // accumColor stays 32-bit: blending accumulates at attachment precision, and
        // the alpha channel carries the splat weight that normalizes BOTH color and
        // depth — half-precision weight drift visibly biases depth (~1e-3 relative).
        let accumColor = try makeTexture(.rgba32Float, [.renderTarget, .shaderRead])
        let accumDepth = try makeTexture(.r32Float, [.renderTarget, .shaderRead])
        var outColor = try makeTexture(.rgba16Float, [.renderTarget, .shaderRead, .shaderWrite])
        var outDepth = try makeTexture(.r32Float, [.renderTarget, .shaderRead, .shaderWrite])
        let splatMask = try makeTexture(.r32Float, [.renderTarget, .shaderRead, .shaderWrite])
        var outMask = splatMask
        let zbuf = try makeTexture(.depth32Float, [.renderTarget], storage: .private)

        // pass A: hardware z-test visibility (nearest depth per pixel)
        let visPipeline = try context.renderPipeline(
            vertex: "reproject_vertex", fragment: "reproject_vis_fragment",
            colorFormats: [.r32Float], depthFormat: .depth32Float)
        // pass B: additive gaussian accumulation gated by the pass-A depth
        let accumPipeline = try context.renderPipeline(
            vertex: "reproject_vertex", fragment: "reproject_accum_fragment",
            colorFormats: [.rgba32Float, .r32Float], depthFormat: nil, blending: true)
        let depthStencil = context.depthStencil(compare: .less)
        let normalizePipeline = try context.pipeline(function: "splat_normalize")
        let fillPipeline = try context.pipeline(function: "hole_fill_step")
        let softenPipeline = try context.pipeline(function: "edge_soften")
        let pullPipeline = try context.pipeline(function: "pull_push_down")
        let pushPipeline = try context.pipeline(function: "pull_push_up")
        let holeSmoothPipeline = try context.pipeline(function: "hole_smooth")

        try context.encodeFrame { commandBuffer in
            // ---- pass A: visibility splat (z-test) ----
            let rpdA = MTLRenderPassDescriptor()
            rpdA.colorAttachments[0].texture = visDepth
            rpdA.colorAttachments[0].loadAction = .clear
            rpdA.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            rpdA.colorAttachments[0].storeAction = .store
            rpdA.depthAttachment.texture = zbuf
            rpdA.depthAttachment.loadAction = .clear
            rpdA.depthAttachment.clearDepth = 1.0
            rpdA.depthAttachment.storeAction = .dontCare

            guard let encA = commandBuffer.makeRenderCommandEncoder(descriptor: rpdA) else {
                throw DepthShaderError.encodingFailed
            }
            encA.setRenderPipelineState(visPipeline)
            encA.setDepthStencilState(depthStencil)
            encA.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(cw),
                                         height: Double(ch), znear: 0, zfar: 1))
            encA.setVertexTexture(depth, index: 0)
            encA.setVertexTexture(mask ?? depth, index: 1)  // unread when hasMask == 0
            encA.setVertexBytes(&sp, length: MemoryLayout<ReprojectShaderParams>.stride, index: 0)
            encA.drawPrimitives(type: .point, vertexStart: 0, vertexCount: cw * ch)
            encA.endEncoding()

            // ---- pass B: gated gaussian accumulation (additive blending) ----
            let rpdB = MTLRenderPassDescriptor()
            rpdB.colorAttachments[0].texture = accumColor
            rpdB.colorAttachments[0].loadAction = .clear
            rpdB.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            rpdB.colorAttachments[0].storeAction = .store
            rpdB.colorAttachments[1].texture = accumDepth
            rpdB.colorAttachments[1].loadAction = .clear
            rpdB.colorAttachments[1].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            rpdB.colorAttachments[1].storeAction = .store

            guard let encB = commandBuffer.makeRenderCommandEncoder(descriptor: rpdB) else {
                throw DepthShaderError.encodingFailed
            }
            encB.setRenderPipelineState(accumPipeline)
            encB.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(cw),
                                         height: Double(ch), znear: 0, zfar: 1))
            encB.setVertexTexture(depth, index: 0)
            encB.setVertexTexture(mask ?? depth, index: 1)
            encB.setVertexBytes(&sp, length: MemoryLayout<ReprojectShaderParams>.stride, index: 0)
            encB.setFragmentTexture(color, index: 0)
            encB.setFragmentTexture(visDepth, index: 1)
            encB.setFragmentBytes(&sp, length: MemoryLayout<ReprojectShaderParams>.stride, index: 0)
            encB.drawPrimitives(type: .point, vertexStart: 0, vertexCount: cw * ch)
            encB.endEncoding()

            func fillPass(width w: Int, height h: Int,
                          _ body: (MTLComputeCommandEncoder) -> Void) throws {
                guard let enc = commandBuffer.makeComputeCommandEncoder() else {
                    throw DepthShaderError.encodingFailed
                }
                body(enc)
                let threads = MTLSize(width: 16, height: 16, depth: 1)
                enc.dispatchThreadgroups(MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1),
                                         threadsPerThreadgroup: threads)
                enc.endEncoding()
            }

            // ---- pass C: normalize weighted sums -> color/depth/mask ----
            try fillPass(width: cw, height: ch) { enc in
                enc.setComputePipelineState(normalizePipeline)
                enc.setTexture(accumColor, index: 0)
                enc.setTexture(accumDepth, index: 1)
                enc.setTexture(outColor, index: 2)
                enc.setTexture(outDepth, index: 3)
                enc.setTexture(outMask, index: 4)
            }

            var curColor = outColor
            var curDepth = outDepth
            var curMask = outMask

            // small-hole diffusion fill, ping-pong between two reusable scratch sets
            if params.fillRadius > 0 {
                let scratchColor = try makeTexture(.rgba16Float, [.shaderRead, .shaderWrite])
                let scratchDepth = try makeTexture(.r32Float, [.shaderRead, .shaderWrite])
                let scratchMask = try makeTexture(.r32Float, [.shaderRead, .shaderWrite])
                for i in 0..<params.fillRadius {
                    let (dIn, mIn, cIn): (MTLTexture, MTLTexture, MTLTexture)
                    if i % 2 == 0 {
                        dIn = curDepth; mIn = curMask; cIn = curColor
                    } else {
                        dIn = scratchDepth; mIn = scratchMask; cIn = scratchColor
                    }
                    let (dOut, mOut, cOut): (MTLTexture, MTLTexture, MTLTexture)
                    if i % 2 == 0 {
                        dOut = scratchDepth; mOut = scratchMask; cOut = scratchColor
                    } else {
                        dOut = curDepth; mOut = curMask; cOut = curColor
                    }
                    try fillPass(width: cw, height: ch) { enc in
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
                    curColor = scratchColor
                    curDepth = scratchDepth
                    curMask = scratchMask
                }
            }

            // pull-push pyramid fill: every mask=0 pixel gets inpainted color/depth
            // (farthest-depth pull + depth-layer-gated bilinear push), mask untouched.
            // Chained micro-steps cull mask=0 pixels from the next splat, so only the
            // final frame needs this (skipped intermediates are bit-identical).
            var dims = [(cw, ch)]
            while dims.last!.0 > 1 || dims.last!.1 > 1 {
                let (w, h) = dims.last!
                dims.append((max(1, (w + 1) / 2), max(1, (h + 1) / 2)))
            }
            let levels = fullFill ? dims.count : 1
            if levels > 1 {
                var depthP = [curDepth]
                var weightP = [curMask]
                var colorP = [curColor]
                for l in 1..<levels {
                    let (w, h) = dims[l]
                    let dTex = try makeTexture(.r32Float, [.shaderRead, .shaderWrite], width: w, height: h)
                    let wTex = try makeTexture(.r32Float, [.shaderRead, .shaderWrite], width: w, height: h)
                    let cTex = try makeTexture(.rgba16Float, [.shaderRead, .shaderWrite], width: w, height: h)
                    try fillPass(width: w, height: h) { enc in
                        enc.setComputePipelineState(pullPipeline)
                        enc.setTexture(depthP[l - 1], index: 0)
                        enc.setTexture(weightP[l - 1], index: 1)
                        enc.setTexture(colorP[l - 1], index: 2)
                        enc.setTexture(dTex, index: 3)
                        enc.setTexture(wTex, index: 4)
                        enc.setTexture(cTex, index: 5)
                    }
                    depthP.append(dTex)
                    weightP.append(wTex)
                    colorP.append(cTex)
                }
                var fillD = depthP[levels - 1]
                var fillC = colorP[levels - 1]
                for l in stride(from: levels - 2, through: 0, by: -1) {
                    let (w, h) = dims[l]
                    let dTex = try makeTexture(.r32Float, [.shaderRead, .shaderWrite], width: w, height: h)
                    let cTex = try makeTexture(.rgba16Float, [.shaderRead, .shaderWrite], width: w, height: h)
                    try fillPass(width: w, height: h) { enc in
                        enc.setComputePipelineState(pushPipeline)
                        enc.setTexture(depthP[l], index: 0)
                        enc.setTexture(weightP[l], index: 1)
                        enc.setTexture(colorP[l], index: 2)
                        enc.setTexture(fillD, index: 3)
                        enc.setTexture(fillC, index: 4)
                        enc.setTexture(dTex, index: 5)
                        enc.setTexture(cTex, index: 6)
                        enc.setBytes(&holeSmooth, length: MemoryLayout<HoleSmoothShaderParams>.stride,
                                     index: 0)
                    }
                    fillD = dTex
                    fillC = cTex
                }
                curDepth = fillD
                curColor = fillC
            }

            // melt pull-push plateaus in the inpainted band (mask=0 pixels only,
            // depth-layer gated so foreground cannot bleed into the background fill)
            if levels > 1 {
                let scratchD = try makeTexture(.r32Float, [.shaderRead, .shaderWrite])
                let scratchC = try makeTexture(.rgba16Float, [.shaderRead, .shaderWrite])
                for i in 0..<2 {
                    let (dIn, cIn, dOut, cOut) = i % 2 == 0
                        ? (curDepth, curColor, scratchD, scratchC)
                        : (scratchD, scratchC, curDepth, curColor)
                    try fillPass(width: cw, height: ch) { enc in
                        enc.setComputePipelineState(holeSmoothPipeline)
                        enc.setTexture(dIn, index: 0)
                        enc.setTexture(curMask, index: 1)
                        enc.setTexture(cIn, index: 2)
                        enc.setTexture(dOut, index: 3)
                        enc.setTexture(cOut, index: 4)
                        enc.setBytes(&holeSmooth, length: MemoryLayout<HoleSmoothShaderParams>.stride,
                                     index: 0)
                    }
                }
            }

            // soften jagged splat edges on the color (depth keeps its true values)
            if params.softenEdges {
                let softColor = try makeTexture(.rgba16Float, [.shaderRead, .shaderWrite])
                try fillPass(width: cw, height: ch) { enc in
                    enc.setComputePipelineState(softenPipeline)
                    enc.setTexture(curDepth, index: 0)
                    enc.setTexture(curMask, index: 1)
                    enc.setTexture(curColor, index: 2)
                    enc.setTexture(softColor, index: 3)
                    enc.setBytes(&soften, length: MemoryLayout<SoftenShaderParams>.stride, index: 0)
                }
                curColor = softColor
            }
            outColor = curColor
            outDepth = curDepth
            outMask = curMask
        }

        return ReprojectResult(color: outColor,
                               depth: DepthTexture(texture: outDepth),
                               mask: DepthTexture(texture: outMask),
                               width: cw, height: ch,
                               steps: 1, appliedYawDeg: params.yawDeg,
                               appliedPitchDeg: params.pitchDeg)
    }
}
