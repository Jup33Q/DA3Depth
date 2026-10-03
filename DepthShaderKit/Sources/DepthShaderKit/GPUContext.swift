import Metal

/// Shared GPU context: default device, one command queue, lazily built pipelines.
public final class GPUContext {
    public static let shared = GPUContext()

    public let device: MTLDevice
    let queue: MTLCommandQueue

    private var pipelines: [String: MTLComputePipelineState] = [:]
    private var renderPipelines: [String: MTLRenderPipelineState] = [:]
    private var depthStencils: [MTLCompareFunction: MTLDepthStencilState] = [:]

    /// GPU execution time of the most recent encode, in seconds.
    public internal(set) var lastGPUTime: Double = 0

    public init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not available on this system")
        }
        self.device = device
        self.queue = device.makeCommandQueue()!
    }

    func pipeline(function name: String) throws -> MTLComputePipelineState {
        if let cached = pipelines[name] { return cached }
        guard let library = try? device.makeDefaultLibrary(bundle: .module) else {
            throw DepthShaderError.libraryUnavailable
        }
        guard let function = library.makeFunction(name: name) else {
            throw DepthShaderError.kernelNotFound(name)
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        pipelines[name] = pipeline
        return pipeline
    }

    /// Cached render pipeline for point-sprite splatting passes (color attachments +
    /// an optional depth attachment that provides the hardware z-test — no atomics).
    /// `blending` enables additive (one/one) blending on all color attachments for
    /// the gaussian accumulation pass.
    func renderPipeline(vertex vfName: String, fragment ffName: String,
                        colorFormats: [MTLPixelFormat],
                        depthFormat: MTLPixelFormat?,
                        blending: Bool = false) throws -> MTLRenderPipelineState {
        let key = "\(vfName)|\(ffName)|\(colorFormats.map(\.rawValue))|\(depthFormat?.rawValue ?? 0)|\(blending)"
        if let cached = renderPipelines[key] { return cached }
        guard let library = try? device.makeDefaultLibrary(bundle: .module) else {
            throw DepthShaderError.libraryUnavailable
        }
        guard let vf = library.makeFunction(name: vfName),
              let ff = library.makeFunction(name: ffName) else {
            throw DepthShaderError.kernelNotFound("\(vfName)/\(ffName)")
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vf
        descriptor.fragmentFunction = ff
        for (i, format) in colorFormats.enumerated() {
            descriptor.colorAttachments[i].pixelFormat = format
            if blending {
                let a = descriptor.colorAttachments[i]!
                a.isBlendingEnabled = true
                a.sourceRGBBlendFactor = .one
                a.destinationRGBBlendFactor = .one
                a.sourceAlphaBlendFactor = .one
                a.destinationAlphaBlendFactor = .one
            }
        }
        if let depthFormat {
            descriptor.depthAttachmentPixelFormat = depthFormat
        }
        let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        renderPipelines[key] = pipeline
        return pipeline
    }

    func depthStencil(compare: MTLCompareFunction) -> MTLDepthStencilState {
        if let cached = depthStencils[compare] { return cached }
        let descriptor = MTLDepthStencilDescriptor()
        descriptor.depthCompareFunction = compare
        descriptor.isDepthWriteEnabled = true
        let state = device.makeDepthStencilState(descriptor: descriptor)!
        depthStencils[compare] = state
        return state
    }

    /// Wraps a full command-buffer encode (render + optional compute passes) with
    /// one commit+wait, mirroring `encodeBatch` for non-compute work.
    func encodeFrame(_ body: (MTLCommandBuffer) throws -> Void) throws {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw DepthShaderError.encodingFailed
        }
        try body(commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        lastGPUTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    }

    /// One compute dispatch inside a batched command buffer.
    struct EncodeOp {
        let pipeline: MTLComputePipelineState
        let width: Int
        let height: Int
        let body: (MTLComputeCommandEncoder) -> Void
    }

    /// Encodes several dispatches into a single command buffer with one commit+wait,
    /// avoiding per-kernel sync overhead. Encoders run in array order.
    func encodeBatch(_ ops: [EncodeOp]) throws {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw DepthShaderError.encodingFailed
        }
        for op in ops {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw DepthShaderError.encodingFailed
            }
            encoder.setComputePipelineState(op.pipeline)
            op.body(encoder)
            let threads = MTLSize(width: 16, height: 16, depth: 1)
            let groups = MTLSize(width: (op.width + threads.width - 1) / threads.width,
                                 height: (op.height + threads.height - 1) / threads.height,
                                 depth: 1)
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
            encoder.endEncoding()
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        lastGPUTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    }

    func encode<T>(_ pipeline: MTLComputePipelineState, width: Int, height: Int,
                   _ body: (MTLComputeCommandEncoder) -> T) throws -> T {        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DepthShaderError.encodingFailed
        }
        encoder.setComputePipelineState(pipeline)
        let result = body(encoder)
        let threads = MTLSize(width: 16, height: 16, depth: 1)
        let groups = MTLSize(width: (width + threads.width - 1) / threads.width,
                             height: (height + threads.height - 1) / threads.height,
                             depth: 1)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        lastGPUTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        return result
    }
}

public enum DepthShaderError: Error {
    case libraryUnavailable
    case kernelNotFound(String)
    case encodingFailed
    case textureCreationFailed
}
