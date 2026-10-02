import Metal

/// Shared GPU context: default device, one command queue, lazily built pipelines.
public final class GPUContext {
    public static let shared = GPUContext()

    public let device: MTLDevice
    let queue: MTLCommandQueue

    private var pipelines: [String: MTLComputePipelineState] = [:]

    /// GPU execution time of the most recent encode, in seconds.
    public private(set) var lastGPUTime: Double = 0

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
                   _ body: (MTLComputeCommandEncoder) -> T) throws -> T {
        guard let commandBuffer = queue.makeCommandBuffer(),
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
