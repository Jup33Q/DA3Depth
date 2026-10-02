import Metal

/// r32Float texture wrapping a row-major [Float] depth map on the GPU.
public struct DepthTexture {
    public let width: Int
    public let height: Int
    let texture: MTLTexture

    public init(values: [Float], width: Int, height: Int,
                device: MTLDevice = GPUContext.shared.device) throws {
        precondition(values.count == width * height, "values count must equal width * height")
        self.width = width
        self.height = height
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw DepthShaderError.textureCreationFailed
        }
        values.withUnsafeBytes { ptr in
            texture.replace(region: MTLRegionMake2D(0, 0, width, height),
                            mipmapLevel: 0, withBytes: ptr.baseAddress!,
                            bytesPerRow: width * MemoryLayout<Float>.stride)
        }
        self.texture = texture
    }

    public func readback() -> [Float] {
        var out = [Float](repeating: 0, count: width * height)
        out.withUnsafeMutableBytes { ptr in
            texture.getBytes(ptr.baseAddress!, bytesPerRow: width * MemoryLayout<Float>.stride,
                             from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        return out
    }
}
