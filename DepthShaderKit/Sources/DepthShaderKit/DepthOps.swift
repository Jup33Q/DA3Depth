import Metal

public struct DepthOps {
    public var context: GPUContext
    public init(context: GPUContext = .shared) { self.context = context }

    func makeOutput(width: Int, height: Int) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        guard let texture = context.device.makeTexture(descriptor: descriptor) else {
            throw DepthShaderError.textureCreationFailed
        }
        return texture
    }

    public func flippedHorizontal(_ input: DepthTexture) throws -> DepthTexture {
        let out = try makeOutput(width: input.width, height: input.height)
        let pipeline = try context.pipeline(function: "flip_horizontal")
        try context.encode(pipeline, width: input.width, height: input.height) { encoder in
            encoder.setTexture(input.texture, index: 0)
            encoder.setTexture(out, index: 1)
        }
        return DepthTexture(texture: out)
    }

    public func resized(_ input: DepthTexture, to tw: Int, _ th: Int) throws -> DepthTexture {
        let out = try makeOutput(width: tw, height: th)
        let pipeline = try context.pipeline(function: "resize_bilinear")
        var params = ResizeParams(sx: Float(input.width) / Float(tw),
                                  sy: Float(input.height) / Float(th))
        try context.encode(pipeline, width: tw, height: th) { encoder in
            encoder.setTexture(input.texture, index: 0)
            encoder.setTexture(out, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<ResizeParams>.stride, index: 0)
        }
        return DepthTexture(texture: out)
    }

    public func flippedHorizontal(values: [Float], width: Int, height: Int) throws -> [Float] {
        try flippedHorizontal(DepthTexture(values: values, width: width, height: height)).readback()
    }

    public func resized(values: [Float], width: Int, height: Int,
                        to tw: Int, _ th: Int) throws -> [Float] {
        try resized(DepthTexture(values: values, width: width, height: height), to: tw, th).readback()
    }
}

struct ResizeParams {
    var sx: Float
    var sy: Float
}

extension DepthTexture {
    init(texture: MTLTexture) {
        self.texture = texture
        self.width = texture.width
        self.height = texture.height
    }
}
