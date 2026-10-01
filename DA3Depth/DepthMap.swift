import Foundation
import AppKit
import ImageIO
import UniformTypeIdentifiers

/// Float depth map + rendering (gray / inferno pseudocolor) + PNG export (8-bit & 16-bit).
struct DepthMap {
    var values: [Float]  // row-major
    var width: Int
    var height: Int

    func flippedHorizontal() -> DepthMap {
        var out = [Float](repeating: 0, count: values.count)
        for y in 0..<height {
            for x in 0..<width {
                out[y * width + x] = values[y * width + (width - 1 - x)]
            }
        }
        return DepthMap(values: out, width: width, height: height)
    }

    /// Bilinear upscale to target size.
    func resized(to tw: Int, _ th: Int) -> DepthMap {
        guard tw != width || th != height else { return self }
        var out = [Float](repeating: 0, count: tw * th)
        let sx = Double(width) / Double(tw), sy = Double(height) / Double(th)
        for y in 0..<th {
            let fy = (Double(y) + 0.5) * sy - 0.5
            let y0 = max(0, Int(fy.rounded(.down)))
            let y1 = min(height - 1, y0 + 1)
            let wy = Float(fy - Double(y0))
            for x in 0..<tw {
                let fx = (Double(x) + 0.5) * sx - 0.5
                let x0 = max(0, Int(fx.rounded(.down)))
                let x1 = min(width - 1, x0 + 1)
                let wx = Float(fx - Double(x0))
                let a = values[y0 * width + x0], b = values[y0 * width + x1]
                let c = values[y1 * width + x0], d = values[y1 * width + x1]
                out[y * tw + x] = a * (1 - wx) * (1 - wy) + b * wx * (1 - wy)
                                + c * (1 - wx) * wy + d * wx * wy
            }
        }
        return DepthMap(values: out, width: tw, height: th)
    }

    /// min-max normalized to [0,1]: near (small depth) -> 0 (black), far -> 1 (white)
    func normalized() -> [Float] {
        guard let lo = values.min(), let hi = values.max(), hi > lo else {
            return [Float](repeating: 0, count: values.count)
        }
        let inv = 1.0 / (hi - lo)
        return values.map { ($0 - lo) * inv }
    }

    private func gray8() -> [UInt8] {
        normalized().map { UInt8(min(255, max(0, ($0 * 255).rounded()))) }
    }

    private func rgb8() -> [UInt8] {
        let n = normalized()
        var out = [UInt8](repeating: 0, count: n.count * 3)
        for i in 0..<n.count {
            let idx = min(255, max(0, Int((n[i] * 255).rounded())))
            let c = Colormap.infernoLUT[idx]
            out[i * 3] = c.0; out[i * 3 + 1] = c.1; out[i * 3 + 2] = c.2
        }
        return out
    }

    func grayCGImage() -> CGImage? {
        let bytes = gray8()
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        return CGImage(width: width, height: height,
                       bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width,
                       space: CGColorSpaceCreateDeviceGray(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                       provider: provider, decode: nil,
                       shouldInterpolate: true, intent: .defaultIntent)
    }

    func colorCGImage() -> CGImage? {
        let bytes = rgb8()
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        return CGImage(width: width, height: height,
                       bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: width * 3,
                       space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                       provider: provider, decode: nil,
                       shouldInterpolate: true, intent: .defaultIntent)
    }

    func pngGray8() -> Data? { pngData(grayCGImage()) }
    func pngColor8() -> Data? { pngData(colorCGImage()) }

    /// 16-bit grayscale PNG, near=0 far=65535.
    func pngGray16() -> Data? {
        let n = normalized()
        var words = n.map { UInt16(min(65535, max(0, ($0 * 65535).rounded()))) }
        guard let cg = words.withUnsafeMutableBytes({ ptr -> CGImage? in
            guard let provider = CGDataProvider(data: Data(bytes: ptr.baseAddress!, count: ptr.count) as CFData)
            else { return nil }
            return CGImage(width: width, height: height,
                           bitsPerComponent: 16, bitsPerPixel: 16, bytesPerRow: width * 2,
                           space: CGColorSpaceCreateDeviceGray(),
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue
                                                    | CGBitmapInfo.byteOrder16Little.rawValue),
                           provider: provider, decode: nil,
                           shouldInterpolate: false, intent: .defaultIntent)
        }) else { return nil }
        return pngData(cg)
    }

    private func pngData(_ cg: CGImage?) -> Data? {
        guard let cg else { return nil }
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }
}
