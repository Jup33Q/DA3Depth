import Foundation
import CoreML
import AppKit

/// CoreML inference engine for DA3MONO-LARGE slim models (raw depth + sky logits).
/// Preprocessing mirrors the official pipeline: longest side -> 504 (upper_bound_resize),
/// each dimension rounded to nearest multiple of 14, ImageNet normalization.
/// Non-3:4 aspect ratios are center-padded (black) into the nearest fixed model shape
/// and the padding is cropped from the output depth.
final class DepthEngine: @unchecked Sendable {
    static let modelsDir = FileManager.default.homeDirectoryForCurrentUser
        .appending(path: "dev/DA3Depth/models")

    enum Orientation: String {
        case portrait = "504x378"   // H=504, W=378
        case landscape = "378x504"  // H=378, W=504
        var hw: (Int, Int) { self == .portrait ? (504, 378) : (378, 504) }
    }

    private var models: [Orientation: MLModel] = [:]
    private let lock = NSLock()

    struct Result {
        var depth: [Float]   // cropped, sky-processed, row-major (th x tw)
        var width: Int
        var height: Int
        var wasPadded: Bool
        var inferenceMs: Double
    }

    enum EngineError: LocalizedError {
        case modelMissing(String)
        case badImage
        var errorDescription: String? {
            switch self {
            case .modelMissing(let p): return "模型文件缺失: \(p)"
            case .badImage: return "无法解码图片"
            }
        }
    }

    private func model(for o: Orientation) throws -> MLModel {
        lock.lock()
        if let m = models[o] { lock.unlock(); return m }
        lock.unlock()
        let url = Self.modelsDir.appending(path: "DA3MonoLarge_\(o.rawValue).mlpackage")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw EngineError.modelMissing(url.path)
        }
        // MLModel(contentsOf:) cannot auto-compile .mlpackage in an adhoc-signed app;
        // compile explicitly and cache the .mlmodelc next to the models.
        let fm = FileManager.default
        let compiledURL = Self.modelsDir.appending(path: "compiled/DA3MonoLarge_\(o.rawValue).mlmodelc")
        let srcDate = (try? fm.attributesOfItem(atPath: url.path)[.modificationDate] as? Date) ?? .distantFuture
        let dstDate = (try? fm.attributesOfItem(atPath: compiledURL.path)[.modificationDate] as? Date) ?? .distantPast
        if !fm.fileExists(atPath: compiledURL.path) || dstDate < srcDate {
            try? fm.removeItem(at: compiledURL)
            try fm.createDirectory(at: compiledURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            let tmp = try MLModel.compileModel(at: url)
            try fm.moveItem(at: tmp, to: compiledURL)
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        let m = try MLModel(contentsOf: compiledURL, configuration: cfg)
        lock.lock(); models[o] = m; lock.unlock()
        return m
    }

    // MARK: - Preprocess

    private static let mean: [Float] = [0.485, 0.456, 0.406]
    private static let std: [Float] = [0.229, 0.224, 0.225]

    private static func rounded14(_ x: Int) -> Int { max(14, Int((Double(x) / 14.0).rounded()) * 14) }

    /// Render source image into a black (mw x mh) RGBA buffer with the resized image centered.
    /// Returns (rgba bytes, targetW, targetH within buffer, padded flag).
    private func rasterize(_ cg: CGImage, modelW: Int, modelH: Int)
        -> (bytes: [UInt8], tw: Int, th: Int, padded: Bool)?
    {
        let ow = cg.width, oh = cg.height
        // official: longest side -> 504
        let scale = 504.0 / Double(max(ow, oh))
        var tw = Self.rounded14(Int((Double(ow) * scale).rounded()))
        var th = Self.rounded14(Int((Double(oh) * scale).rounded()))
        var padded = false
        if tw > modelW || th > modelH {
            // aspect not covered by fixed model shape: fit inside model box, pad rest
            let s2 = min(Double(modelW) / Double(ow), Double(modelH) / Double(oh))
            tw = Self.rounded14(Int((Double(ow) * s2).rounded()))
            th = Self.rounded14(Int((Double(oh) * s2).rounded()))
            padded = true
        }
        var bytes = [UInt8](repeating: 0, count: modelW * modelH * 4)
        let ok = bytes.withUnsafeMutableBytes { ptr -> Bool in
            guard let ctx = CGContext(
                data: ptr.baseAddress, width: modelW, height: modelH,
                bitsPerComponent: 8, bytesPerRow: modelW * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
            ctx.fill(CGRect(x: 0, y: 0, width: modelW, height: modelH))
            ctx.interpolationQuality = .high
            let dx = (modelW - tw) / 2, dy = (modelH - th) / 2
            ctx.draw(cg, in: CGRect(x: dx, y: dy, width: tw, height: th))
            return true
        }
        guard ok else { return nil }
        return (bytes, tw, th, padded)
    }

    private static func makeInput(_ rgba: [UInt8], _ mw: Int, _ mh: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: mh), NSNumber(value: mw)], dataType: .float32)
        let ptr = arr.dataPointer.bindMemory(to: Float32.self, capacity: 3 * mh * mw)
        let plane = mh * mw
        for y in 0..<mh {
            for x in 0..<mw {
                let s = (y * mw + x) * 4
                let d = y * mw + x
                ptr[d] = (Float32(rgba[s]) / 255.0 - mean[0]) / std[0]
                ptr[plane + d] = (Float32(rgba[s + 1]) / 255.0 - mean[1]) / std[1]
                ptr[2 * plane + d] = (Float32(rgba[s + 2]) / 255.0 - mean[2]) / std[2]
            }
        }
        return arr
    }

    // MARK: - Output reading

    /// Stride-aware copy of a (1, H, W) multi-array into a dense row-major [Float].
    private static func toFloatArray(_ ma: MLMultiArray) -> [Float] {
        let shape = ma.shape.map { $0.intValue }
        let strides = ma.strides.map { $0.intValue }  // element strides
        let h = shape.count >= 2 ? shape[shape.count - 2] : 1
        let w = shape.count >= 1 ? shape[shape.count - 1] : ma.count
        let sh = strides.count >= 2 ? strides[strides.count - 2] : w
        let sw = strides.count >= 1 ? strides[strides.count - 1] : 1
        var out = [Float](repeating: 0, count: h * w)
        switch ma.dataType {
        case .float32:
            let p = ma.dataPointer.bindMemory(to: Float32.self, capacity: ma.count)
            for y in 0..<h { for x in 0..<w { out[y * w + x] = Float(p[y * sh + x * sw]) } }
        case .float16:
            let p = ma.dataPointer.bindMemory(to: Float16.self, capacity: ma.count)
            for y in 0..<h { for x in 0..<w { out[y * w + x] = Float(p[y * sh + x * sw]) } }
        case .double:
            let p = ma.dataPointer.bindMemory(to: Double.self, capacity: ma.count)
            for y in 0..<h { for x in 0..<w { out[y * w + x] = Float(p[y * sh + x * sw]) } }
        default:
            for y in 0..<h { for x in 0..<w { out[y * w + x] = ma[[0, y, x] as [NSNumber]].floatValue } }
        }
        return out
    }

    /// DepthAnything3Net._process_mono_sky_estimation replica:
    /// sky = skyLogit >= 0.3; sky pixels set to 0.99-quantile of non-sky depth.
    private static func applySky(depth: inout [Float], sky: [Float]) {
        var nonSky = [Float]()
        nonSky.reserveCapacity(depth.count)
        var skyCount = 0
        for i in 0..<depth.count {
            if sky[i] < 0.3 { nonSky.append(depth[i]) } else { skyCount += 1 }
        }
        guard nonSky.count > 10, skyCount > 10 else { return }
        nonSky.sort()
        let q = nonSky[min(nonSky.count - 1, Int((Double(nonSky.count - 1) * 0.99).rounded()))]
        for i in 0..<depth.count where sky[i] >= 0.3 { depth[i] = q }
    }

    // MARK: - Inference

    func infer(_ image: NSImage) throws -> Result {
        guard let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let cg = rep.cgImage else { throw EngineError.badImage }

        let orientation: Orientation = cg.height >= cg.width ? .portrait : .landscape
        let (mh, mw) = orientation.hw
        let model = try model(for: orientation)

        guard let (rgba, tw, th, padded) = rasterize(cg, modelW: mw, modelH: mh) else {
            throw EngineError.badImage
        }
        let input = try Self.makeInput(rgba, mw, mh)

        let t0 = CFAbsoluteTimeGetCurrent()
        let out = try model.prediction(from: try MLDictionaryFeatureProvider(
            dictionary: ["image": MLFeatureValue(multiArray: input)]))
        let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        guard let depthMA = out.featureValue(for: "depth")?.multiArrayValue,
              let skyMA = out.featureValue(for: "sky")?.multiArrayValue else {
            throw EngineError.badImage
        }
        var depth = Self.toFloatArray(depthMA)
        let sky = Self.toFloatArray(skyMA)
        Self.applySky(depth: &depth, sky: sky)

        // crop padding (centered)
        if padded || tw != mw || th != mh {
            let ox = (mw - tw) / 2, oy = (mh - th) / 2
            var cropped = [Float](repeating: 0, count: tw * th)
            for y in 0..<th {
                for x in 0..<tw {
                    cropped[y * tw + x] = depth[(y + oy) * mw + (x + ox)]
                }
            }
            depth = cropped
        }
        return Result(depth: depth, width: tw, height: th, wasPadded: padded, inferenceMs: ms)
    }
}
