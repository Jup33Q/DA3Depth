import SwiftUI

/// Shared app state driven by both the SwiftUI UI and the built-in MCP server.
@MainActor
final class AppState: ObservableObject {
    enum ColorMode: String, CaseIterable, Identifiable {
        case gray = "灰度"
        case color = "伪彩 (Inferno)"
        var id: String { rawValue }
    }

    enum ExportKind: String {
        case gray8, color8, gray16
        var suffix: String {
            switch self {
            case .gray8: "depth_gray"
            case .color8: "depth_color"
            case .gray16: "depth_16bit"
            }
        }
    }

    @Published var inputImage: NSImage?
    @Published var inputName = "image"
    @Published var depth: DepthMap?
    @Published var flipped = false
    @Published var mode: ColorMode = .gray
    @Published var processing = false
    @Published var status = "拖入图片开始 · CoreML computeUnits=ALL · FP16（ANE/GPU 自动调度，算子不兼容时自动回退 GPU/CPU）"
    @Published var wasPadded = false
    @Published var lastInferenceMs: Double = 0

    let engine = DepthEngine()
    var lastPath: String?

    var displayDepth: DepthMap? { flipped ? depth?.flippedHorizontal() : depth }

    var inputCGImage: CGImage? {
        guard let img = inputImage, let tiff = img.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff) else { return nil }
        return rep.cgImage
    }

    var previewCGImage: CGImage? {
        guard let d = displayDepth else { return nil }
        return mode == .gray ? d.grayCGImage() : d.colorCGImage()
    }

    /// Load image from disk and run CoreML inference. Returns status line.
    @discardableResult
    func loadAndInfer(path: String) async throws -> String {
        guard let img = NSImage(contentsOfFile: path) else {
            throw DepthEngine.EngineError.badImage
        }
        inputImage = img
        inputName = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
        lastPath = path
        processing = true
        status = "推理中…"
        do {
            let r = try await Task.detached { [engine] in try engine.infer(img) }.value
            depth = DepthMap(values: r.depth, width: r.width, height: r.height)
            wasPadded = r.wasPadded
            lastInferenceMs = r.inferenceMs
            status = String(format: "完成 · %dx%d · 推理 %.0f ms · CoreML ALL units (FP16)",
                            r.width, r.height, r.inferenceMs)
        } catch {
            status = "失败: \(error.localizedDescription)"
            processing = false
            throw error
        }
        processing = false
        return status
    }

    /// Export current depth (upscaled to input resolution, honoring flip) to a PNG file.
    @discardableResult
    func export(_ kind: ExportKind, to url: URL) throws -> String {
        guard let base = depth, let input = inputImage else {
            throw DepthEngine.EngineError.badImage
        }
        let ow = Int(input.size.width), oh = Int(input.size.height)
        var up = (ow > 0 && oh > 0) ? base.resized(to: ow, oh) : base
        if flipped { up = up.flippedHorizontal() }  // flip after upscale: exact mirror of unflipped export
        let data: Data? = switch kind {
        case .gray8: up.pngGray8()
        case .color8: up.pngColor8()
        case .gray16: up.pngGray16()
        }
        guard let data else { throw DepthEngine.EngineError.badImage }
        try data.write(to: url)
        return "已导出: \(url.lastPathComponent)（\(up.width)x\(up.height)）"
    }
}
