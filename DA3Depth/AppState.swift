import SwiftUI
import ImageIO
import UniformTypeIdentifiers

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
        case warped_color, warped_depth16, warped_mask
        var suffix: String {
            switch self {
            case .gray8: "depth_gray"
            case .color8: "depth_color"
            case .gray16: "depth_16bit"
            case .warped_color: "warped_color"
            case .warped_depth16: "warped_depth_16bit"
            case .warped_mask: "warped_mask"
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
    /// Edit-layer params (degrees for rotation), mirrored into the 编辑 panel.
    struct EditParams: Equatable {
        var deg: Float = 0, tx: Float = 0, ty: Float = 0, scale: Float = 1, dz: Float = 0
    }
    @Published var editParams = EditParams()
    @Published var editCanUndo = false
    @Published var editCanRedo = false
    @Published var hasEdits = false

    let engine = DepthEngine()
    let editEngine = EditEngine()
    var lastPath: String?
    /// Pristine inferred depth, kept for edit_reset.
    private var rawDepth: DepthMap?

    /// Latest reproject_view (M7) output, at input-image resolution.
    struct WarpedResult {
        var rgba: [UInt8]
        var depth: [Float]
        var mask: [Float]
        var width: Int
        var height: Int
    }
    var warped: WarpedResult?

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
            rawDepth = depth
            try await editEngine.load(values: r.depth, width: r.width, height: r.height)
            editParams = EditParams()
            editCanUndo = false
            editCanRedo = false
            hasEdits = false
            wasPadded = r.wasPadded
            lastInferenceMs = r.inferenceMs
            warped = nil  // prior reprojection no longer matches the new frame
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
        switch kind {
        case .warped_color, .warped_depth16, .warped_mask:
            return try exportWarped(kind, to: url)
        case .gray8, .color8, .gray16:
            break
        }
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
        case .warped_color, .warped_depth16, .warped_mask: nil  // handled above
        }
        guard let data else { throw DepthEngine.EngineError.badImage }
        try data.write(to: url)
        return "已导出: \(url.lastPathComponent)（\(up.width)x\(up.height)）"
    }

    /// Export the latest reproject_view output: warped RGB / warped gray16 depth
    /// (normalized over mask-valid pixels, holes = 0) / validity mask.
    private func exportWarped(_ kind: ExportKind, to url: URL) throws -> String {
        guard let w = warped else {
            throw DepthEngine.EngineError.badImage
        }
        let cg: CGImage?
        switch kind {
        case .warped_color:
            cg = Self.rgbaCGImage(w.rgba, width: w.width, height: w.height)
        case .warped_depth16:
            var lo = Float.greatestFiniteMagnitude, hi = -Float.greatestFiniteMagnitude
            for i in 0..<w.depth.count where w.mask[i] > 0.5 {
                lo = min(lo, w.depth[i]); hi = max(hi, w.depth[i])
            }
            guard hi > lo else { throw DepthEngine.EngineError.badImage }
            let inv = 1 / (hi - lo)
            var words = [UInt16](repeating: 0, count: w.depth.count)
            for i in 0..<w.depth.count where w.mask[i] > 0.5 {
                words[i] = UInt16(min(65535, max(0, ((w.depth[i] - lo) * inv * 65535).rounded())))
            }
            cg = words.withUnsafeMutableBytes { ptr in
                guard let provider = CGDataProvider(data: Data(bytes: ptr.baseAddress!, count: ptr.count) as CFData)
                else { return nil }
                return CGImage(width: w.width, height: w.height,
                               bitsPerComponent: 16, bitsPerPixel: 16, bytesPerRow: w.width * 2,
                               space: CGColorSpaceCreateDeviceGray(),
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue
                                                        | CGBitmapInfo.byteOrder16Little.rawValue),
                               provider: provider, decode: nil,
                               shouldInterpolate: false, intent: .defaultIntent)
            }
        case .warped_mask:
            let bytes = w.mask.map { UInt8($0 > 0.5 ? 255 : 0) }
            guard let provider = CGDataProvider(data: Data(bytes) as CFData) else {
                throw DepthEngine.EngineError.badImage
            }
            cg = CGImage(width: w.width, height: w.height,
                         bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: w.width,
                         space: CGColorSpaceCreateDeviceGray(),
                         bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                         provider: provider, decode: nil,
                         shouldInterpolate: false, intent: .defaultIntent)
        case .gray8, .color8, .gray16:
            fatalError("unreachable")
        }
        guard let cg, let data = Self.pngData(cg) else {
            throw DepthEngine.EngineError.badImage
        }
        try data.write(to: url)
        return "已导出: \(url.lastPathComponent)（\(w.width)x\(w.height)）"
    }

    // MARK: - 2.5D 重投影（M7，MCP 驱动）

    /// Reproject the current view around the vertical axis (yaw) / horizontal axis
    /// (pitch) through the pivot depth plane. Angles clamp to ±30° and run as
    /// recursive <=5° micro-steps. Output pairs warped RGB + warped depth + mask.
    @discardableResult
    func reprojectView(yawDeg: Float, pitchDeg: Float = 0, fillHoles: Bool = true,
                       softenEdges: Bool = true) async throws -> String {
        guard let d = displayDepth, let cg = inputCGImage else {
            throw DepthEngine.EngineError.badImage
        }
        let w = cg.width, h = cg.height
        guard var rgba = Self.rgbaBytes(of: cg) else {
            throw DepthEngine.EngineError.badImage
        }
        if flipped { Self.flipRGBA(&rgba, width: w, height: h) }  // match displayDepth
        processing = true
        status = "重投影中…"
        defer { processing = false }
        let r = try await editEngine.reproject(depthValues: d.values, depthWidth: d.width,
                                               depthHeight: d.height, rgba: rgba,
                                               canvasWidth: w, canvasHeight: h,
                                               yawDeg: yawDeg, pitchDeg: pitchDeg,
                                               fillRadius: fillHoles ? 4 : 0,
                                               softenEdges: softenEdges)
        warped = WarpedResult(rgba: r.rgba, depth: r.depth, mask: r.mask, width: w, height: h)
        let msg = String(format: "已重投影: yaw %.2f°→%.2f° pitch %.2f°→%.2f° · %dx%d · %d 微步 · 覆盖率 %.1f%%",
                         yawDeg, r.appliedYaw, pitchDeg, r.appliedPitch, w, h, r.steps,
                         r.coverage * 100)
        status = msg
        return msg
    }

    // MARK: - RGBA / PNG helpers

    static func rgbaBytes(of cg: CGImage) -> [UInt8]? {
        let w = cg.width, h = cg.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        let ok = bytes.withUnsafeMutableBytes { ptr -> Bool in
            guard let ctx = CGContext(data: ptr.baseAddress, width: w, height: h,
                                      bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        return ok ? bytes : nil
    }

    static func flipRGBA(_ bytes: inout [UInt8], width w: Int, height h: Int) {
        for y in 0..<h {
            for x in 0..<w / 2 {
                let a = (y * w + x) * 4, b = (y * w + (w - 1 - x)) * 4
                for c in 0..<4 { bytes.swapAt(a + c, b + c) }
            }
        }
    }

    static func rgbaCGImage(_ bytes: [UInt8], width w: Int, height h: Int) -> CGImage? {
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        return CGImage(width: w, height: h,
                       bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                       space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil,
                       shouldInterpolate: true, intent: .defaultIntent)
    }

    static func pngData(_ cg: CGImage) -> Data? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }

    // MARK: - 编辑（单一代码路径：UI 与 MCP 共用，结果逐像素一致）

    private static func checksum(_ values: [Float]) -> Double {
        values.reduce(0.0) { $0 + Double($1) }
    }

    /// Apply transform params to the edit layer (absolute values; nil keeps current).
    /// Returns "canvas WxH checksum <v>".
    @discardableResult
    func applyTransform(rotateDeg: Float? = nil, tx: Float? = nil, ty: Float? = nil,
                        scale: Float? = nil, zShift: Float? = nil) async throws -> String {
        guard rawDepth != nil else { throw DepthEngine.EngineError.badImage }
        let r = try await editEngine.apply(rotateDeg: rotateDeg, tx: tx, ty: ty,
                                           scale: scale, zShift: zShift)
        depth = DepthMap(values: r.values, width: r.width, height: r.height)
        editCanUndo = r.canUndo
        editCanRedo = r.canRedo
        hasEdits = true
        if let t = await editEngine.currentTransform() {
            editParams = EditParams(deg: t.rotation * 180 / .pi, tx: t.tx, ty: t.ty, scale: t.scale, dz: t.zShift)
        }
        return String(format: "canvas %dx%d checksum %.6f", r.width, r.height,
                      Self.checksum(r.values))
    }

    /// Bake the current canvas into a new identity base layer (flatten for cumulative edits).
    @discardableResult
    func fuseEdits() async throws -> String {
        guard rawDepth != nil else { throw DepthEngine.EngineError.badImage }
        let r = try await editEngine.bake()
        depth = DepthMap(values: r.values, width: r.width, height: r.height)
        editParams = EditParams()
        editCanUndo = false
        editCanRedo = false
        return String(format: "已融合为基准层 · canvas %dx%d checksum %.6f",
                      r.width, r.height, Self.checksum(r.values))
    }

    /// Clear all edits back to the raw inferred depth.
    @discardableResult
    func editReset() async throws -> String {
        guard let raw = rawDepth else { throw DepthEngine.EngineError.badImage }
        try await editEngine.load(values: raw.values, width: raw.width, height: raw.height)
        depth = raw
        editParams = EditParams()
        editCanUndo = false
        editCanRedo = false
        hasEdits = false
        return "已重置为原始深度（\(raw.width)x\(raw.height)）"
    }

    @discardableResult
    func editUndo() async throws -> String {
        guard let r = try await editEngine.undo() else { return "无可撤销操作" }
        depth = DepthMap(values: r.values, width: depth!.width, height: depth!.height)
        editCanUndo = r.canUndo
        editCanRedo = r.canRedo
        if let t = await editEngine.currentTransform() {
            editParams = EditParams(deg: t.rotation * 180 / .pi, tx: t.tx, ty: t.ty, scale: t.scale, dz: t.zShift)
        }
        return "已撤销"
    }

    @discardableResult
    func editRedo() async throws -> String {
        guard let r = try await editEngine.redo() else { return "无可重做操作" }
        depth = DepthMap(values: r.values, width: depth!.width, height: depth!.height)
        editCanUndo = r.canUndo
        editCanRedo = r.canRedo
        if let t = await editEngine.currentTransform() {
            editParams = EditParams(deg: t.rotation * 180 / .pi, tx: t.tx, ty: t.ty, scale: t.scale, dz: t.zShift)
        }
        return "已重做"
    }
}
