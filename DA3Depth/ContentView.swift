import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @ObservedObject var state: AppState
    @State private var isDropTargeted = false
    @State private var editDeg = "0"
    @State private var editTx = "0"
    @State private var editTy = "0"
    @State private var editScale = "1"
    @State private var editDz = "0"
    @State private var dragBaseDeg: Float?
    @State private var dragApplyInFlight = false

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            if state.depth != nil {
                Divider()
                editBar
            }
            Divider()
            HSplitView {
                pane(title: "原图", cgImage: state.inputCGImage)
                    .frame(minWidth: 320)
                pane(title: "深度图（近黑远白）", cgImage: state.previewCGImage)
                    .frame(minWidth: 320)
                    .overlay { if state.processing { ProgressView("推理中…").padding().background(.regularMaterial) } }
                    .gesture(rotationDrag)
            }
            .onDrop(of: [.fileURL, .image], isTargeted: $isDropTargeted) { handleDrop($0) }
            .overlay {
                if state.inputImage == nil {
                    Text("将图片拖到这里\n（或点左上角「打开图片」）")
                        .multilineTextAlignment(.center)
                        .foregroundStyle(.secondary)
                        .font(.title3)
                        .allowsHitTesting(false)
                }
            }
            Divider()
            statusBar
        }
        .background(isDropTargeted ? Color.accentColor.opacity(0.08) : .clear)
    }

    // MARK: - Subviews

    private var toolbar: some View {
        HStack(spacing: 12) {
            Button("打开图片…") { openPanel() }
            Picker("显示", selection: $state.mode) {
                ForEach(AppState.ColorMode.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .frame(width: 220)
            Toggle("水平翻转", isOn: $state.flipped)
                .toggleStyle(.switch)
            Spacer()
            Button("导出灰度 PNG") { exportPanel(.gray8) }.disabled(state.depth == nil)
            Button("导出伪彩 PNG") { exportPanel(.color8) }.disabled(state.depth == nil)
            Button("导出 16-bit PNG") { exportPanel(.gray16) }.disabled(state.depth == nil)
        }
        .padding(10)
    }

    private func pane(title: String, cgImage: CGImage?) -> some View {
        VStack(spacing: 6) {
            Text(title).font(.headline).foregroundStyle(.secondary)
            GeometryReader { geo in
                if let cgImage {
                    Image(cgImage, scale: 1, label: Text(title))
                        .resizable()
                        .scaledToFit()
                        .frame(width: geo.size.width, height: geo.size.height)
                } else {
                    Color.clear
                }
            }
        }
        .padding(10)
    }

    private var statusBar: some View {
        HStack {
            Text(state.status).font(.caption).foregroundStyle(.secondary)
            if state.wasPadded {
                Text("· 非 3:4 画幅，已居中补边（输出已裁剪）")
                    .font(.caption).foregroundStyle(.orange)
            }
            Spacer()
            Text("MCP: 127.0.0.1:8378").font(.caption2).foregroundStyle(.quaternary)
        }
        .padding(.horizontal, 12).padding(.vertical, 6)
    }

    // MARK: - 编辑 panel

    private var editBar: some View {
        HStack(spacing: 10) {
            Text("编辑").font(.headline).foregroundStyle(.secondary)
            editField("旋转°", $editDeg, width: 60)
            editField("平移X", $editTx, width: 56)
            editField("平移Y", $editTy, width: 56)
            editField("缩放", $editScale, width: 52)
            editField("Z偏移", $editDz, width: 60)
            Button("应用") { applyEdits() }
            Button("撤销") { Task { @MainActor in
                state.status = (try? await state.editUndo()) ?? "撤销失败"
            } }.disabled(!state.editCanUndo)
            Button("重做") { Task { @MainActor in
                state.status = (try? await state.editRedo()) ?? "重做失败"
            } }.disabled(!state.editCanRedo)
            Button("重置") { Task { @MainActor in
                state.status = (try? await state.editReset()) ?? "重置失败"
            } }.disabled(!state.hasEdits)
            Spacer()
            Text("在深度图上横向拖动可旋转").font(.caption2).foregroundStyle(.quaternary)
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .onChange(of: state.editParams) { _, p in
            editDeg = fmt(p.deg); editTx = fmt(p.tx); editTy = fmt(p.ty)
            editScale = fmt(p.scale); editDz = fmt(p.dz)
        }
    }

    private func editField(_ label: String, _ text: Binding<String>, width: CGFloat) -> some View {
        HStack(spacing: 4) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            TextField(label, text: text)
                .frame(width: width)
                .textFieldStyle(.roundedBorder)
                .onSubmit { applyEdits() }
        }
    }

    private func fmt(_ v: Float) -> String {
        String(format: "%g", v)
    }

    private func parsedFields() -> (deg: Float, tx: Float, ty: Float, scale: Float, dz: Float)? {
        guard let deg = Float(editDeg), let tx = Float(editTx), let ty = Float(editTy),
              let scale = Float(editScale), let dz = Float(editDz), scale > 0 else { return nil }
        return (deg, tx, ty, scale, dz)
    }

    private func applyEdits() {
        guard let f = parsedFields() else { state.status = "编辑参数无效"; return }
        Task { @MainActor in
            do {
                state.status = try await state.applyTransform(rotateDeg: f.deg, tx: f.tx, ty: f.ty,
                                                              scale: f.scale, zShift: f.dz)
            } catch {
                state.status = "编辑失败: \(error.localizedDescription)"
            }
        }
    }

    private var rotationDrag: some Gesture {
        DragGesture(minimumDistance: 4)
            .onChanged { v in
                if dragBaseDeg == nil { dragBaseDeg = state.editParams.deg }
                let deg = dragBaseDeg! + Float(v.translation.width) * 0.3
                editDeg = fmt(deg)
                guard !dragApplyInFlight else { return }
                dragApplyInFlight = true
                Task { @MainActor in
                    _ = try? await state.applyTransform(rotateDeg: deg)
                    dragApplyInFlight = false
                }
            }
            .onEnded { _ in
                dragBaseDeg = nil
                applyEdits()
            }
    }

    // MARK: - Actions

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let p = providers.first else { return false }
        if p.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            p.loadItem(forTypeIdentifier: UTType.fileURL.identifier) { item, _ in
                var url: URL?
                if let data = item as? Data { url = URL(dataRepresentation: data, relativeTo: nil) }
                else if let u = item as? URL { url = u }
                if let url { load(url) }
            }
            return true
        }
        if p.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
            p.loadItem(forTypeIdentifier: UTType.image.identifier) { item, _ in
                var tmp: URL?
                if let data = item as? Data {
                    tmp = FileManager.default.temporaryDirectory.appending(path: "da3_drop.png")
                    try? data.write(to: tmp!)
                } else if let u = item as? URL { tmp = u }
                if let tmp { load(tmp) }
            }
            return true
        }
        return false
    }

    private func openPanel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .heic, .tiff, .webP]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url { load(url) }
    }

    private func load(_ url: URL) {
        Task { @MainActor in
            _ = try? await state.loadAndInfer(path: url.path)
        }
    }

    private func exportPanel(_ kind: AppState.ExportKind) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "\(state.inputName)_\(kind.suffix)\(state.flipped ? "_flipped" : "").png"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            state.status = try state.export(kind, to: url)
        } catch {
            state.status = "导出失败: \(error.localizedDescription)"
        }
    }
}
