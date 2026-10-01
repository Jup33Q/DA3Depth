import Foundation
import Network

/// Minimal MCP-style JSON-RPC 2.0 over HTTP server (127.0.0.1:8378) for scripted
/// control / regression (replaces GUI automation in tests).
///
/// POST /mcp  Content-Type: application/json
/// Methods: initialize, tools/list, tools/call
/// Tools:   load_image{path}, infer{}, set_flip{value}, set_mode{mode},
///          export{kind: gray8|color8|gray16, path}, status{}
final class MCPServer: @unchecked Sendable {
    private var listener: NWListener?
    private weak var state: AppState?

    init(state: AppState) { self.state = state }

    func start(port: UInt16 = 8378) {
        do {
            listener = try NWListener(using: .tcp, on: NWEndpoint.Port(rawValue: port)!)
        } catch {
            print("MCP server failed to bind: \(error)")
            return
        }
        listener?.newConnectionHandler = { [weak self] conn in
            conn.start(queue: .global(qos: .userInitiated))
            self?.receive(conn, Data())
        }
        listener?.start(queue: .global(qos: .userInitiated))
        print("MCP server listening on 127.0.0.1:\(port)")
    }

    private func receive(_ conn: NWConnection, _ buf: Data) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, done, _ in
            guard let self else { conn.cancel(); return }
            var acc = buf + (data ?? Data())
            if let response = self.tryHandle(&acc) {
                conn.send(content: response, completion: .contentProcessed { _ in conn.cancel() })
            } else if done {
                conn.cancel()
            } else {
                self.receive(conn, acc)
            }
        }
    }

    /// Returns a complete HTTP response once a full request has arrived, else nil.
    private func tryHandle(_ buf: inout Data) -> Data? {
        guard let headerEnd = buf.range(of: Data("\r\n\r\n".utf8)) else { return nil }
        let header = String(decoding: buf[..<headerEnd.lowerBound], as: UTF8.self)
        var contentLength = 0
        for line in header.split(separator: "\r\n") {
            let parts = line.split(separator: ":", maxSplits: 1)
            if parts.count == 2, parts[0].lowercased() == "content-length" {
                contentLength = Int(parts[1].trimmingCharacters(in: .whitespaces)) ?? 0
            }
        }
        let bodyStart = headerEnd.upperBound
        guard buf.count >= bodyStart + contentLength else { return nil }
        let body = buf[bodyStart..<(bodyStart + contentLength)]

        let rpc = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] ?? [:]
        let id = rpc["id"] ?? NSNull()
        let method = rpc["method"] as? String ?? ""
        let params = rpc["params"] as? [String: Any] ?? [:]

        let result: [String: Any]
        if method == "initialize" {
            result = ["protocolVersion": "2024-11-05",
                      "capabilities": ["tools": [:]],
                      "serverInfo": ["name": "da3depth", "version": "1.0.0"]]
        } else if method == "tools/list" {
            result = ["tools": Self.toolList]
        } else if method == "tools/call" {
            result = Self.toolCall(params: params, state: state)
        } else {
            result = ["note": "ignored method \(method)"]
        }
        let payload: [String: Any] = ["jsonrpc": "2.0", "id": id, "result": result]
        let out = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data()
        var resp = Data("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: \(out.count)\r\nConnection: close\r\n\r\n".utf8)
        resp.append(out)
        return resp
    }

    private nonisolated(unsafe) static let toolList: [[String: Any]] = [
        ["name": "load_image", "description": "Load image from path and run CoreML depth inference",
         "inputSchema": ["type": "object", "properties": ["path": ["type": "string"]], "required": ["path"]]],
        ["name": "infer", "description": "Re-run inference on the current image",
         "inputSchema": ["type": "object", "properties": [:]]],
        ["name": "set_flip", "description": "Horizontal flip on/off",
         "inputSchema": ["type": "object", "properties": ["value": ["type": "boolean"]], "required": ["value"]]],
        ["name": "set_mode", "description": "Display mode gray|color",
         "inputSchema": ["type": "object", "properties": ["mode": ["type": "string"]], "required": ["mode"]]],
        ["name": "export", "description": "Export depth PNG: kind gray8|color8|gray16 to path",
         "inputSchema": ["type": "object", "properties": ["kind": ["type": "string"], "path": ["type": "string"]], "required": ["kind", "path"]]],
        ["name": "status", "description": "Current state (image, depth dims, inference ms)",
         "inputSchema": ["type": "object", "properties": [:]]],
    ]

    private static func toolCall(params: [String: Any], state: AppState?) -> [String: Any] {
        guard let state else { return err("no app state") }
        let name = params["name"] as? String ?? ""
        let args = params["arguments"] as? [String: Any] ?? [:]
        let sem = DispatchSemaphore(value: 0)
        var text = ""
        Task { @MainActor in
            defer { sem.signal() }
            switch name {
            case "load_image":
                guard let path = args["path"] as? String else { text = "missing path"; break }
                do { text = try await state.loadAndInfer(path: path) }
                catch { text = "error: \(error.localizedDescription)" }
            case "infer":
                guard let path = state.lastPath else { text = "no image"; break }
                do { text = try await state.loadAndInfer(path: path) }
                catch { text = "error: \(error.localizedDescription)" }
            case "set_flip":
                state.flipped = args["value"] as? Bool ?? false
                text = "flipped=\(state.flipped)"
            case "set_mode":
                state.mode = (args["mode"] as? String) == "color" ? .color : .gray
                text = "mode=\(state.mode.rawValue)"
            case "export":
                guard let kind = AppState.ExportKind(rawValue: args["kind"] as? String ?? ""),
                      let path = args["path"] as? String else { text = "bad kind/path"; break }
                do { text = try state.export(kind, to: URL(fileURLWithPath: path)) }
                catch { text = "error: \(error.localizedDescription)" }
            case "status":
                let d = state.depth
                text = "status: \(state.status) | depth=\(d.map { "\($0.width)x\($0.height)" } ?? "none") flipped=\(state.flipped) mode=\(state.mode.rawValue)"
            default:
                text = "unknown tool \(name)"
            }
        }
        sem.wait()
        return ["content": [["type": "text", "text": text]]]
    }

    private static func err(_ msg: String) -> [String: Any] {
        ["content": [["type": "text", "text": "error: \(msg)"]], "isError": true]
    }
}
