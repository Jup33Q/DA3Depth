import SwiftUI

@main
struct DA3DepthApp: App {
    @StateObject private var state = AppState()
    private let mcp: MCPServer

    init() {
        let s = AppState()
        _state = StateObject(wrappedValue: s)
        mcp = MCPServer(state: s)
        // DA3DEPTH_MCP_PORT overrides the default port (used by tools/m9_mcp_chain.py
        // to coexist with a user-running instance).
        let port = UInt16(ProcessInfo.processInfo.environment["DA3DEPTH_MCP_PORT"] ?? "") ?? 8378
        mcp.start(port: port)
    }

    var body: some Scene {
        WindowGroup("DA3Depth — Depth Anything 3 (DA3MONO-LARGE, CoreML)") {
            ContentView(state: state)
                .frame(minWidth: 980, minHeight: 640)
        }
        .windowResizability(.contentMinSize)
    }
}
