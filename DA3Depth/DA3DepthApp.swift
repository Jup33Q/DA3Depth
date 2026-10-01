import SwiftUI

@main
struct DA3DepthApp: App {
    @StateObject private var state = AppState()
    private let mcp: MCPServer

    init() {
        let s = AppState()
        _state = StateObject(wrappedValue: s)
        mcp = MCPServer(state: s)
        mcp.start()
    }

    var body: some Scene {
        WindowGroup("DA3Depth — Depth Anything 3 (DA3MONO-LARGE, CoreML)") {
            ContentView(state: state)
                .frame(minWidth: 980, minHeight: 640)
        }
        .windowResizability(.contentMinSize)
    }
}
