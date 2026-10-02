// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DepthShaderKit",
    platforms: [.macOS("15.0")],
    products: [
        .library(name: "DepthShaderKit", targets: ["DepthShaderKit"]),
        .executable(name: "depthshader-demo", targets: ["depthshader-demo"]),
        .executable(name: "depthshader-bench", targets: ["depthshader-bench"]),
        .executable(name: "depthshader-parity", targets: ["depthshader-parity"]),
    ],
    targets: [
        .target(name: "DepthShaderKit"),
        .executableTarget(name: "depthshader-demo", dependencies: ["DepthShaderKit"]),
        .executableTarget(name: "depthshader-bench", dependencies: ["DepthShaderKit"]),
        .executableTarget(name: "depthshader-parity", dependencies: ["DepthShaderKit"]),
        .testTarget(name: "DepthShaderKitTests", dependencies: ["DepthShaderKit"]),
    ]
)
