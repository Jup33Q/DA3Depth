# DA3Depth × Shader 深度编辑 路线图

> 基线：`~/dev/DA3Depth`（原版，保持不动）→ 工作副本：`~/dev/DA3Depth-ShaderEdit`（git clone，models/ 与 design/ 已就位，models 为指向原版的符号链接）。
> 目标：给 DA3Depth 加「纯 shader 向」的深度图编辑——旋转 / 位移 / 缩放 + 融合编辑。功能先独立开发成 `DepthShaderKit` Swift Package，验收后再结合进 app。

## 现状摸底（原版 DA3Depth）

- SwiftUI macOS app（macOS 15+，xcodegen 生成工程，bundle `local.da3depth.DA3Depth`）
- `DepthEngine.swift`：CoreML DA3MONO-LARGE 推理（378x504 / 504x378 两个方向模型，ANE/GPU 自动调度）
- `DepthMap.swift`：`[Float]` row-major CPU 数组；翻转、双线性缩放、归一化、灰度/inferno 伪彩、PNG 8/16-bit 导出 —— **全部 CPU 循环，是性能热点**
- `MCPServer.swift`：JSON-RPC over HTTP `127.0.0.1:8378`，工具：load_image / infer / set_flip / set_mode / export / status
- 模型路径硬编码 `~/dev/DA3Depth/models`（工作副本用符号链接复用，勿改）

## 架构决定

1. **纯 shader 向**：所有变换/融合/预览着色走 Metal compute（.metal kernel），CPU 只做编排与调度。
2. **深度纹理格式**：`MTLPixelFormat.r32Float`，与 `DepthMap.values` 一一对应，零转换损耗。
3. **非破坏编辑**：图层只存「源纹理 + 变换参数 + mask」，不烘焙像素；撤销=改参数重渲染。
4. **深度重采样纪律**：深度图禁用朴素双线性（边缘产生前背景混合光晕）。平坦区 bilinear，深度断层处退化 nearest / 取众数。

## 里程碑

### M1 — GPU 基础层（独立包脚手架）
- 新建 `DepthShaderKit/` Swift Package（含 CLI demo 与 bench 可执行目标，不依赖 DA3Depth app）
- `DepthMap` → `MTLTexture(r32Float)` 上传 / 回读，GPU↔CPU roundtrip
- parity 测试：GPU 版翻转/缩放 vs 现有 CPU 实现，误差 = 0（翻转）/ <1e-4（缩放）
- 验收：roundtrip 无损；parity 全绿

### M2 — 仿射变换 kernel（旋转/位移/缩放）
- 单 kernel 支持平面内 rotate / translate / scale，任意输出画布尺寸 + 有效区 mask 输出
- 深度感知重采样（断层检测阈值参数化）
- 深度数值修正随变换融合：scale s → depth/s；z 平移 dz → depth+dz（保持度量一致）
- 验收：4K 深度图单次变换 < 1ms（M 系列 GPU）；90° 旋转与 CPU 参考逐像素一致

### M3 — 融合编辑 kernel
- 双层合成：z-buffer 语义（同像素取近者）+ 图层 mask
- 过渡带：1–2px 深度加权 alpha，消除硬边锯齿
- 接缝窄带引导滤波（可选以 RGB 为引导图），烫平台阶保留真实边缘
- 验收：4K 双层融合 < 1ms；接缝带外与输入逐像素一致

### M4 — 非破坏编辑栈 + 脏区重渲染
- `EditLayer{ source, affine, zShift, mask }` 栈；参数变更只标脏 tile（128×128）重渲染
- undo/redo = 栈操作
- 验收：连续拖动旋转手柄 60fps；undo 后纹理与历史状态逐像素一致

### M5 — benchmark & 回归工具
- 沿用原版 `tools/m*_bench.py` 风格：`tools/m7_shader_parity.py`（CPU vs GPU）、`tools/m8_shader_bench.py`
- 验收：CI 式一键跑通，报告含各 kernel ms / MPix/s

### M6 — 结合进 DA3Depth app
- app 引用本地包 DepthShaderKit；`ContentView` 加「编辑」面板（变换数值输入 + 拖动手柄）
- `AppState` 接入编辑栈；导出路径复用现有 `export()`（编辑结果 → `DepthMap`）
- `MCPServer` 新增工具：`transform_depth{rotate,translate,scale,z_shift}`、`fuse_depth{layers}`、`edit_reset`
- 验收：MCP 脚本完成「推理→变换→融合→导出」全链路；UI 操作与 MCP 结果逐像素一致

### M7 —（二期，可选）2.5D 面外重投影
- 面外旋转/视差位移：深度反投影 → point sprite splatting（走光栅化 z-test，不用原子操作）
- 小洞邻域扩散填补；大洞留到 LDI / inpainting（另立项）
- 仅在 M1–M6 全部验收后启动

## 工程约定

- xcodeproj 由 xcodegen 生成：`xcodegen`（v2.46.0 已装）；勿手改 `.xcodeproj`
- models/ 不进 git；工作副本用符号链接指向原版
- 所有 kernel 需有 CPU 参考实现做 parity（参考实现放测试目录，不进主路径）
- 提交语言：英文 commit message，风格参照原版（`92d8b13`, `7e74f4c`）

## 激活提示词

```
激活 DA3Depth Shader 编辑路线图。工作副本在 ~/dev/DA3Depth-ShaderEdit（原版 ~/dev/DA3Depth 保持不动，models/ 是指向原版的符号链接，勿改 DepthEngine.modelsDir 的硬编码路径）。plan 在 docs/shader-edit-roadmap.md，先读它确认当前里程碑再继续。技术栈已定：纯 Metal compute shader（r32Float 深度纹理），功能先独立开发为 DepthShaderKit Swift Package（M1–M5 验收），再结合进 app 与 MCP(M6)。纪律：深度图禁朴素双线性重采样（断层处退 nearest）；编辑非破坏（只存变换参数）；每个 kernel 要有 CPU parity 参考。xcodeproj 用 xcodegen 重新生成，勿手改。MCP 端口 8378。
```
