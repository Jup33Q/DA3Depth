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

### M7 — 2.5D 面外重投影 ✅（2026-10-03 验收；M7-fix 同日复验）
- [x] 面外旋转/视差位移：深度反投影 → point sprite splatting（渲染管线 depth32Float 附件走硬件 z-test，无原子操作）。坐标约定：X 横轴（右）、Y 竖轴（上）、Z 朝屏幕外（相机沿 -Z 看，深度 d → Z=-d）；绕 pivot 深度面 yaw（竖轴）+ 可选 pitch（横轴）；正 yaw 内容左移
- [x] 同一 splat pass 输出变形彩图（rgba16Float）+ 变形深度 + 有效区 mask，共享 z-test 配对；源深度最近邻上采样到画布分辨率（禁双线性），彩图全分辨率采样
- [x] 小洞邻域扩散填补（8 邻域均值，`fillRadius` 轮）；大洞不处理，mask=0 标出（LDI/inpainting 另立项）
- [x] 限位 + 微动 + 递归：单步 ≤5°（`maxStepDeg`），单轴总量 clamp ±30°（`maxTotalDeg`）；超限时拆成等角微步递归链式重投影（pivot 固定为首帧均值深度）；`edge_soften` kernel 在洞缘/深度断层处做 3×3 高斯半强度糊化弥补 splat 锐边
- [x] MCP `reproject_view{yaw_deg, pitch_deg?, fill_holes?, soften_edges?}`；export 新增 `warped_color|warped_depth16|warped_mask`（depth16 按 mask 有效区归一化）
- [x] CPU parity：`Tests/.../ReprojectTests.swift` + `depthshader-parity` 双份参考，逐像素对齐（splat 边界 ±1px 容差）；`tools/m10_reproject_chain.py` 一键验收
- 验收记录：swift test 29 项全绿；m7_shader_parity 18/18 PASS（yaw5 depth max diff 4.8e-7）；m8 bench 4K 单次重投影 GPU 5.2ms（wall 25ms，与其他 kernel 同属毫秒级）；m10 全 PASS——首尾帧各 ±5° 导出 1152×1536 变形彩图/gray16 深度/mask 至 `~/Desktop/videos/01/_reset/reproject-m10/`，覆盖率 84.7–89.1%（入画边缘带为大洞正确留空），重复链路字节级一致；yaw12→3 微步、yaw45→clamp 30°；m9 链路回归 PASS

#### M7-fix — reproject_view 质量修复 ✅（2026-10-03 复验，splats 雪花 + 去遮挡填补 + warped_depth16）

修复前缺陷（868b43b）：1px splat 盖不住深度梯度摊开的间距 → 主体内部雪花黑洞（与角度无关）；fill 只补小洞，大洞彩图直出纯黑、深度直出 0；warped_depth16 按 mask 有效区归一化，与 gray16 基准不一致。

修复内容（全部保持纪律：无原子操作、RGB/深度同权重共享遮挡序、编辑非破坏、CPU parity 双份同步）：
- splat 改两 pass 高斯圆盘（visibility splatting）：pass A 硬件 z-test 记每像素最近深度；pass B 同一组 splat 以高斯圆盘权重做 additive 累加（API 顺序混合，逐顶点序可复现），权重 × 对 pass A 深度的软门控（σ=depthBreak/3，>depthBreak 硬切——前景不会洇进背景）；pass C 归一化出 color/depth/mask。splat 足迹自适应：`ps = clamp(ceil(hypot(distR, distD)), 1, 8)`（右/下邻居投影间距，断层处不增长——那是真去遮挡，归填补管）。
- 去遮挡填补升级：扩散小洞（mask 升 1）→ pull-push 金字塔（pull：颜色加权均值、深度取最远 max；push：断层门控 bilinear，只取最远层 tap 重归一化）→ 填补带用深度梯度（一阶 normal 信息）做 5×5 门控平面拟合外推（Cramer 解 + 邻域范围钳制，退化回门控均值；颜色保持门控 3×3 高斯）。彩图/深度任何情况下无纯黑/零值，mask=0 标出填补区。
- 微步收紧到 ≤1°：链式每步只重掷 mask=1 真实内容（上一步 mask 传入 vertex 做剔除），小去遮挡逐步被扩散吸收。实验记录：1° 干净且 mask 语义保持（±5° 覆盖率 ~96-97%）；0.2°（25 步）出横向拖影梯田且覆盖率 100%（出画区被错误标为有效），弃用。中间步跳过 pull-push/平面填补（mask=0 不参与下一步 splat，结果逐位一致），省 ~20% wall。
- warped_depth16：归一化基准改为源深度 min/max（与 gray16 导出同源），reproject_view 返回文本注明基准；全图无 0 值洞。
- 中间发现：accumColor 不能用 rgba16Float——混合按附件精度累加，alpha 权重半精度漂移会偏置归一化深度（+1.4e-3），保持 rgba32Float。

验收记录：swift test 29 全绿（含新增 pull-push/无黑洞用例）；m7 parity 20/20（新增 carved-hole 填补 depth/mask parity，diff ≤2.4e-7）；m10 全 PASS（yaw5→5 微步、yaw12→12、yaw45→clamp30/30 步，重复链路字节级一致）；m9 回归 PASS。首帧 1152×1536 覆盖率 +5° 97.0% / −5° 96.3% / +2.5° 98.4%（修复前 88.8/89.1/95.4）；尾帧 96.5/95.5（修复前 84.7–88 区间）；四组导出数值门：彩图纯黑 0 像素、mask=1 区黑洞 0、depth16 零值 0。性能：4K 5°（5 微步）wall 82ms / 单微步 GPU ~14ms（修复前单步 wall 25ms / GPU 5.2ms——双 pass 高斯 splat 单步更贵，微步加密 ×5；质量导向，可接受）；MCP 端 1152×1536 ±5° 约 550–600ms（Debug 构建，含 CPU 回读/转换）。样图：`~/Desktop/videos/01/_reset/reproject-m7fix/`（另留 step1/step0.2/planar 实验对比目录）。

## 工程约定

- xcodeproj 由 xcodegen 生成：`xcodegen`（v2.46.0 已装）；勿手改 `.xcodeproj`
- models/ 不进 git；工作副本用符号链接指向原版
- 所有 kernel 需有 CPU 参考实现做 parity（参考实现放测试目录，不进主路径）
- 提交语言：英文 commit message，风格参照原版（`92d8b13`, `7e74f4c`）

## 激活提示词

```
激活 DA3Depth Shader 编辑路线图。工作副本在 ~/dev/DA3Depth-ShaderEdit（原版 ~/dev/DA3Depth 保持不动，models/ 是指向原版的符号链接，勿改 DepthEngine.modelsDir 的硬编码路径）。plan 在 docs/shader-edit-roadmap.md，先读它确认当前里程碑再继续。技术栈已定：纯 Metal compute shader（r32Float 深度纹理），功能先独立开发为 DepthShaderKit Swift Package（M1–M5 验收），再结合进 app 与 MCP(M6)。纪律：深度图禁朴素双线性重采样（断层处退 nearest）；编辑非破坏（只存变换参数）；每个 kernel 要有 CPU parity 参考。xcodeproj 用 xcodegen 重新生成，勿手改。MCP 端口 8378。
```
