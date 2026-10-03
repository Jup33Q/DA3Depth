# M7 激活提示词 — 2.5D 面外重投影(竖轴旋转)

```
激活 DA3Depth Shader 编辑路线图的 M7 子任务：「2.5D 面外重投影(绕竖轴 yaw 旋转)」。

【背景】
- 工作副本 ~/dev/DA3Depth-ShaderEdit(原版 ~/dev/DA3Depth 保持不动;models/ 是指向原版的符号链接,勿改 DepthEngine.modelsDir 硬编码路径)。plan 在 docs/shader-edit-roadmap.md,先读它。
- M1–M6 已全部验收(commit 430eec1=M6 结合进 app、8201135=M1–M5 DepthShaderKit、854c609=roadmap)。DepthShaderKit 已有 Metal compute 的 affine/fuse/guided kernel、非破坏 EditStack、CPU parity 参考、bench 工具(tools/m7_shader_parity.py、m8_shader_bench.py)。
- 现状缺陷:transform_depth 只有画面内 2D 旋转/位移/缩放,没有绕竖轴(Yaw)的面外旋转——做视差素材需要的是视点绕竖轴转动,这是本任务要补的能力。
- 验证素材现成:~/Desktop/videos/01/_reset/ 下有 first/last-frame-1152x1536.png 及对应的 16bit 原始深度 depth-raw-first.png / depth-raw-last.png(DA3MONO-LARGE 推理,近暗远白)。

【任务目标】
按 roadmap M7 实现深度反投影重投影:
1. DepthShaderKit 新增重投影 kernel:深度 r32Float 反投影为点云 → 绕竖轴 yaw(°)旋转(可顺带支持小角度 pitch)→ point sprite splatting 重投影回画布。走光栅化 z-test,不用原子操作(roadmap 既定纪律)。
2. 同一重投影同时作用于源 RGB 彩图(共享 z-buffer/深度顺序),输出变形后彩图 + 变形后深度 + 有效区 mask——使用方需要彩图深度同变换配对。
3. 小洞(旋转视差产生的遮挡空洞)做邻域扩散填补;大洞不处理,mask 标出即可(LDI/inpainting 另立项)。
4. MCPServer(端口 8378)新增工具 reproject_view{yaw_deg, pitch_deg?, fill_holes?},export 增加可导出变形彩图的 kind。
5. CPU parity 参考实现放测试目录,重投影坐标数学与 GPU 版逐像素对齐(允许 splat 边界 ±1px)。

【验收标准】
- MCP 脚本一键完成:load_image → infer → reproject_view{yaw_deg:5} → 导出变形彩图 PNG + gray16 深度 + mask,输入 1152×1536。
- 5° yaw 下:主体轮廓无拉丝,遮挡顺序正确(近处盖远处),小洞填补目视无明显补丁;mask 准确标出未填补大洞。
- 4K 深度单次重投影耗时与现有 kernel 同量级(M 系列 GPU 毫秒级);parity 测试全绿。
- 首尾帧两张验证素材各跑 ±5°,输出可供 LKG 视差配对目视检查。

【执行要求】
- xcodeproj 用 xcodegen 重新生成,勿手改 .xcodeproj;构建必须 DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild(本机 xcode-select 指向 CommandLineTools,直接调 xcodebuild 会报错)。
- 深度图禁朴素双线性重采样(断层处退 nearest);编辑保持非破坏;每个 kernel 配 CPU parity 参考;英文 commit message,风格参照 92d8b13/7e74f4c。
- 调试 MCP 前先 kill 占用 8378 的旧 app 进程再启动新构建(ShaderEdit 版 DerivedData 目录为 DA3Depth-eefjaqseudvrepgcgugpeiwuspbf,与旧版 fsnvkbyuvsmnepaduglxqbznrrlb 不同,别启错)。
- 完成后更新 docs/shader-edit-roadmap.md 的 M7 勾选与验收记录。
```
