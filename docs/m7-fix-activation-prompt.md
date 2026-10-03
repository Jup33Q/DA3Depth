# M7-fix 激活提示词 — 重投影质量修复(splat 雪花 + 去遮挡填补)

```
激活 DA3Depth Shader 编辑路线图的 M7-fix 子任务：「reproject_view 质量修复:splat 雪花消除 + 去遮挡填补升级」。

【背景】
- 工作副本 ~/dev/DA3Depth-ShaderEdit(原版 ~/dev/DA3Depth 不动;models/ 是指向原版的符号链接)。plan 在 docs/shader-edit-roadmap.md。M7 已实现并提交(commit 868b43b: point-sprite splat kernel + 硬件 z-test + ≤5° 微步递推 + 洞填补 + MCP reproject_view + warped 导出),机制验证通过:yaw ±5° 能产出正确的立体视差对(主体反向平移、遮挡关系随视角变化、去遮挡区出现在正确一侧)。
- 但输出质量不可用,两个缺陷(实测数据见下)。本任务只做质量修复,不改 API 形态。
- MCP 现状:reproject_view{yaw_deg 必填, pitch_deg, fill_holes=default true, soften_edges=default true};export kind 新增 warped_color / warped_depth16 / warped_mask(导出最近一次 reproject 结果)。

【实测数据(2026-10-03,输入 first-frame-1152x1536.png + DA3MONO-LARGE 378x504 深度)】
- yaw +5°:覆盖率 88.8%;yaw -5°:89.1%;yaw +2.5°:95.4%
- 缺陷1 splat 雪花:主体与裙摆内部遍布单像素级黑洞,与角度无关(2.5° 依旧)。高度怀疑根因:深度点云源是 378x504(推理原生分辨率),重投影目标画布 1152x1536,源点间距被放大 ~3.05x,1px point sprite 盖不满 → 稀疏雪花。
- 缺陷2 去遮挡填补不力:5° 时人物背后大片黑白雪花带,fill_holes 只补微小洞;大洞按 mask=0 设计保留,但彩图里直接输出纯黑,无法作为下游参考图使用。
- 缺陷3 变形深度输出不达标:warped_depth16 虽有导出且几何正确(近暗远白、渐变保留),但与彩图一样满是雪花洞,去遮挡区纯 0 未做任何深度域填补——0 值会被下游读成「最近距离」,且归一化表现与原始 gray16 导出不一致(主体整体偏黑),无法直接当新视角的配对深度图用。样图 vrot-first-yaw+5-depth16.png。
- 样图在 ~/Desktop/videos/01/_reset/:vrot-first-yaw+5.png / vrot-first-yaw-5.png / vrot-first-yaw+2.5.png / vrot-first-yaw+2.5-mask.png / vrot-first-yaw+5-depth16.png,先打开看再动手。

【任务目标】
1. 消灭 splat 雪花(三选一或组合,以验收为准):
   a. splat 足迹按「源分辨率→目标画布放大比」自适应放大(当前场景 ≥3px);
   b. 源点先上采样到画布分辨率再 splat(双线性上采样深度时遵守断层退 nearest 纪律);
   c. 改 quad/mesh splatting(相邻四点构三角面片,深度断层处断开)。
2. 去遮挡区填补升级:pull-push 金字塔填补(推荐,-hole 大小不敏感)或大幅增大扩散半径;彩图任何情况下不得直出纯黑未填补区——填不满的区域用最近有效颜色延展,mask 保持 0 标出。
3. 保持既有纪律:硬件 z-test 不用原子操作;RGB 与深度共享同一遮挡顺序;微步递推不变。
4. 变形深度输出补齐(本任务新增重点):warped_depth16 必须达到与彩图同等级质量——
   a. 去遮挡区在深度域做 pull-push 填补,遵守断层纪律(前景深度不得扩散进背景);
   b. 未填补区不得输出 0(0 会被下游读成最近距离),用最近有效深度延展填充,mask 标 0;
   c. gray16 归一化区间与原始深度导出一致(同一 min/max 基准,或在导出 metadata/返回文本里注明);
   d. 彩图 / 深度 / mask 三者有效区严格一致,保证「新视角彩图 + 新视角深度」可直接配对进下游创意板。

【验收标准】
- yaw ±5° / 1152x1536:splat 覆盖区(非去遮挡 mask=0 区)零雪花,目视无单像素黑洞;整图无纯黑区域;去遮挡带填补过渡自然,无明显补丁感。
- warped_depth16:无 0 值黑洞,深度断层不被填补抹平,归一化与原始 gray16 导出一致;与变形彩图逐像素同视角,可直接当新视角几何参考。
- 首帧验证素材(~/Desktop/videos/01/_reset/first-frame-1152x1536.png)±5° 输出(彩图+深度)可作 LKG 视差配对目视检查;尾帧 last-frame-1152x1536.png 同跑一遍确认无回归。
- 微步、角度钳制(±30°)、parity 测试、bench 全部保持绿色;单次 5° 重投影耗时不能显著劣于现版(报告中给 ms 对比)。

【执行要求】
- xcodeproj 用 xcodegen 重新生成,勿手改;构建必须 DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild(本机 xcode-select 指向 CommandLineTools)。
- 深度图禁朴素双线性重采样(断层处退 nearest);编辑非破坏;kernel 改動同步更新 CPU parity 参考;英文 commit message(风格参照 868b43b)。
- 调试 MCP 前先 kill 占用 8378 的旧进程再启动新构建;ShaderEdit 版 DerivedData 目录 DA3Depth-eefjaqseudvrepgcgugpeiwuspbf,别启错旧版。
- 完成后更新 docs/shader-edit-roadmap.md M7 验收记录(附修复前后覆盖率对比)。
```
