import XCTest
@testable import DepthShaderKit

final class EditStackTests: XCTestCase {
    let cw = 300, ch = 200  // deliberately not tile-aligned; 128px tiles straddle content

    private func makeSource(_ w: Int, _ h: Int, seed: UInt64) throws -> DepthTexture {
        try DepthTexture(values: pattern(width: w, height: h, seed: seed), width: w, height: h)
    }

    private func makeStack(guided: GuidedFilterParams? = nil) throws -> DepthEditStack {
        let base = EditLayer(source: try makeSource(cw, ch, seed: 1))
        let top = EditLayer(source: try makeSource(120, 90, seed: 2),
                            transform: DepthAffineTransform(rotation: 20 * .pi / 180,
                                                            tx: 30, ty: 10, scale: 1.1,
                                                            zShift: -0.5, threshold: 0.05))
        return try DepthEditStack(canvasWidth: cw, canvasHeight: ch, layers: [base, top],
                                  guided: guided)
    }

    /// Reference: run the M2/M3 ops directly, full canvas, same thresholds as the stack.
    private func manualRender(_ stack: DepthEditStack) throws -> [Float] {
        let ops = DepthOps()
        let fuseParams = FuseParams(blendThreshold: stack.fuseBlendThreshold,
                                    seamThreshold: stack.fuseSeamThreshold)
        var transformed: [AffineResult] = []
        for layer in stack.layers {
            transformed.append(try ops.affine(layer.source, transform: layer.transform,
                                              canvasWidth: stack.canvasWidth,
                                              canvasHeight: stack.canvasHeight))
        }
        var acc: FuseResult
        if transformed.count == 1 {
            let zeros = [Float](repeating: 0, count: stack.canvasWidth * stack.canvasHeight)
            let empty = DepthLayer(
                depth: try DepthTexture(values: zeros, width: stack.canvasWidth, height: stack.canvasHeight),
                mask: try DepthTexture(values: zeros, width: stack.canvasWidth, height: stack.canvasHeight))
            acc = try ops.fuse(transformed[0].layer, empty, params: fuseParams)
        } else {
            acc = try ops.fuse(transformed[0].layer, transformed[1].layer, params: fuseParams)
            for i in 2..<transformed.count {
                acc = try ops.fuse(acc.layer, transformed[i].layer, params: fuseParams)
            }
        }
        if let g = stack.guided {
            return try ops.guidedSmooth(acc, params: g).readback()
        }
        return acc.depth.readback()
    }

    func testStackMatchesManualPipeline() throws {
        let stack = try makeStack(guided: GuidedFilterParams(radius: 2, depthThreshold: 1.0))
        let manual = try manualRender(stack)
        XCTAssertEqual(stack.canvas.readback(), manual)
    }

    func testSingleLayerStack() throws {
        let base = EditLayer(source: try makeSource(cw, ch, seed: 3))
        let stack = try DepthEditStack(canvasWidth: cw, canvasHeight: ch, layers: [base])
        let expected = pattern(width: cw, height: ch, seed: 3)
        XCTAssertEqual(stack.canvas.readback(), expected)
    }

    /// Key invariant: dirty-tile incremental re-render is bit-exact vs a full re-render.
    /// Cases straddle tile boundaries and (with guided on) put seam/blend bands at tile edges.
    func testDirtyRenderMatchesFull() throws {
        let guided = GuidedFilterParams(radius: 2, depthThreshold: 1.0)
        let moves: [DepthAffineTransform] = [
            // pure translation landing exactly on a tile boundary
            DepthAffineTransform(tx: 128, ty: 128, scale: 1, zShift: -0.5, threshold: 0.05),
            // rotation sweeping the footprint across tiles
            DepthAffineTransform(rotation: 47 * .pi / 180, tx: -20, ty: 40, scale: 1.3,
                                 zShift: 0.2, threshold: 0.05),
            // move fully off-canvas (empty new footprint, old must be repainted)
            DepthAffineTransform(tx: 5000, ty: 5000, scale: 1),
            // come back
            DepthAffineTransform(rotation: -5 * .pi / 180, tx: 0, ty: 0, scale: 0.8,
                                 threshold: 0.05),
        ]
        let stack = try makeStack(guided: guided)
        for move in moves {
            try stack.updateLayer(1, transform: move)
            let incremental = stack.canvas.readback()
            let full = try stack.render().readback()
            XCTAssertEqual(incremental, full, "dirty render diverged for transform \(move)")
        }
    }

    func testUndoRedoBitExact() throws {
        let stack = try makeStack(guided: GuidedFilterParams(radius: 2, depthThreshold: 1.0))
        let original = stack.canvas.readback()
        let t1 = DepthAffineTransform(rotation: 33 * .pi / 180, tx: -15, ty: 25,
                                      scale: 0.9, threshold: 0.05)
        try stack.updateLayer(1, transform: t1)
        let changed = stack.canvas.readback()
        XCTAssertNotEqual(original, changed)

        XCTAssertTrue(try stack.undo())
        XCTAssertEqual(stack.canvas.readback(), original, "undo must restore the historical render")

        XCTAssertTrue(try stack.redo())
        XCTAssertEqual(stack.canvas.readback(), changed, "redo must restore the changed render")

        XCTAssertTrue(try stack.undo())
        XCTAssertEqual(stack.canvas.readback(), original)
        XCTAssertFalse(try stack.undo(), "history exhausted")
        XCTAssertFalse(try stack.undo())
    }

    func testLayerCRUD() throws {
        let stack = try makeStack()
        let twoLayer = stack.canvas.readback()
        try stack.addLayer(EditLayer(source: try makeSource(60, 60, seed: 4),
                                     transform: DepthAffineTransform(scale: 2, zShift: -2)))
        let threeLayer = stack.canvas.readback()
        XCTAssertNotEqual(twoLayer, threeLayer)
        XCTAssertEqual(stack.layers.count, 3)

        try stack.removeLayer(at: 2)
        XCTAssertEqual(stack.canvas.readback(), twoLayer)

        // undo the remove (snapshot restore with count change -> full re-render)
        XCTAssertTrue(try stack.undo())
        XCTAssertEqual(stack.layers.count, 3)
        XCTAssertEqual(stack.canvas.readback(), threeLayer)
    }

    func testFootprintMath() throws {
        let stack = try makeStack()
        let src = stack.layers[1].source  // 120x90
        // identity: source centered on canvas; pixel extents expand the bbox by 0.5
        let f = stack.footprint(DepthAffineTransform(), source: src)
        XCTAssertEqual(f, TileRect(x: 89, y: 54, w: 121, h: 91))
        // 90° rotation swaps extents
        let r = stack.footprint(DepthAffineTransform(rotation: .pi / 2), source: src)
        XCTAssertEqual(r, TileRect(x: 104, y: 39, w: 91, h: 121))
        // fully off-canvas -> nil
        XCTAssertNil(stack.footprint(DepthAffineTransform(tx: 5000), source: src))
    }
}
