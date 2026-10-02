import Metal
import Foundation

/// A non-destructive edit layer: source pixels + transform parameters only.
public struct EditLayer {
    public var source: DepthTexture
    public var transform: DepthAffineTransform
    public init(source: DepthTexture, transform: DepthAffineTransform = DepthAffineTransform()) {
        self.source = source
        self.transform = transform
    }
}

struct TileRect: Equatable {
    var x: Int, y: Int, w: Int, h: Int
}

/// Non-destructive edit stack: composites layers bottom→top (M2 affine + M3 fuse) onto a
/// cached canvas. Parameter changes re-render only the dirty 128x128 tiles covering the
/// union of the layer's old and new forward footprints. Undo/redo are parameter snapshots.
public final class DepthEditStack {
    public static let tileSize = 128

    public let canvasWidth: Int
    public let canvasHeight: Int
    public private(set) var layers: [EditLayer]
    /// Optional seam smoothing applied after the fuse chain (M3 guided pass).
    public var guided: GuidedFilterParams?

    private let ops: DepthOps
    private let context: GPUContext

    private var layerDepth: [MTLTexture] = []
    private var layerMask: [MTLTexture] = []
    private var composite: FuseResult!
    private var pingA: FuseResult?
    private var pingB: FuseResult?
    private var emptyLayer: DepthLayer!
    private var guidedOut: MTLTexture?

    private var undoStack: [[EditLayer]] = []
    private var redoStack: [[EditLayer]] = []

    public init(canvasWidth: Int, canvasHeight: Int, layers: [EditLayer],
                guided: GuidedFilterParams? = nil, ops: DepthOps = DepthOps()) throws {
        precondition(!layers.isEmpty, "edit stack needs at least a base layer")
        self.canvasWidth = canvasWidth
        self.canvasHeight = canvasHeight
        self.layers = layers
        self.guided = guided
        self.ops = ops
        self.context = ops.context
        func tex() throws -> MTLTexture { try ops.makeOutput(width: canvasWidth, height: canvasHeight) }
        for _ in layers {
            layerDepth.append(try tex())
            layerMask.append(try tex())
        }
        composite = FuseResult(depth: DepthTexture(texture: try tex()),
                               mask: DepthTexture(texture: try tex()),
                               winner: DepthTexture(texture: try tex()),
                               seam: DepthTexture(texture: try tex()))
        if layers.count > 2 {
            pingA = FuseResult(depth: DepthTexture(texture: try tex()),
                               mask: DepthTexture(texture: try tex()),
                               winner: DepthTexture(texture: try tex()),
                               seam: DepthTexture(texture: try tex()))
            pingB = FuseResult(depth: DepthTexture(texture: try tex()),
                               mask: DepthTexture(texture: try tex()),
                               winner: DepthTexture(texture: try tex()),
                               seam: DepthTexture(texture: try tex()))
        }
        let zeros = [Float](repeating: 0, count: canvasWidth * canvasHeight)
        emptyLayer = DepthLayer(depth: try DepthTexture(values: zeros, width: canvasWidth, height: canvasHeight),
                                mask: try DepthTexture(values: zeros, width: canvasWidth, height: canvasHeight))
        if guided != nil { guidedOut = try tex() }
        try renderFull()
    }

    /// Current composited canvas (post-guided if enabled).
    public var canvas: DepthTexture {
        guidedOut.map { DepthTexture(texture: $0) } ?? composite.depth
    }

    public var canUndo: Bool { !undoStack.isEmpty }
    public var canRedo: Bool { !redoStack.isEmpty }

    /// Full re-render of the whole canvas.
    public func render() throws -> DepthTexture {
        try renderFull()
        return canvas
    }

    /// Updates a layer's transform and re-renders only the dirty tiles.
    @discardableResult
    public func updateLayer(_ index: Int, transform: DepthAffineTransform) throws -> DepthTexture {
        undoStack.append(layers)
        redoStack.removeAll()
        let dirty = footprint(layers[index].transform, source: layers[index].source)
            .union(footprint(transform, source: layers[index].source))
        layers[index].transform = transform
        if let dirty {
            try renderDirty(dirty, changedLayers: [index])
        }
        return canvas
    }

    public func addLayer(_ layer: EditLayer) throws {
        undoStack.append(layers)
        redoStack.removeAll()
        layers.append(layer)
        layerDepth.append(try ops.makeOutput(width: canvasWidth, height: canvasHeight))
        layerMask.append(try ops.makeOutput(width: canvasWidth, height: canvasHeight))
        if layers.count > 2 && pingA == nil {
            func tex() throws -> MTLTexture { try ops.makeOutput(width: canvasWidth, height: canvasHeight) }
            pingA = FuseResult(depth: DepthTexture(texture: try tex()),
                               mask: DepthTexture(texture: try tex()),
                               winner: DepthTexture(texture: try tex()),
                               seam: DepthTexture(texture: try tex()))
            pingB = FuseResult(depth: DepthTexture(texture: try tex()),
                               mask: DepthTexture(texture: try tex()),
                               winner: DepthTexture(texture: try tex()),
                               seam: DepthTexture(texture: try tex()))
        }
        try renderFull()
    }

    public func removeLayer(at index: Int) throws {
        precondition(layers.count > 1, "cannot remove the last layer")
        undoStack.append(layers)
        redoStack.removeAll()
        layers.remove(at: index)
        layerDepth.remove(at: index)
        layerMask.remove(at: index)
        try renderFull()
    }

    @discardableResult
    public func undo() throws -> Bool {
        guard let previous = undoStack.popLast() else { return false }
        redoStack.append(layers)
        try restore(previous)
        return true
    }

    @discardableResult
    public func redo() throws -> Bool {
        guard let next = redoStack.popLast() else { return false }
        undoStack.append(layers)
        try restore(next)
        return true
    }

    private func restore(_ snapshot: [EditLayer]) throws {
        if snapshot.count != layers.count {
            layers = snapshot
            while layerDepth.count < layers.count {
                layerDepth.append(try ops.makeOutput(width: canvasWidth, height: canvasHeight))
                layerMask.append(try ops.makeOutput(width: canvasWidth, height: canvasHeight))
            }
            if layers.count > 2 && pingA == nil {
                func tex() throws -> MTLTexture { try ops.makeOutput(width: canvasWidth, height: canvasHeight) }
                pingA = FuseResult(depth: DepthTexture(texture: try tex()),
                                   mask: DepthTexture(texture: try tex()),
                                   winner: DepthTexture(texture: try tex()),
                                   seam: DepthTexture(texture: try tex()))
                pingB = FuseResult(depth: DepthTexture(texture: try tex()),
                                   mask: DepthTexture(texture: try tex()),
                                   winner: DepthTexture(texture: try tex()),
                                   seam: DepthTexture(texture: try tex()))
            }
            try renderFull()
            return
        }
        var dirty: TileRect?
        var changed = Set<Int>()
        for i in layers.indices {
            let a = footprint(layers[i].transform, source: layers[i].source)
            let b = footprint(snapshot[i].transform, source: snapshot[i].source)
            if a != b || layers[i].transform != snapshot[i].transform {
                dirty = dirty.union(a).union(b)
                changed.insert(i)
            }
        }
        layers = snapshot
        if let dirty {
            try renderDirty(dirty, changedLayers: changed)
        }
    }

    // MARK: - dirty region math

    /// Forward footprint: the 4 source corners mapped by dst = s*R(θ)*(src - cSrc) + cCanvas + t,
    /// expanded by half a pixel and clamped to the canvas. nil if fully off-canvas.
    func footprint(_ t: DepthAffineTransform, source: DepthTexture) -> TileRect? {
        let csx = Double(source.width - 1) / 2, csy = Double(source.height - 1) / 2
        let cdx = Double(canvasWidth - 1) / 2, cdy = Double(canvasHeight - 1) / 2
        let c = cos(Double(t.rotation)), s = sin(Double(t.rotation))
        let k = Double(t.scale)
        var loX = Double.infinity, loY = Double.infinity
        var hiX = -Double.infinity, hiY = -Double.infinity
        for (px, py) in [(0.0, 0.0), (Double(source.width - 1), 0.0),
                         (0.0, Double(source.height - 1)),
                         (Double(source.width - 1), Double(source.height - 1))] {
            let dx = k * (c * (px - csx) - s * (py - csy)) + cdx + Double(t.tx)
            let dy = k * (s * (px - csx) + c * (py - csy)) + cdy + Double(t.ty)
            loX = min(loX, dx); loY = min(loY, dy)
            hiX = max(hiX, dx); hiY = max(hiY, dy)
        }
        let x0 = max(0, Int(floor(loX - 0.5)))
        let y0 = max(0, Int(floor(loY - 0.5)))
        let x1 = min(canvasWidth, Int(ceil(hiX + 0.5)))
        let y1 = min(canvasHeight, Int(ceil(hiY + 0.5)))
        guard x0 < x1, y0 < y1 else { return nil }
        return TileRect(x: x0, y: y0, w: x1 - x0, h: y1 - y0)
    }

    /// Snaps a rect to the tile grid, padded by `pad` pixels on each side, clamped to canvas.
    private func tileRect(_ rect: TileRect, pad: Int) -> TileRect {
        let t = DepthEditStack.tileSize
        let x0 = max(0, rect.x - pad) / t * t
        let y0 = max(0, rect.y - pad) / t * t
        let x1 = min(canvasWidth, ((rect.x + rect.w + pad + t - 1) / t) * t)
        let y1 = min(canvasHeight, ((rect.y + rect.h + pad + t - 1) / t) * t)
        return TileRect(x: x0, y: y0, w: x1 - x0, h: y1 - y0)
    }

    // MARK: - rendering

    private func renderFull() throws {
        let full = TileRect(x: 0, y: 0, w: canvasWidth, h: canvasHeight)
        try renderRegion(fuseRect: full, guidedRect: full, changedLayers: Set(layers.indices))
    }

    /// Padding rationale: fuse/affine are pixel-local, so their inputs only change inside the
    /// dirty rect. The guided pass at pixel p reads fused depth/seam within radius r, so its
    /// output can change inside dirty⊕r; recomputing that needs fused data over dirty⊕2r.
    private func renderDirty(_ dirty: TileRect, changedLayers: Set<Int>) throws {
        let r = guided?.radius ?? 0
        try renderRegion(fuseRect: tileRect(dirty, pad: 2 * r),
                         guidedRect: tileRect(dirty, pad: r),
                         changedLayers: changedLayers)
    }

    private func renderRegion(fuseRect: TileRect, guidedRect: TileRect,
                              changedLayers: Set<Int>) throws {
        var items: [GPUContext.EncodeOp] = []
        let affinePipeline = try context.pipeline(function: "affine_transform")
        for i in changedLayers {
            var params = layers[i].transform.params(sourceWidth: layers[i].source.width,
                                                    sourceHeight: layers[i].source.height,
                                                    canvasWidth: canvasWidth,
                                                    canvasHeight: canvasHeight)
            params.ox = UInt32(fuseRect.x)
            params.oy = UInt32(fuseRect.y)
            let src = layers[i].source.texture
            let dstD = layerDepth[i], dstM = layerMask[i]
            items.append(GPUContext.EncodeOp(pipeline: affinePipeline,
                                             width: fuseRect.w, height: fuseRect.h) { encoder in
                var p = params
                encoder.setTexture(src, index: 0)
                encoder.setTexture(dstD, index: 1)
                encoder.setTexture(dstM, index: 2)
                encoder.setBytes(&p, length: MemoryLayout<AffineParams>.stride, index: 0)
            })
        }

        let fusePipeline = try context.pipeline(function: "fuse_layers")
        func layerAt(_ i: Int) -> DepthLayer {
            DepthLayer(depth: DepthTexture(texture: layerDepth[i]),
                       mask: DepthTexture(texture: layerMask[i]))
        }
        func fuseOp(_ a: DepthLayer, _ b: DepthLayer, into out: FuseResult) -> GPUContext.EncodeOp {
            var params = FuseKernelParams(blendThreshold: fuseBlendThreshold,
                                          seamThreshold: fuseSeamThreshold,
                                          ox: UInt32(fuseRect.x), oy: UInt32(fuseRect.y))
            return GPUContext.EncodeOp(pipeline: fusePipeline,
                                       width: fuseRect.w, height: fuseRect.h) { encoder in
                encoder.setTexture(a.depth.texture, index: 0)
                encoder.setTexture(a.mask.texture, index: 1)
                encoder.setTexture(b.depth.texture, index: 2)
                encoder.setTexture(b.mask.texture, index: 3)
                encoder.setTexture(out.depth.texture, index: 4)
                encoder.setTexture(out.mask.texture, index: 5)
                encoder.setTexture(out.winner.texture, index: 6)
                encoder.setTexture(out.seam.texture, index: 7)
                encoder.setBytes(&params, length: MemoryLayout<FuseKernelParams>.stride, index: 0)
            }
        }
        let n = layers.count
        if n == 1 {
            items.append(fuseOp(layerAt(0), emptyLayer, into: composite))
        } else {
            var acc: FuseResult?
            var usePingA = true
            for i in 1..<n {
                let isLast = i == n - 1
                let out = isLast ? composite! : (usePingA ? pingA! : pingB!)
                usePingA.toggle()
                if let acc {
                    items.append(fuseOp(acc.layer, layerAt(i), into: out))
                } else {
                    items.append(fuseOp(layerAt(0), layerAt(1), into: out))
                }
                acc = out
            }
        }

        if let guided, let guidedOut {
            let guidedPipeline = try context.pipeline(function: "guided_smooth")
            var params = GuidedKernelParams(epsilon: guided.epsilon,
                                            depthThreshold: guided.depthThreshold,
                                            radius: Int32(guided.radius), useDepthGuide: 1,
                                            ox: UInt32(guidedRect.x), oy: UInt32(guidedRect.y))
            let comp = composite!
            items.append(GPUContext.EncodeOp(pipeline: guidedPipeline,
                                             width: guidedRect.w, height: guidedRect.h) { encoder in
                encoder.setTexture(comp.depth.texture, index: 0)
                encoder.setTexture(comp.mask.texture, index: 1)
                encoder.setTexture(comp.seam.texture, index: 2)
                encoder.setTexture(comp.depth.texture, index: 3)
                encoder.setTexture(guidedOut, index: 4)
                encoder.setBytes(&params, length: MemoryLayout<GuidedKernelParams>.stride, index: 0)
            })
        }
        try context.encodeBatch(items)
    }

    /// Fuse parameters used when compositing the stack; exposed for M6 wiring.
    public var fuseBlendThreshold: Float = 0.1
    public var fuseSeamThreshold: Float = 0.1
}

private extension TileRect {
    func union(_ other: TileRect?) -> TileRect {
        guard let other else { return self }
        let x0 = min(x, other.x), y0 = min(y, other.y)
        let x1 = max(x + w, other.x + other.w), y1 = max(y + h, other.y + other.h)
        return TileRect(x: x0, y: y0, w: x1 - x0, h: y1 - y0)
    }
}

private extension Optional where Wrapped == TileRect {
    func union(_ other: TileRect?) -> TileRect? {
        switch (self, other) {
        case let (a?, b?): return a.union(b)
        case let (a?, nil): return a
        case let (nil, b?): return b
        case (nil, nil): return nil
        }
    }
}
