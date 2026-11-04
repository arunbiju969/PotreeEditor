import * as THREE from '/libs/three.js/build/three.module.js';

// Global state
let currentPointCloud = null;
let activeTool = 'nav'; // 'nav', 'polygon', 'box', 'inspect'
let activeSelectionMesh = null;
let currentSelectionStats = null;

// Layer registry.
//
// Potree has no separate layer manager: viewer.scene.pointclouds IS the list (Scene.js
// addPointCloud pushes, sidebar.js walks it for one row per cloud, and deleting a cloud
// is a splice plus a remove from the three.js graph).  Per-layer state lives on the
// cloud itself -- pc.visible for the eye, pc.material.activeAttributeName and
// pointSize for appearance -- so a layer is a cloud plus its own material, not a
// parallel data structure.  Names are the key, the way LoadProject.js reuses clouds by
// name, which means removing a layer can never shift another layer's identity.
//
// "Active" is the one thing potree never had: its picking code reaches for
// scene.pointclouds[0] (viewer.js), while our edits, picks and saves go through the
// layer the user selected.
let activeLayerName = null;

function uniqueLayerName(base) {
    const taken = new Set(viewer.scene.pointclouds.map(pc => pc.name));
    const stem = base || ('cloud_' + (viewer.scene.pointclouds.length + 1));
    let name = stem;
    for (let i = 2; taken.has(name); i++) name = stem + '_' + i;
    return name;
}

// Name a layer after the thing the user actually opened.  Potree's own name comes from
// the cloud's metadata, and the converter writes that generically, so the URL wins:
// for a converted cloud it is the folder (the entry file is dropped), for COPC/LAS it is
// the file name with its extension removed.  This is the key the layer rows carry, so it
// stays put even when another layer is removed.
function layerNameFromUrl(url) {
    // Strip the query before decoding, then treat everything else as path: these URLs
    // never carry a fragment, and the folder names do contain '#' (clipped_Area#1_...),
    // so splitting on '#' before decoding would truncate the real name.
    const decoded = decodeURIComponent(String(url || '').split('?')[0]);
    const parts = decoded.split('/').filter(Boolean);
    let base = parts.pop() || '';
    if (/^(metadata\.json|cloud\.js)$/i.test(base)) base = parts.pop() || '';
    return base.replace(/\.copc\.laz$/i, '')
               .replace(/\.(laz|las|json|js|txt)$/i, '');
}

function layerByName(name) {
    return viewer.scene.pointclouds.find(pc => pc.name === name) || null;
}

function getAvailableAttributes(pc) {
    if (!pc || !pc.getAttributes) return ['elevation', 'rgba', 'classification', 'intensity'];
    const attrs = pc.getAttributes();
    if (!attrs || !attrs.attributes) return ['elevation', 'rgba', 'classification', 'intensity'];
    const rawAttrs = attrs.attributes.map(a => a.name);
    const result = [];

    // Standard spatial & visual channels
    result.push('elevation');
    if (rawAttrs.some(a => ['rgba', 'rgb', 'color'].includes(a))) {
        result.push('rgba');
    }

    // Include every scalar dimension present in the point cloud
    for (const a of rawAttrs) {
        if (['position', 'POSITION_CARTESIAN', 'rgba', 'rgb', 'color'].includes(a)) continue;
        result.push(a);
        if (a === 'intensity') {
            result.push('intensity gradient');
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Viridis 12-step color palette matching the forestry single-tree view
// ---------------------------------------------------------------------------
const VIRIDIS16 = [
    [68, 1, 84], [72, 36, 117], [65, 68, 135], [53, 95, 141],
    [42, 120, 142], [33, 145, 140], [34, 168, 132], [68, 190, 112],
    [122, 209, 81], [165, 220, 54], [210, 226, 27], [253, 231, 37]
];

const FORESTRY_VIRIDIS_GRADIENT = VIRIDIS16.map((rgb, idx) => [
    idx / (VIRIDIS16.length - 1),
    new THREE.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
]);

// Categorical Tree-ID ramp. Uses golden-angle hues over a 330° window [45°-375°]
// strictly bypassing the brown/mud band [15°-45°].
// Stop 0 is rich earthy brown reserved for ground.
const FORESTRY_TREE_STOPS = 64;
const FORESTRY_GROUND_COLOR = new THREE.Color(0.48, 0.27, 0.14);
const FORESTRY_TREE_GRADIENT = (() => {
    const stops = [];
    stops.push([0.0, FORESTRY_GROUND_COLOR]);
    for (let i = 1; i < FORESTRY_TREE_STOPS; i++) {
        const t = ((i * 0.61803398875) % 1.0 + 1.0) % 1.0;
        const hue = ((45.0 + t * 330.0) % 360.0) / 360.0;
        stops.push([i / FORESTRY_TREE_STOPS, new THREE.Color().setHSL(hue, 0.88, 0.54)]);
    }
    // Wrap the last band back to the first tree hue, closing the cycle
    stops.push([1.0, stops[1][1].clone()]);
    return stops;
})();

function treeAttrGradient(attrName) {
    const n = String(attrName || '').toLowerCase().replace(/[\s_-]/g, '');
    return ['itc', 'treeid', 'tree_id', 'tree'].includes(n)
        ? FORESTRY_TREE_GRADIENT
        : FORESTRY_VIRIDIS_GRADIENT;
}

if (typeof Potree !== 'undefined') {
    if (Potree.Gradients) {
        Potree.Gradients.VIRIDIS = FORESTRY_VIRIDIS_GRADIENT;
    }
    if (Potree.ClassificationScheme && Potree.ClassificationScheme.DEFAULT && Potree.ClassificationScheme.DEFAULT[2]) {
        Potree.ClassificationScheme.DEFAULT[2].color = [0.48, 0.27, 0.14, 1.0];
    }
}

function applyColorAttribute(pc, attr) {
    if (!pc || !pc.material) return;

    // Tree/ITC gets the vivid cyclic ramp; other scalar fields use forestry Viridis
    pc.material.gradient = treeAttrGradient(attr);
    pc.material.activeAttributeName = attr;

    if (pc.getAttribute && pc.material.setRange) {
        const attribute = pc.getAttribute(attr);
        if (attribute) {
            const currentRange = pc.material.getRange ? pc.material.getRange(attr) : null;
            const range = attribute.range || attribute.initialRange;
            if (Array.isArray(range) && range.length >= 2 && Number.isFinite(range[0]) && Number.isFinite(range[1])) {
                let [minVal, maxVal] = range;
                if (minVal === maxVal) maxVal = minVal + 1;
                if (!currentRange || (currentRange[0] === 0 && currentRange[1] === 0) || attr === 'semantic_pred') {
                    pc.material.setRange(attr, [minVal, maxVal]);
                }
            }
        }
    }
    if (pc.material.updateShaderSource) {
        pc.material.updateShaderSource();
    }
}

function layerInfo(pc) {
    if (!pc) return null;
    return {
        name: pc.name,
        visible: pc.visible !== false,
        active: pc.name === activeLayerName,
        colorAttribute: pc.material ? pc.material.activeAttributeName : null,
        // The per-layer style object is the source of truth for what the widgets wrote,
        // so size/shape/sizing round-trip through the layer list.
        pointSize: layerStyle(pc).pointSize,
        pointType: layerStyle(pc).pointType,
        pointSizeType: layerStyle(pc).pointSizeType,
        points: (pc.potree && pc.potree.numPoints) || pc.numPoints || 0,
        attributes: getAvailableAttributes(pc)
    };
}

function listLayers() {
    return viewer.scene.pointclouds.map(layerInfo);
}

function removeLayer(name) {
    const pc = layerByName(name);
    if (!pc) return listLayers();
    viewer.scene.scenePointCloud.remove(pc);
    const idx = viewer.scene.pointclouds.indexOf(pc);
    if (idx >= 0) viewer.scene.pointclouds.splice(idx, 1);
    if (pc.geometry && typeof pc.geometry.dispose === 'function') {
        try { pc.geometry.dispose(); } catch (e) {}
    }
    if (currentPointCloud === pc) currentPointCloud = null;
    if (activeLayerName === name) {
        const next = viewer.scene.pointclouds[0];
        currentPointCloud = next || null;
        activeLayerName = next ? next.name : null;
    }
    viewer.render();
    return listLayers();
}

function setActiveLayer(name) {
    const pc = layerByName(name);
    if (!pc) return listLayers();
    currentPointCloud = pc;
    activeLayerName = pc.name;
    if (pc.material) viewerSettings.colorMode = pc.material.activeAttributeName;
    viewer.render();
    return listLayers();
}

function setLayerVisible(name, visible) {
    const pc = layerByName(name);
    if (!pc) return listLayers();
    pc.visible = !!visible;
    viewer.render();
    return listLayers();
}

function setLayerColorAttribute(name, attr) {
    const pc = layerByName(name);
    if (pc) applyColorAttribute(pc, attr);
    if (pc && pc === currentPointCloud) {
        viewerSettings.colorMode = attr;
        updateTreeColors();
    }
    viewer.render();
    return listLayers();
}

function setLayerPointSize(name, size) {
    const pc = layerByName(name);
    if (pc) {
        layerStyle(pc).pointSize = Number(size);
        applyMaterialSettings(pc);
    }
    viewer.render();
    return listLayers();
}

// Optimistic Visual Staging Storage (edit.id -> THREE.Group)
const stagedVisuals = new Map();

// Viewer Settings State
const viewerSettings = {
    pointType: 'SQUARE',         // 'SQUARE', 'CIRCLE', 'PARABOLOID'
    pointSizeType: 'ADAPTIVE',   // 'ADAPTIVE', 'FIXED', 'ATTENUATED'
    pointSize: 1.0,
    splatQuality: 'standard',   // 'standard', 'hq'
    pointBudget: 2_500_000,
    minNodeSize: 10,
    edlEnabled: true,
    edlRadius: 1.4,
    edlStrength: 1.0,
    edlOpacity: 1.0,
    background: 'black',
    colorMode: 'rgba'
};

// Initialize Potree Viewer
const renderArea = document.getElementById('potree_render_area');
const viewer = new Potree.Viewer(renderArea);
window.viewer = viewer;

// Dedicated 3D Editor Overlay Scene (rendered on top of point clouds and EDL passes)
const overlayScene = new THREE.Scene();
overlayScene.name = 'potree_editor_overlay_scene';

// Potree's octree camera carries a frustum tailored to the octree's bounding
// box (aspect/plane skew). Rendering the overlay with a pose-synced camera of
// the canvas aspect makes world positions project symmetrically — the tree
// view lands dead-centre with plain lookAt math.
const overlayCamP = new THREE.PerspectiveCamera(60, 1, 0.05, 2e6);
const overlayCamO = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.05, 2e6);

// Enforce close near clipping plane (5cm) on main camera so points never clip when close up or at steep angles
viewer.addEventListener('update', () => {
    const cam = viewer.scene.getActiveCamera();
    if (cam && (cam.near > 0.05 || cam.near <= 0 || isNaN(cam.near))) {
        cam.near = 0.05;
        cam.updateProjectionMatrix();
    }
});

viewer.addEventListener('render.pass.perspective_overlay', () => {
    const camera = viewer.scene.getActiveCamera();
    if (camera && overlayScene.children.length > 0) {
        const canvas = viewer.renderer.domElement;
        let cam2;
        if (camera.isOrthographicCamera) {
            overlayCamO.left = camera.left; overlayCamO.right = camera.right;
            overlayCamO.top = camera.top;   overlayCamO.bottom = camera.bottom;
            overlayCamO.zoom = camera.zoom; overlayCamO.near = 0.05;
            overlayCamO.far = camera.far;
            cam2 = overlayCamO;
        } else {
            overlayCamP.fov = camera.fov;
            overlayCamP.aspect = canvas.clientHeight > 0 ? canvas.clientWidth / canvas.clientHeight : 1;
            overlayCamP.near = 0.05;
            overlayCamP.far = Math.max(camera.far, 10000);
            overlayCamP.zoom = camera.zoom;
            cam2 = overlayCamP;
        }
        cam2.position.copy(camera.position);
        cam2.quaternion.copy(camera.quaternion);
        cam2.updateProjectionMatrix();
        cam2.updateMatrixWorld(true);
        viewer.renderer.render(overlayScene, cam2);
    }
});

viewer.setEDLEnabled(viewerSettings.edlEnabled);
viewer.setEDLRadius(viewerSettings.edlRadius);
viewer.setEDLStrength(viewerSettings.edlStrength);
viewer.setEDLOpacity(viewerSettings.edlOpacity);
viewer.setFOV(60);
viewer.setPointBudget(viewerSettings.pointBudget);
viewer.setMinNodeSize(viewerSettings.minNodeSize);
viewer.setBackground(viewerSettings.background);

// UI Elements
const toastMsg = document.getElementById('toastMsg');
const selectionBanner = document.getElementById('selectionBanner');
const selectionSummary = document.getElementById('selectionSummary');
const btnQuickNoise = document.getElementById('btnQuickNoise');
const btnQuickAssignTree = document.getElementById('btnQuickAssignTree');
const quickTreeIdInput = document.getElementById('quickTreeIdInput');
const inspectCard = document.getElementById('inspectCard');
const closeInspect = document.getElementById('closeInspect');

let toastTimer = null;
function showToast(text, duration = 3000) {
    if (!toastMsg) return;
    if (toastTimer) clearTimeout(toastTimer);
    toastMsg.textContent = text;
    toastMsg.style.display = 'block';
    toastTimer = setTimeout(() => {
        toastMsg.style.display = 'none';
    }, duration);
}
window.showToast = showToast;

let isIndexingActive = false;
let indexingHideTimer = null;

window.updateIndexingStatus = function(data) {
    if (!data) return;
    const badge = document.getElementById('indexingBadge');
    const icon = document.getElementById('indexingBadgeIcon');
    const text = document.getElementById('indexingBadgeText');
    const bar = document.getElementById('indexingBadgeBar');
    const prog = document.getElementById('indexingBadgeProgress');
    if (!badge || !text) return;

    if (indexingHideTimer) {
        clearTimeout(indexingHideTimer);
        indexingHideTimer = null;
    }

    const type = data.type;
    if (type === 'started' || type === 'progress') {
        isIndexingActive = true;
        badge.style.display = 'flex';
        badge.style.borderColor = '#0284c7';
        badge.style.color = '#38bdf8';
        if (icon) icon.textContent = data.stage === 'trees' ? '🌲' : '⚡';
        text.textContent = data.msg || 'Automatic Background Indexing...';
        if (bar && prog && data.percent !== undefined) {
            bar.style.display = 'block';
            prog.style.width = Math.min(100, Math.max(0, data.percent)) + '%';
        }
    } else if (type === 'completed' || type === 'ready') {
        isIndexingActive = false;
        badge.style.display = 'flex';
        badge.style.borderColor = '#059669';
        badge.style.color = '#34d399';
        if (icon) icon.textContent = '✓';
        text.textContent = data.msg || 'Spatial index ready';
        if (bar) bar.style.display = 'none';
        indexingHideTimer = setTimeout(() => {
            badge.style.display = 'none';
        }, 8000);
    } else if (type === 'error') {
        isIndexingActive = false;
        badge.style.display = 'flex';
        badge.style.borderColor = '#dc2626';
        badge.style.color = '#f87171';
        if (icon) icon.textContent = '⚠';
        text.textContent = 'Streaming Fallback Active';
        if (bar) bar.style.display = 'none';
    }
};

// Single POST helper: awaits, checks res.ok, throws with the endpoint name.
async function postJSON(url, body) {
    const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    if (!res.ok) {
        throw new Error(`${url} -> HTTP ${res.status}`);
    }
    return res.json();
}

// ---------------------------------------------------------------------------
// Tool Activation API (Invoked from native PyQt6 Ribbon Bar)
// ---------------------------------------------------------------------------
window.setActiveTool = function(tool) {
    activeTool = tool;
    if (tool === 'nav') {
        renderArea.style.cursor = 'default';
        showToast('Navigation Mode: Orbit / Pan');
    } else if (tool === 'polygon') {
        renderArea.style.cursor = 'crosshair';
        startPolygonSelection();
    } else if (tool === 'box') {
        renderArea.style.cursor = 'crosshair';
        startBoxSelection();
    } else if (tool === 'inspect') {
        renderArea.style.cursor = 'crosshair';
        showToast('Pick Point Mode: Ctrl+Click any point in the cloud to inspect and highlight');
    }
};

// ---------------------------------------------------------------------------
// Helpers for Potree Material & Quality Settings
// ---------------------------------------------------------------------------
function getPointShapeEnum(name) {
    if (typeof name === 'number') return name;
    const str = String(name || '').toUpperCase();
    if (str === 'CIRCLE') return Potree.PointShape.CIRCLE;
    if (str === 'PARABOLOID') return Potree.PointShape.PARABOLOID;
    return Potree.PointShape.SQUARE;
}

function getPointSizeTypeEnum(name) {
    if (typeof name === 'number') return name;
    const str = String(name || '').toUpperCase();
    if (str === 'FIXED') return Potree.PointSizeType.FIXED;
    if (str === 'ATTENUATED') return Potree.PointSizeType.ATTENUATED;
    return Potree.PointSizeType.ADAPTIVE;
}

// Per-layer render style.  Size, shape and sizing mode are per-material properties in
// potree, so each cloud carries its own; viewerSettings holds the style a newly opened
// layer starts from.  Names (SQUARE / ADAPTIVE / ...) are stored rather than enum
// values, so the Qt widgets read back exactly what they wrote -- potree's enums have no
// reverse mapping.  potree 2.0 also keeps the size in material.size (the uniforms.size
// value); it has accessors for pointSizeType but no material.pointSize, so writing
// material.pointSize would silently do nothing.
function layerStyle(pc) {
    if (!pc) return viewerSettings;
    if (!pc.userData) pc.userData = {};
    if (!pc.userData.potreeEditorStyle) {
        pc.userData.potreeEditorStyle = {
            pointType: viewerSettings.pointType,
            pointSizeType: viewerSettings.pointSizeType,
            pointSize: viewerSettings.pointSize
        };
    }
    return pc.userData.potreeEditorStyle;
}

function applyMaterialSettings(pc = currentPointCloud) {
    if (!pc || !pc.material) return;
    const mat = pc.material;
    const style = layerStyle(pc);
    mat.shape = getPointShapeEnum(style.pointType);
    mat.pointSizeType = getPointSizeTypeEnum(style.pointSizeType);
    mat.size = Number(style.pointSize);
    if (typeof FORESTRY_VIRIDIS_GRADIENT !== 'undefined' && FORESTRY_VIRIDIS_GRADIENT) {
        mat.gradient = treeAttrGradient(mat.activeAttributeName);
    }
}

function updateAllPointClouds() {
    if (viewer.scene && viewer.scene.pointclouds) {
        viewer.scene.pointclouds.forEach(applyMaterialSettings);
    }
}

// ---------------------------------------------------------------------------
// Point Cloud Loading
// ---------------------------------------------------------------------------
window.loadPointCloudUrl = async function(url, displayName) {
    console.log('[PotreeHybrid] Loading point cloud:', url);
    showToast('Loading point cloud: ' + url.split('/').pop());

    try {
        const e = await Potree.loadPointCloud(url);
        const pc = e.pointcloud;

        // Register as another layer rather than replacing whatever is open.  Potree's
        // scene list is already the layer manager, so clouds coexist; the new one
        // becomes active and keeps its own material, visibility and point size.
        // Every cloud is mounted under a hashed alias (cloud_25662), so the URL alone
        // carries nothing readable; Python passes the folder or file the user picked.
        const wanted = String(displayName || '').replace(/^.*[\\/]/, '')
                           .replace(/\.copc\.laz$/i, '').replace(/\.(laz|las|json|js|txt)$/i, '');
        pc.name = uniqueLayerName(wanted || layerNameFromUrl(url) || pc.name);
        currentPointCloud = pc;
        activeLayerName = pc.name;

        // Apply all configured material settings
        applyMaterialSettings(pc);

        const hasRGBA = pc.getAttributes().attributes.some(a => a.name === 'rgba');
        if (hasRGBA && viewerSettings.colorMode === 'rgba') {
            applyColorAttribute(pc, 'rgba');
        } else if (viewerSettings.colorMode) {
            applyColorAttribute(pc, viewerSettings.colorMode);
        } else {
            applyColorAttribute(pc, 'color');
        }

        viewer.scene.addPointCloud(pc);
        viewer.zoomTo(pc);

        showToast('Point cloud loaded successfully!');
        return true;
    } catch (err) {
        console.error('[PotreeHybrid] Load error:', err);
        showToast('Error loading point cloud: ' + err.message, 5000);
        return false;
    }
};

window.reloadCurrentPointCloud = async function() {
    if (!currentPointCloud) return;
    const name = currentPointCloud.name;
    const url = currentPointCloud.url || (currentPointCloud.pcoGeometry && currentPointCloud.pcoGeometry.url);
    if (!url) return;
    console.log('[PotreeHybrid] Reloading active point cloud layer:', name);
    removeLayer(name);
    const cleanUrl = url.split('?')[0];
    await window.loadPointCloudUrl(cleanUrl, name);
};

// ---------------------------------------------------------------------------
// 3D Billboard Sprite Label Generator
// ---------------------------------------------------------------------------
function makeTextSprite(message, colorStr) {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    // Background pill container
    ctx.fillStyle = 'rgba(15, 23, 42, 0.90)';
    if (ctx.roundRect) {
        ctx.roundRect(4, 4, 248, 56, 10);
    } else {
        ctx.rect(4, 4, 248, 56);
    }
    ctx.fill();

    // Colored border
    ctx.strokeStyle = colorStr || '#10b981';
    ctx.lineWidth = 4;
    if (ctx.roundRect) {
        ctx.roundRect(4, 4, 248, 56, 10);
    } else {
        ctx.rect(4, 4, 248, 56);
    }
    ctx.stroke();

    // High-contrast clean typography
    ctx.font = 'bold 22px system-ui, -apple-system, sans-serif';
    ctx.fillStyle = '#f8fafc';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(message, 128, 32);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMaterial = new THREE.SpriteMaterial({ map: texture, depthTest: false });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.scale.set(7.0, 1.75, 1.0);
    sprite.renderOrder = 1002;
    return sprite;
}

// ---------------------------------------------------------------------------
// Optimistic Visual Staging Management API
// ---------------------------------------------------------------------------
window.addStagedVisualPolygon = (edit) => {
    if (!edit || !edit.id) return;
    if (stagedVisuals.has(edit.id)) {
        window.removeStagedVisualPolygon(edit.id);
    }

    const group = new THREE.Group();
    const hexColor = parseInt((edit.color || '#10b981').replace('#', '0x'), 16);

    let pts = [];
    if (edit.points_3d && edit.points_3d.length >= 3) {
        pts = edit.points_3d.map(p => new THREE.Vector3(p[0], p[1], p[2]));
    } else if (edit.coords && edit.coords.length >= 3) {
        const sceneZ = viewer.scene.view.position.z || 0;
        pts = edit.coords.map(c => new THREE.Vector3(c[0], c[1], sceneZ));
    } else {
        return;
    }

    // Closed boundary loop
    const geom = new THREE.BufferGeometry().setFromPoints(pts);
    const lineMat = new THREE.LineBasicMaterial({
        color: hexColor,
        linewidth: 3,
        depthTest: false
    });
    const line = new THREE.LineLoop(geom, lineMat);
    line.renderOrder = 1000;
    group.add(line);

    // Glowing corner vertex nodes
    const ptMat = new THREE.PointsMaterial({
        color: hexColor,
        size: 7,
        sizeAttenuation: false,
        depthTest: false
    });
    const vertexPoints = new THREE.Points(geom, ptMat);
    vertexPoints.renderOrder = 1001;
    group.add(vertexPoints);

    // Centroid and floating 3D label
    let sumX = 0, sumY = 0, sumZ = 0, maxZ = -Infinity;
    for (const p of pts) {
        sumX += p.x;
        sumY += p.y;
        sumZ += p.z;
        if (p.z > maxZ) maxZ = p.z;
    }
    const cx = sumX / pts.length;
    const cy = sumY / pts.length;
    const cz = (maxZ > -100000) ? maxZ + 1.5 : (sumZ / pts.length) + 1.5;

    const sprite = makeTextSprite(edit.label || 'Staged Edit', edit.color || '#10b981');
    sprite.position.set(cx, cy, cz);
    group.add(sprite);

    overlayScene.add(group);
    stagedVisuals.set(edit.id, group);
};

window.removeStagedVisualPolygon = (editId) => {
    if (!stagedVisuals.has(editId)) return;
    const group = stagedVisuals.get(editId);
    overlayScene.remove(group);
    group.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
            if (child.material.map) child.material.map.dispose();
            child.material.dispose();
        }
    });
    stagedVisuals.delete(editId);
};

window.clearAllStagedVisuals = () => {
    for (const [id, group] of stagedVisuals.entries()) {
        overlayScene.remove(group);
        group.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (child.material.map) child.material.map.dispose();
                child.material.dispose();
            }
        });
    }
    stagedVisuals.clear();
};

window.loadStagedVisuals = (editsList) => {
    window.clearAllStagedVisuals();
    if (Array.isArray(editsList)) {
        for (const edit of editsList) {
            window.addStagedVisualPolygon(edit);
        }
    }
};

// ---------------------------------------------------------------------------
// Polygon Selection Tool
// ---------------------------------------------------------------------------
let polygonPoints = [];
let polygonLine = null;

function startPolygonSelection() {
    clearSelection(false);
    polygonPoints = [];
    showToast('Polygon: Left-click points to trace boundary. Double-click or Right-click to finish.');

    const lineGeo = new THREE.BufferGeometry();
    const lineMat = new THREE.LineBasicMaterial({ color: 0x3b82f6, linewidth: 2, depthTest: false });
    polygonLine = new THREE.Line(lineGeo, lineMat);
    polygonLine.renderOrder = 999;
    overlayScene.add(polygonLine);
}

function finishPolygonSelection() {
    if (polygonPoints.length < 3) {
        showToast('Polygon cancelled: requires at least 3 points.');
        clearSelection();
        activeTool = 'nav';
        return;
    }

    const flatPoints = [];
    for (const pt of polygonPoints) {
        flatPoints.push(pt.x, pt.y, pt.z);
    }
    flatPoints.push(polygonPoints[0].x, polygonPoints[0].y, polygonPoints[0].z);

    const positions = new Float32Array(flatPoints);
    polygonLine.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    polygonLine.geometry.attributes.position.needsUpdate = true;
    activeSelectionMesh = polygonLine;

    const coords2d = polygonPoints.map(p => [p.x, p.y]);
    const coords3d = polygonPoints.map(p => [p.x, p.y, p.z]);

    postJSON('/api/select_polygon', { coordinates: coords2d, points_3d: coords3d })
    .then(data => {
        handleSelectionResult(data);
    })
    .catch(err => {
        console.error('Polygon query failed:', err);
        showToast('Query error: ' + err.message);
    });

    activeTool = 'nav';
}

// ---------------------------------------------------------------------------
// Box Selection Tool
// ---------------------------------------------------------------------------
function startBoxSelection() {
    clearSelection(false);
    showToast('Box: Click and drag to create a 3D selection box.');

    if (viewer.volumeTool) {
        const volume = viewer.volumeTool.startInsertion({ type: Potree.BoxVolume });
        // The shipped 1.8.x renderer fires 'drop' when the volume placement finishes.
        volume.addEventListener('drop', () => {
            const min = volume.boundingBox.min;
            const max = volume.boundingBox.max;
            activeSelectionMesh = volume;

            postJSON('/api/select_box', {
                min: [min.x, min.y, min.z],
                max: [max.x, max.y, max.z]
            })
            .then(data => handleSelectionResult(data))
            .catch(err => {
                console.error('Box query failed:', err);
                showToast('Query error: ' + err.message);
            });

            activeTool = 'nav';
        });
    }
}

// ---------------------------------------------------------------------------
// Accurate Point Cloud Raycasting & Picking
// ---------------------------------------------------------------------------
function getIntersectionAt(screenX, screenY, allowPlaneFallback = true) {
    const camera = viewer.scene.getActiveCamera();
    const pointclouds = viewer.scene.pointclouds;
    if (!camera) return null;

    // 1. Precise Potree GPU Point Cloud Intersection
    if (pointclouds && pointclouds.length > 0 && Potree.Utils && Potree.Utils.getMousePointCloudIntersection) {
        const I = Potree.Utils.getMousePointCloudIntersection(
            { x: screenX, y: screenY },
            camera,
            viewer,
            pointclouds,
            { pickClipped: true, pickWindowSize: 17 }
        );
        if (I && I.location) {
            return {
                position: I.location,
                point: I.point,
                pointcloud: I.pointcloud,
                distance: I.distance
            };
        }
    }

    if (!allowPlaneFallback) {
        return null;
    }

    // 2. Camera ray fallback to horizontal ground plane (for polygon tracing)
    const rect = renderArea.getBoundingClientRect();
    const ndc = new THREE.Vector2(
        (screenX / rect.width) * 2 - 1,
        -(screenY / rect.height) * 2 + 1
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(ndc, camera);

    let targetZ = 0;
    if (viewer.scene.view && viewer.scene.view.getPivot) {
        targetZ = viewer.scene.view.getPivot().z || 0;
    }
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -targetZ);
    const hitPos = new THREE.Vector3();
    const hit = raycaster.ray.intersectPlane(plane, hitPos);
    if (hit) {
        return {
            position: hitPos,
            point: null,
            pointcloud: null,
            distance: camera.position.distanceTo(hitPos)
        };
    }
    return null;
}

// ---------------------------------------------------------------------------
// Picked Point Visual Highlighting System
// ---------------------------------------------------------------------------
let pickedPointMarker = null;
let pickedPointReticleGroup = null;
let pickedPointHalo = null;
let pickedPointPole = null;
let pickedPointLabel = null;

const ASPRS_CLASSES = {
    0: 'Created / Unclassified',
    1: 'Unassigned',
    2: 'Ground',
    3: 'Low Vegetation',
    4: 'Medium Vegetation',
    5: 'High Vegetation',
    6: 'Building',
    7: 'Low Point / Noise',
    8: 'Model Key-point',
    9: 'Water',
    10: 'Rail',
    11: 'Road Surface',
    12: 'Overlap / Reserved',
    13: 'Wire - Guard',
    14: 'Wire - Conductor',
    15: 'Transmission Tower',
    16: 'Wire-Structure Connector',
    17: 'Bridge Deck',
    18: 'High Noise'
};

function formatClassName(classCode) {
    if (classCode === null || classCode === undefined || classCode === '-') return '-';
    const num = Number(classCode);
    if (!isNaN(num) && ASPRS_CLASSES[num]) {
        return `${num} (${ASPRS_CLASSES[num]})`;
    }
    return String(classCode);
}

function extractAttr(pt, ...keys) {
    if (!pt) return null;
    for (const key of keys) {
        if (pt[key] !== undefined && pt[key] !== null) {
            const v = pt[key];
            if (Array.isArray(v) || (v && v.buffer && v.length !== undefined)) {
                return v.length > 0 ? v[0] : null;
            }
            return v;
        }
    }
    return null;
}

function clearPickedPointHighlight() {
    const btnInspectViewTree = document.getElementById('btnInspectViewTree');
    if (btnInspectViewTree) btnInspectViewTree.style.display = 'none';
    if (pickedPointMarker) {
        overlayScene.remove(pickedPointMarker);
        pickedPointMarker.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (child.material.map) child.material.map.dispose();
                child.material.dispose();
            }
        });
        pickedPointMarker = null;
        pickedPointReticleGroup = null;
        pickedPointHalo = null;
        pickedPointPole = null;
        pickedPointLabel = null;
    }
}

function highlightPickedPoint(pos, labelText = null, colorHex = 0x00f5ff, colorStr = '#00f5ff') {
    clearPickedPointHighlight();
    if (!pos) return;

    pickedPointMarker = new THREE.Group();
    pickedPointMarker.position.set(pos.x, pos.y, pos.z);

    // 1. Central Core Sphere (exact point coordinate)
    const sphereGeom = new THREE.SphereGeometry(0.35, 16, 16);
    const sphereMat = new THREE.MeshBasicMaterial({
        color: colorHex,
        depthTest: false,
        depthWrite: false
    });
    const sphere = new THREE.Mesh(sphereGeom, sphereMat);
    sphere.renderOrder = 10001;
    pickedPointMarker.add(sphere);

    // 2. Camera-Facing Reticle Sub-Group (orients to camera every frame so it never flattens)
    pickedPointReticleGroup = new THREE.Group();

    // Concentric Target Ring
    const innerRingGeom = new THREE.RingGeometry(0.7, 0.95, 32);
    const innerRingMat = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        side: THREE.DoubleSide,
        depthTest: false,
        depthWrite: false
    });
    const innerRing = new THREE.Mesh(innerRingGeom, innerRingMat);
    innerRing.renderOrder = 10002;
    pickedPointReticleGroup.add(innerRing);

    // Pulsing Outer Beacon Halo (animated in render loop)
    const haloGeom = new THREE.RingGeometry(1.1, 1.55, 32);
    const haloMat = new THREE.MeshBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: 0.85,
        side: THREE.DoubleSide,
        depthTest: false,
        depthWrite: false
    });
    pickedPointHalo = new THREE.Mesh(haloGeom, haloMat);
    pickedPointHalo.renderOrder = 10000;
    pickedPointReticleGroup.add(pickedPointHalo);

    // Precision Crosshairs (4 perpendicular ticks framing the center)
    const crossPoints = [
        new THREE.Vector3(-1.6, 0, 0), new THREE.Vector3(-0.95, 0, 0),
        new THREE.Vector3(0.95, 0, 0), new THREE.Vector3(1.6, 0, 0),
        new THREE.Vector3(0, -1.6, 0), new THREE.Vector3(0, -0.95, 0),
        new THREE.Vector3(0, 0.95, 0), new THREE.Vector3(0, 1.6, 0)
    ];
    const crossGeom = new THREE.BufferGeometry().setFromPoints(crossPoints);
    const crossMat = new THREE.LineBasicMaterial({
        color: colorHex,
        linewidth: 2,
        depthTest: false,
        depthWrite: false
    });
    const cross = new THREE.LineSegments(crossGeom, crossMat);
    cross.renderOrder = 10002;
    pickedPointReticleGroup.add(cross);

    pickedPointMarker.add(pickedPointReticleGroup);

    // 3. Dynamic Leader Line connecting reticle to label
    const poleGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 3.2, 0)
    ]);
    const poleMat = new THREE.LineBasicMaterial({
        color: 0x38bdf8,
        linewidth: 2,
        depthTest: false,
        depthWrite: false
    });
    pickedPointPole = new THREE.Line(poleGeom, poleMat);
    pickedPointPole.renderOrder = 10002;
    pickedPointMarker.add(pickedPointPole);

    // 4. Floating 3D Badge Sprite
    const text = labelText || `X:${pos.x.toFixed(2)} Y:${pos.y.toFixed(2)} Z:${pos.z.toFixed(2)}`;
    pickedPointLabel = makeTextSprite(text, colorStr);
    pickedPointLabel.position.set(0, 3.8, 0);
    pickedPointMarker.add(pickedPointLabel);

    // Render directly in overlayScene on top of point clouds
    overlayScene.add(pickedPointMarker);
}

function updatePickedPointLabel(text, colorStr = '#00f5ff') {
    if (!pickedPointMarker || !pickedPointLabel) return;
    const oldPos = pickedPointLabel.position.clone();
    pickedPointMarker.remove(pickedPointLabel);
    if (pickedPointLabel.geometry) pickedPointLabel.geometry.dispose();
    if (pickedPointLabel.material) {
        if (pickedPointLabel.material.map) pickedPointLabel.material.map.dispose();
        pickedPointLabel.material.dispose();
    }
    pickedPointLabel = makeTextSprite(text, colorStr);
    pickedPointLabel.position.copy(oldPos);
    pickedPointMarker.add(pickedPointLabel);
}

// Keep highlight marker scale sharp, orient reticle towards camera, and pulse the halo ring
viewer.addEventListener('update', () => {
    if (pickedPointMarker) {
        const camera = viewer.scene.getActiveCamera();
        if (camera) {
            const dist = camera.position.distanceTo(pickedPointMarker.position);
            let s = 1.0;
            if (Potree.Utils && Potree.Utils.projectedRadius && viewer.renderer) {
                const pr = Potree.Utils.projectedRadius(1, camera, dist, viewer.renderer.domElement.clientWidth, viewer.renderer.domElement.clientHeight);
                s = Math.max(0.01, 15 / Math.max(0.0001, pr));
            } else {
                s = Math.max(0.01, dist * 0.02);
            }
            pickedPointMarker.scale.set(s, s, s);

            // Orient the target reticle directly towards the camera so it is always open and circular
            if (pickedPointReticleGroup) {
                pickedPointReticleGroup.quaternion.copy(camera.quaternion);
            }

            // Calculate the camera's visual "up" vector in local marker space
            const camUp = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
            const badgeOffset = camUp.clone().multiplyScalar(3.8);

            // Position callout badge visually above the target reticle from any viewing angle
            if (pickedPointLabel) {
                pickedPointLabel.position.copy(badgeOffset);
            }

            // Update leader line to connect reticle to callout badge
            if (pickedPointPole) {
                const poleTarget = badgeOffset.clone().multiplyScalar(0.85);
                const posAttr = pickedPointPole.geometry.attributes.position;
                posAttr.setXYZ(1, poleTarget.x, poleTarget.y, poleTarget.z);
                posAttr.needsUpdate = true;
            }

            // Pulse the beacon halo
            if (pickedPointHalo) {
                const t = (Date.now() % 1200) / 1200;
                const pulse = 1.0 + t * 0.8;
                pickedPointHalo.scale.set(pulse, pulse, pulse);
                if (pickedPointHalo.material) {
                    pickedPointHalo.material.opacity = Math.max(0, 0.85 * (1.0 - t));
                }
            }
        }
    }
});

// ---------------------------------------------------------------------------
// Inspection / Pick Point Tool
// ---------------------------------------------------------------------------
// Raycast the Forestry tab's isolated tree overlay (THREE.Points) and show the
// same picked-point highlight + Point Attributes card as the main cloud picker.
function pickTreeOverlayPoint(mouseEvent) {
    const activeData = treeActiveFilteredData || treeRawData;
    if (!treeOverlayPoints || !activeData) return;

    const rect = renderArea.getBoundingClientRect();
    const ndc = new THREE.Vector2(
        ((mouseEvent.clientX - rect.left) / rect.width) * 2 - 1,
        -((mouseEvent.clientY - rect.top) / rect.height) * 2 + 1
    );
    const camera = viewer.scene.getActiveCamera();
    if (!camera) return;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(ndc, camera);
    // Point-pick tolerance in world meters, scaled to view distance
    const viewDist = camera.position.distanceTo(treeOverlayPoints.position);
    raycaster.params.Points.threshold = Math.max(0.02, viewDist * 0.004);

    const hits = raycaster.intersectObject(treeOverlayPoints, false);
    if (!hits.length) {
        showToast('No point found under cursor. Click directly on a point.');
        return;
    }

    const i = hits[0].index;
    const { xyz, intensity, classification, scalars, targetCount } = activeData;
    // Filtered datasets reuse the original tree centroid (overlay node position never moves)
    const ox = activeData.cx !== undefined ? activeData.cx : (treeRawData ? treeRawData.cx : 0);
    const oy = activeData.cy !== undefined ? activeData.cy : (treeRawData ? treeRawData.cy : 0);
    const oz = activeData.cz !== undefined ? activeData.cz : (treeRawData ? treeRawData.cz : 0);
    const world = new THREE.Vector3(
        xyz[i * 3] + ox,
        xyz[i * 3 + 1] + oy,
        xyz[i * 3 + 2] + oz
    );

    const tidArr = scalars ? (scalars.itc || scalars.tree_id || scalars.treeID) : null;
    const isTarget = targetCount === undefined || i < targetCount;
    const tid = isTarget ? treeOverlayItc : (tidArr ? Math.round(tidArr[i]) : 0);

    highlightPickedPoint(world, `Tree #${treeOverlayItc}${isTarget ? '' : ` (neighbor #${tid})`} | Z: ${world.z.toFixed(2)}m`);

    const inspX = document.getElementById('inspX');
    const inspY = document.getElementById('inspY');
    const inspZ = document.getElementById('inspZ');
    if (inspX) inspX.textContent = world.x.toFixed(3);
    if (inspY) inspY.textContent = world.y.toFixed(3);
    if (inspZ) inspZ.textContent = world.z.toFixed(3);

    const inspClass = document.getElementById('inspClass');
    const inspInt = document.getElementById('inspInt');
    const inspITC = document.getElementById('inspITC');
    const btnInspectViewTree = document.getElementById('btnInspectViewTree');

    if (inspClass) inspClass.textContent = classification ? formatClassName(classification[i]) : '-';
    if (inspInt) inspInt.textContent = intensity ? intensity[i] : '-';
    if (inspITC) inspITC.textContent = tid > 0 ? tid : '-';
    if (btnInspectViewTree) btnInspectViewTree.style.display = 'none'; // already in the tree view

    const extraTbody = document.getElementById('inspExtraAttrs');
    if (extraTbody) {
        extraTbody.innerHTML = '';
        if (scalars) {
            for (const [k, arr] of Object.entries(scalars)) {
                if (!arr || i >= arr.length) continue;
                let val = arr[i];
                if (typeof val === 'number' && !Number.isInteger(val)) val = Number(val.toFixed(3));
                const label = k.replace(/_/g, ' ').replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                const tr = document.createElement('tr');
                tr.innerHTML = `<td>${label}:</td><td>${val}</td>`;
                extraTbody.appendChild(tr);
            }
        }
    }

    if (inspectCard) inspectCard.style.display = 'block';
}

async function pickPointAt(mouseEvent) {
    if (treeOverlayPoints && treeOverlayPoints.visible) {
        return pickTreeOverlayPoint(mouseEvent);
    }

    const rect = renderArea.getBoundingClientRect();
    const x = mouseEvent.clientX - rect.left;
    const y = mouseEvent.clientY - rect.top;

    // Direct pointcloud point hit required (no sky/empty space fallback)
    const pickResult = getIntersectionAt(x, y, false);
    if (!pickResult || !pickResult.position) {
        showToast('No point found under cursor. Click directly on a point.');
        return;
    }

    const pos = pickResult.position;

    // Immediately highlight the picked point in the 3D viewport
    highlightPickedPoint(pos, `Picked Point | Z: ${pos.z.toFixed(2)}m`);

    const inspX = document.getElementById('inspX');
    const inspY = document.getElementById('inspY');
    const inspZ = document.getElementById('inspZ');
    if (inspX) inspX.textContent = pos.x.toFixed(3);
    if (inspY) inspY.textContent = pos.y.toFixed(3);
    if (inspZ) inspZ.textContent = pos.z.toFixed(3);

    const inspClass = document.getElementById('inspClass');
    const inspInt = document.getElementById('inspInt');
    const inspITC = document.getElementById('inspITC');
    const btnInspectViewTree = document.getElementById('btnInspectViewTree');

    const updateViewTreeButton = (itcVal) => {
        if (!btnInspectViewTree) return;
        const num = Number(itcVal);
        if (itcVal !== null && itcVal !== undefined && !isNaN(num) && num > 0) {
            btnInspectViewTree.style.display = 'block';
            btnInspectViewTree.textContent = `View Tree #${num} (All Points)`;
            btnInspectViewTree.onclick = () => {
                if (inspectCard) inspectCard.style.display = 'none';
                if (pickedPointMarker) pickedPointMarker.visible = false;
                window.viewTreeById(num, pos.x, pos.y);
                postJSON('/api/event', {
                    event: 'tree_view_requested',
                    itc: num,
                    x: pos.x,
                    y: pos.y
                }).catch(() => {});
            };
        } else {
            btnInspectViewTree.style.display = 'none';
        }
    };

    // Extract attributes from Potree GPU pick if immediately available
    const initialClass = extractAttr(pickResult.point, 'classification');
    const initialInt = extractAttr(pickResult.point, 'intensity');
    const initialITC = extractAttr(pickResult.point, 'itc', 'tree_id', 'treeID');

    if (inspClass) inspClass.textContent = formatClassName(initialClass);
    if (inspInt) inspInt.textContent = initialInt !== null ? initialInt : '-';
    if (inspITC) inspITC.textContent = initialITC !== null ? initialITC : '-';
    updateViewTreeButton(initialITC);

    // Populate any extra scalar fields dynamically into inspExtraAttrs
    const extraTbody = document.getElementById('inspExtraAttrs');
    const extraAttrs = {};
    if (extraTbody) {
        extraTbody.innerHTML = '';
        const standardKeys = new Set(['x', 'y', 'z', 'position', 'classification', 'intensity', 'itc', 'tree_id', 'treeID', 'pointCloud', 'index']);
        if (pickResult.point) {
            for (const [k, v] of Object.entries(pickResult.point)) {
                if (standardKeys.has(k) || k.startsWith('_')) continue;
                let val = extractAttr(pickResult.point, k);
                if (val !== null && val !== undefined) {
                    extraAttrs[k] = val;
                    if (typeof val === 'number') {
                        val = Number.isInteger(val) ? val : Number(val.toFixed(3));
                    }
                    const tr = document.createElement('tr');
                    const label = k.replace(/_/g, ' ').replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                    tr.innerHTML = `<td>${label}:</td><td>${val}</td>`;
                    extraTbody.appendChild(tr);
                }
            }
        }
    }
    if (inspectCard) inspectCard.style.display = 'block';

    try {
        const data = await postJSON('/api/pick_point', {
            x: pos.x,
            y: pos.y,
            z: pos.z,
            itc: initialITC,
            attributes: extraAttrs
        });
        const picked = data.picked;

        if (picked) {
            const finalClass = picked.classification ?? initialClass;
            const finalInt = picked.intensity ?? initialInt;
            if (inspClass) inspClass.textContent = formatClassName(finalClass);
            if (inspInt) inspInt.textContent = finalInt ?? '-';

            const activeITC = (picked.itc !== undefined && picked.itc !== null && Number(picked.itc) > 0)
                ? Number(picked.itc)
                : (initialITC !== null ? Number(initialITC) : 0);

            if (picked.staged_hit) {
                if (inspITC) inspITC.textContent = `${picked.staged_hit.label} (Staged)`;
                updatePickedPointLabel(picked.staged_hit.label, picked.staged_hit.color || '#10b981');
                updateViewTreeButton(null);
            } else if (activeITC > 0) {
                if (inspITC) inspITC.textContent = activeITC;
                updatePickedPointLabel(`Tree #${activeITC}`, '#10b981');
                updateViewTreeButton(activeITC);
            } else {
                if (inspITC) inspITC.textContent = 'None';
                updateViewTreeButton(null);
                if (finalClass !== null && finalClass !== undefined) {
                    updatePickedPointLabel(formatClassName(finalClass), '#00f5ff');
                }
            }
        } else if (initialClass !== null) {
            updatePickedPointLabel(formatClassName(initialClass), '#00f5ff');
            updateViewTreeButton(initialITC);
        }
    } catch (e) {
        console.error('Pick point API error:', e);
    }
}

// ---------------------------------------------------------------------------
// Selection Result Handler
// ---------------------------------------------------------------------------
function handleSelectionResult(data) {
    currentSelectionStats = data;

    if (!selectionSummary || !selectionBanner) return;

    if (data.mode === 'staged') {
        const vCount = data.vertex_count || polygonPoints.length;
        selectionSummary.innerHTML = `Polygon Delineated: <strong>${vCount}</strong> vertices | Ready to stage`;
        selectionBanner.style.display = 'flex';
        showToast(`Polygon ready (${vCount} points). Assign Tree ID or Mark Noise to stage visually.`);
        return;
    }

    const count = data.count || 0;
    const itcText = (data.itc_values && data.itc_values.length > 0)
        ? ` | ITC: [${data.itc_values.slice(0, 5).join(', ')}${data.itc_values.length > 5 ? '...' : ''}]`
        : '';
    const classText = (data.classes && data.classes.length > 0)
        ? ` | Classes: [${data.classes.join(', ')}]`
        : '';

    selectionSummary.innerHTML = `Selected: <strong>${count.toLocaleString()}</strong> points${classText}${itcText}`;
    selectionBanner.style.display = count > 0 ? 'flex' : 'none';
    showToast(`Found ${count.toLocaleString()} points inside selection.`);
}

function clearSelection(notifyServer = true) {
    clearPickedPointHighlight();
    if (inspectCard) inspectCard.style.display = 'none';

    if (polygonLine) {
        overlayScene.remove(polygonLine);
        polygonLine = null;
    }
    if (activeSelectionMesh) {
        if (overlayScene.children.includes(activeSelectionMesh)) {
            overlayScene.remove(activeSelectionMesh);
        }
        if (viewer.scene.scene.children.includes(activeSelectionMesh)) {
            viewer.scene.scene.remove(activeSelectionMesh);
        }
        if (viewer.scene.volumes && viewer.scene.volumes.includes(activeSelectionMesh)) {
            viewer.scene.removeVolume(activeSelectionMesh);
        }
        activeSelectionMesh = null;
    }
    polygonPoints = [];
    currentSelectionStats = null;
    if (selectionBanner) selectionBanner.style.display = 'none';

    if (notifyServer) {
        postJSON('/api/select_polygon', { coordinates: [], points_3d: [] })
            .catch(err => console.warn('Clear-selection notify failed:', err));
        showToast('Selection cleared.');
    }
}

// ---------------------------------------------------------------------------
// Mouse & Click Event Listeners
// ---------------------------------------------------------------------------
let mouseDownPos = { x: 0, y: 0 };

renderArea.addEventListener('mousedown', (e) => {
    mouseDownPos = { x: e.clientX, y: e.clientY };
});

window.addEventListener('keydown', (e) => {
    if ((e.key === 'Control' || e.key === 'Meta') && (activeTool === 'nav' || activeTool === 'inspect')) {
        renderArea.style.cursor = 'crosshair';
    }
});

window.addEventListener('keyup', (e) => {
    if (e.key === 'Control' || e.key === 'Meta') {
        if (activeTool === 'nav') {
            renderArea.style.cursor = 'default';
        }
    }
});

window.addEventListener('blur', () => {
    if (activeTool === 'nav') {
        renderArea.style.cursor = 'default';
    }
});

renderArea.addEventListener('click', (e) => {
    // If user dragged to orbit/pan camera, ignore click to avoid accidental picking
    const isDrag = Math.hypot(e.clientX - mouseDownPos.x, e.clientY - mouseDownPos.y) > 5;
    if (isDrag) {
        return;
    }

    const isCtrl = e.ctrlKey || e.metaKey;

    if (treeOverlayPoints && treeOverlayPoints.visible) {
        // Filter Mode ON: clicks toggle whole-tree selection
        if (treeFilterMode) {
            pickTreeInStand(e);
            return;
        }
        // Filter Mode OFF: Ctrl+Click picks a single point (attributes card), like the Layers tab
        if (isCtrl) {
            pickPointAt(e);
            return;
        }
    }

    if (activeTool === 'polygon') {
        const rect = renderArea.getBoundingClientRect();
        const pick = getIntersectionAt(e.clientX - rect.left, e.clientY - rect.top);
        if (pick && pick.position) {
            polygonPoints.push(pick.position.clone());

            const flat = [];
            for (const pt of polygonPoints) {
                flat.push(pt.x, pt.y, pt.z);
            }
            polygonLine.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(flat), 3));
            polygonLine.geometry.attributes.position.needsUpdate = true;
        }
    } else if (isCtrl && (activeTool === 'inspect' || activeTool === 'nav')) {
        pickPointAt(e);
    } else if (activeTool === 'inspect' && !isCtrl) {
        showToast('Hold Ctrl and click on a point to pick it.');
    }
});

renderArea.addEventListener('dblclick', (e) => {
    if (activeTool === 'polygon') {
        e.preventDefault();
        finishPolygonSelection();
        return;
    }

    if (treeOverlayPoints && treeOverlayPoints.visible) {
        zoomToTreeAt(e);
    }
});

renderArea.addEventListener('contextmenu', (e) => {
    if (activeTool === 'polygon') {
        e.preventDefault();
        finishPolygonSelection();
    }
});

if (closeInspect) {
    closeInspect.addEventListener('click', () => {
        if (inspectCard) inspectCard.style.display = 'none';
        clearPickedPointHighlight();
    });
}

// Quick Staging Actions from Floating Banner
if (btnQuickAssignTree) {
    btnQuickAssignTree.addEventListener('click', () => {
        if (!polygonPoints || polygonPoints.length < 3) {
            showToast('Draw a polygon first before staging Tree ID.');
            return;
        }
        const val = parseInt(quickTreeIdInput.value, 10) || 1;
        const coords2d = polygonPoints.map(p => [p.x, p.y]);
        const coords3d = polygonPoints.map(p => [p.x, p.y, p.z]);
        const edit = {
            type: 'polygon',
            coords: coords2d,
            points_3d: coords3d,
            action: 'assign_itc',
            value: val,
            label: `Tree #${val}`,
            color: '#10b981'
        };

        postJSON('/api/stage_edit', edit)
        .then(data => {
            if (data.success) {
                window.addStagedVisualPolygon(data.edit);
                clearSelection(false);
                quickTreeIdInput.value = val + 1; // Auto-increment for convenient crown drafting
                showToast(`Staged ${data.edit.label} visually. Click Commit when ready.`);
            }
        })
        .catch(err => {
            console.error('Error staging tree:', err);
            showToast('Error staging tree: ' + err.message);
        });
    });
}

if (btnQuickNoise) {
    btnQuickNoise.addEventListener('click', () => {
        if (!polygonPoints || polygonPoints.length < 3) {
            showToast('Draw a polygon first before staging noise.');
            return;
        }
        const coords2d = polygonPoints.map(p => [p.x, p.y]);
        const coords3d = polygonPoints.map(p => [p.x, p.y, p.z]);
        const edit = {
            type: 'polygon',
            coords: coords2d,
            points_3d: coords3d,
            action: 'mark_noise',
            value: 9999,
            label: 'Noise (7)',
            color: '#ef4444'
        };

        postJSON('/api/stage_edit', edit)
        .then(data => {
            if (data.success) {
                window.addStagedVisualPolygon(data.edit);
                clearSelection(false);
                showToast('Staged Noise (Class 7) visually. Click Commit when ready.');
            }
        })
        .catch(err => {
            console.error('Error staging noise:', err);
            showToast('Error staging noise: ' + err.message);
        });
    });
}

// ---------------------------------------------------------------------------
// Public API for Python/Qt Ribbon & Settings Controls
// ---------------------------------------------------------------------------
// Render modes.  Performance is what the app shipped with.  Quality trades speed for
// looks: PARABOLOID is potree's round spherical-looking splat, the size drops, HQ splat
// shading goes on and the budget goes to the top of what the ribbon offers.  Quality
// also switches sizing to FIXED, because ADAPTIVE grows coarse nodes back to large
// splats (the "huge points when zoomed out" complaint) and Fixed skips the visibility
// texture path altogether.  Balanced sits in the middle.
//
// A mode is a scene-wide change: it restyles every layer and becomes the starting style
// for the next cloud opened.  Mixing one layer in Quality with another in Performance
// reads as a rendering glitch rather than a deliberate choice.
const RENDER_MODES = {
    performance: { pointType: 'SQUARE', pointSizeType: 'ADAPTIVE', pointSize: 1.0,
                   splatQuality: 'standard', pointBudget: 2_500_000, minNodeSize: 10 },
    balanced: { pointType: 'CIRCLE', pointSizeType: 'ADAPTIVE', pointSize: 0.8,
                splatQuality: 'standard', pointBudget: 5_000_000, minNodeSize: 10 },
    quality: { pointType: 'PARABOLOID', pointSizeType: 'FIXED', pointSize: 0.45,
               splatQuality: 'standard', pointBudget: 15_000_000, minNodeSize: 5 }
};

window.setRenderMode = (mode) => {
    const preset = RENDER_MODES[mode];
    if (!preset) return listLayers();
    viewerSettings.renderMode = mode;
    viewerSettings.pointType = preset.pointType;
    viewerSettings.pointSizeType = preset.pointSizeType;
    viewerSettings.pointSize = preset.pointSize;
    viewerSettings.splatQuality = preset.splatQuality;
    viewerSettings.pointBudget = preset.pointBudget;
    viewerSettings.minNodeSize = preset.minNodeSize;
    for (const pc of viewer.scene.pointclouds) {
        const style = layerStyle(pc);
        style.pointType = preset.pointType;
        style.pointSizeType = preset.pointSizeType;
        style.pointSize = preset.pointSize;
        applyMaterialSettings(pc);
    }
    // Reuse the app's own wrappers, since they know which potree calls actually take.
    for (const call of [['setPointBudget', preset.pointBudget],
                        ['setMinNodeSize', preset.minNodeSize],
                        ['setSplatQuality', preset.splatQuality]]) {
        if (typeof window[call[0]] === 'function') window[call[0]](call[1]);
    }
    viewer.render();
    showToast('Render mode: ' + mode);
    return listLayers();
};

window.getRenderMode = () => viewerSettings.renderMode || 'performance';

// The scene-wide half of the look, for the widgets that are not per-layer: budget, node
// size, splat quality and EDL all live on the viewer, so the ribbon reads them back from
// here rather than keeping a second copy of the preset table in Python.
window.getRenderSettings = () => ({
    renderMode: viewerSettings.renderMode || 'performance',
    pointBudget: viewerSettings.pointBudget,
    minNodeSize: viewerSettings.minNodeSize,
    splatQuality: viewerSettings.splatQuality,
    edlEnabled: viewerSettings.edlEnabled
});

// Size, shape and sizing mode act on the active layer only -- each cloud keeps its own
// style -- while viewerSettings keeps the value as the starting point for the next
// cloud the user opens.  updateAllPointClouds() stays for the settings that really are
// viewer-wide.
window.setPointShape = (shape) => {
    viewerSettings.pointType = shape;
    if (currentPointCloud) {
        layerStyle(currentPointCloud).pointType = shape;
        applyMaterialSettings(currentPointCloud);
    }
    viewer.render();
    showToast(`Point Type: ${shape}`);
};

window.setPointSizeType = (type) => {
    viewerSettings.pointSizeType = type;
    if (currentPointCloud) {
        layerStyle(currentPointCloud).pointSizeType = type;
        applyMaterialSettings(currentPointCloud);
    }
    viewer.render();
    showToast(`Point Sizing: ${type}`);
};

window.setPointSize = (size) => {
    viewerSettings.pointSize = Number(size);
    if (currentPointCloud) {
        layerStyle(currentPointCloud).pointSize = Number(size);
        applyMaterialSettings(currentPointCloud);
    }
    if (treeOverlayPoints && treeOverlayPoints.material) {
        treeOverlayPoints.material.size = getTreePointSize();
        treeOverlayPoints.material.needsUpdate = true;
    }
    viewer.render();
};

window.setSplatQuality = (quality) => {
    viewerSettings.splatQuality = quality;
    const isHQ = (quality === 'hq' || quality === true);
    viewer.useHQ = isHQ;
    showToast(`Splat Quality: ${isHQ ? 'High Quality (HQ)' : 'Standard'}`);
};

window.setPointBudget = (budget) => {
    viewerSettings.pointBudget = Number(budget);
    viewer.setPointBudget(Number(budget));
    showToast(`Point Budget: ${(budget / 1_000_000).toFixed(1)}M points`);
};

window.setEDLEnabled = (enabled) => {
    const isEnabled = Boolean(enabled);
    viewerSettings.edlEnabled = isEnabled;
    viewer.setEDLEnabled(isEnabled);
    showToast(`EDL: ${isEnabled ? 'Enabled' : 'Disabled'}`);
};

window.setEDLRadius = (radius) => {
    const val = Number(radius);
    viewerSettings.edlRadius = val;
    viewer.setEDLRadius(val);
};

window.setEDLStrength = (strength) => {
    const val = Number(strength);
    viewerSettings.edlStrength = val;
    viewer.setEDLStrength(val);
};

window.setEDLOpacity = (opacity) => {
    const val = Number(opacity);
    viewerSettings.edlOpacity = val;
    viewer.setEDLOpacity(val);
};

window.setColorAttribute = (attr) => {
    viewerSettings.colorMode = attr;
    if (currentPointCloud) applyColorAttribute(currentPointCloud, attr);
    updateTreeColors();
    viewer.render();
    showToast(`Color by: ${attr}`);
};

window.setMinNodeSize = (val) => {
    const num = Number(val);
    viewerSettings.minNodeSize = num;
    viewer.setMinNodeSize(num);
    showToast('Min Node Size: ' + num + ' px');
};

window.setBackground = (bg) => {
    viewerSettings.background = bg;
    viewer.setBackground(bg);
    showToast('Background: ' + bg);
};

window.getIntersectionAt = getIntersectionAt;
window.listLayers = listLayers;
window.setActiveLayer = (name) => setActiveLayer(name);
window.setLayerVisible = (name, visible) => setLayerVisible(name, visible);
window.setLayerColorAttribute = (name, attr) => setLayerColorAttribute(name, attr);
window.setLayerPointSize = (name, size) => setLayerPointSize(name, size);
window.removeLayer = (name) => removeLayer(name);
window.activeLayerName = () => activeLayerName;
window.clearSelection = clearSelection;
window.highlightPickedPoint = highlightPickedPoint;
window.clearPickedPointHighlight = clearPickedPointHighlight;

// ---------------------------------------------------------------------------
// Layer Management API
// ---------------------------------------------------------------------------
window.setPointCloudVisibility = (visible) => {
    const v = Boolean(visible);
    if (viewer.scene && viewer.scene.pointclouds) {
        for (const pc of viewer.scene.pointclouds) {
            pc.visible = v;
        }
    }
    showToast(`Point Cloud Layer: ${v ? 'Visible' : 'Hidden'}`);
};

window.setStagedAllVisibility = (visible) => {
    const v = Boolean(visible);
    for (const group of stagedVisuals.values()) {
        group.visible = v;
    }
    showToast(`Staged Trees Layer: ${v ? 'Visible' : 'Hidden'}`);
};

window.setStagedItemVisibility = (editId, visible) => {
    if (stagedVisuals.has(editId)) {
        stagedVisuals.get(editId).visible = Boolean(visible);
    }
};

window.setHighlightVisibility = (visible) => {
    if (pickedPointMarker) {
        pickedPointMarker.visible = Boolean(visible);
    }
};

window.setSelectionVisibility = (visible) => {
    const v = Boolean(visible);
    if (polygonLine) polygonLine.visible = v;
    if (activeSelectionMesh) activeSelectionMesh.visible = v;
};

window.setAllLayersVisibility = (visible) => {
    const v = Boolean(visible);
    window.setPointCloudVisibility(v);
    window.setStagedAllVisibility(v);
    window.setHighlightVisibility(v);
    window.setSelectionVisibility(v);
    showToast(`All Layers: ${v ? 'Visible' : 'Hidden'}`);
};

window.zoomToStagedEdit = (editId) => {
    if (!stagedVisuals.has(editId)) return;
    const group = stagedVisuals.get(editId);
    const box = new THREE.Box3().setFromObject(group);
    if (!box.isEmpty()) {
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z, 8);
        if (viewer.scene && viewer.scene.view) {
            viewer.scene.view.position.set(center.x, center.y - maxDim * 1.5, center.z + maxDim * 1.2);
            viewer.scene.view.lookAt(center);
            showToast('Zoomed to staged edit');
        }
    }
};

// ===========================================================================
// Full-Density Tree View
//
// The octree keeps ~127M of 213M points (1 mm Poisson decimation) for fast
// streaming. A single tree's full point set lives in the LAS master; the
// server streams it here on demand as a binary blob and we render it as a
// second, complete overlay cloud — the main octree stays as context.
// ===========================================================================

let treeOverlayPoints = null;
let treeOverlayItc = null;
let treeContextHidden = [];

// Open: dim the world to just the tree. Close: bring the context back.
function hideTreeContext() {
    showTreeContext();
    viewer.scene.pointclouds.forEach(pc => {
        if (pc.visible) { pc.visible = false; treeContextHidden.push(pc); }
    });
    if (typeof stagedVisuals !== 'undefined') {
        stagedVisuals.forEach(group => {
            if (group.visible) { group.visible = false; treeContextHidden.push(group); }
        });
    }
}

function showTreeContext() {
    treeContextHidden.forEach(o => { o.visible = true; });
    treeContextHidden = [];
}

function treeColorLUT(t) {
    const s = Math.min(1, Math.max(0, t)) * (VIRIDIS16.length - 1);
    const i = Math.floor(s);
    const f = s - i;
    const a = VIRIDIS16[i];
    const b = VIRIDIS16[Math.min(VIRIDIS16.length - 1, i + 1)];
    return [(a[0] + (b[0] - a[0]) * f) / 255, (a[1] + (b[1] - a[1]) * f) / 255, (a[2] + (b[2] - a[2]) * f) / 255];
}

let treeRawData = null;
let treeColorMode = 'auto';
let treePointSizeUser = 1.0;

// ---- Tree bounding box + measurement (Forestry tab) ----
let treeBBoxHelper = null;
let treeBBoxLabel = null;
let treeOverlayTargetBBox = null;   // world-space {min,max} of the target tree
window.treeBBoxEnabled = true;      // bounding box visible by default

function treeTextSprite(text, scaleBase) {
    let canvas = document.createElement('canvas');
    let ctx = canvas.getContext('2d');
    ctx.font = 'bold 72px sans-serif';
    const w = Math.min(900, ctx.measureText(text).width + 44);
    canvas = document.createElement('canvas');
    canvas.width = Math.ceil(w); canvas.height = 110;
    ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(10,14,20,0.78)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = 'bold 72px sans-serif';
    ctx.fillStyle = '#ffd27f';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 22, canvas.height / 2);
    const tex = new THREE.CanvasTexture(canvas);
    tex.minFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true });
    const sprite = new THREE.Sprite(mat);
    const s = scaleBase || 1.4;
    sprite.scale.set(s * (canvas.width / canvas.height), s, 1);
    sprite.renderOrder = 30;
    return sprite;
}

function showTreeBBox() {
    if (treeBBoxHelper) {
        viewer.scene.scene.remove(treeBBoxHelper);
        treeBBoxHelper.geometry.dispose();
        treeBBoxHelper.material.dispose();
        treeBBoxHelper = null;
    }
    if (treeBBoxLabel) {
        viewer.scene.scene.remove(treeBBoxLabel);
        treeBBoxLabel.material.map.dispose();
        treeBBoxLabel.material.dispose();
        treeBBoxLabel = null;
    }
    if (!treeOverlayTargetBBox || window.treeBBoxEnabled === false) {
        viewer.render();
        return;
    }
    const b = treeOverlayTargetBBox;
    const box = new THREE.Box3(new THREE.Vector3(...b.min), new THREE.Vector3(...b.max));
    treeBBoxHelper = new THREE.Box3Helper(box, new THREE.Color(0xffaa33));
    treeBBoxHelper.renderOrder = 29;
    treeBBoxHelper.frustumCulled = false;
    viewer.scene.scene.add(treeBBoxHelper);

    const h = b.max[2] - b.min[2];
    treeBBoxLabel = treeTextSprite(`H ${h.toFixed(2)} m`, Math.min(0.8, Math.max(0.25, h * 0.04)));
    treeBBoxLabel.position.set(b.min[0], b.min[1], b.max[2]);
    viewer.scene.scene.add(treeBBoxLabel);
    viewer.render();
}

window.setTreeBBox = function (visible) {
    window.treeBBoxEnabled = visible !== false;
    showTreeBBox();
};

// ---- Measure: click points on the tree, 3D distances shown in metres ----
let measureOn = false;
let measurePts = [];
let measureObjs = [];
let measureDownXY = null;

function clearMeasure() {
    measureObjs.forEach(o => {
        viewer.scene.scene.remove(o);
        if (o.geometry) o.geometry.dispose();
        if (o.material) {
            if (o.material.map) o.material.map.dispose();
            o.material.dispose();
        }
    });
    measureObjs = [];
    measurePts = [];
    viewer.render();
}

window.setTreeMeasure = function (on) {
    measureOn = !!on;
    if (!measureOn) return;
    const canvas = document.getElementById('potree_render_area').querySelector('canvas')
                   || viewer.canvas || document.querySelector('canvas');
    if (!canvas || canvas.dataset.treeMeasure) return;
    canvas.dataset.treeMeasure = '1';

    canvas.addEventListener('mousedown', (e) => { measureDownXY = [e.clientX, e.clientY]; });
    canvas.addEventListener('click', (e) => {
        if (!measureOn) return;
        if (measureDownXY) {
            const dx = e.clientX - measureDownXY[0], dy = e.clientY - measureDownXY[1];
            if (dx * dx + dy * dy > 36) return;   // drag, not click
        }
        if (!treeOverlayPoints || !treeOverlayPoints.visible) return;
        const rect = canvas.getBoundingClientRect();
        const ndc = new THREE.Vector2(
            ((e.clientX - rect.left) / rect.width) * 2 - 1,
            -((e.clientY - rect.top) / rect.height) * 2 + 1);
        const ray = new THREE.Raycaster();
        ray.params.Points.threshold = 0.15;
        ray.setFromCamera(ndc, viewer.scene.camera);
        const hits = ray.intersectObject(treeOverlayPoints, false);
        if (!hits.length) return;
        const p = hits[0].point.clone();
        p.applyMatrix4(treeOverlayPoints.matrixWorld);   // geometry is centroid-relative
        addMeasurePoint(p);
    });
    canvas.addEventListener('dblclick', () => { if (measureOn) clearMeasure(); });
};

function addMeasurePoint(worldPoint) {
    if (measurePts.length && worldPoint.distanceTo(measurePts[measurePts.length - 1]) < 0.01) return;
    measurePts.push(worldPoint.clone());

    const r = treeOverlayTargetBBox
        ? Math.min(0.6, Math.max(0.12, (treeOverlayTargetBBox.max[2] - treeOverlayTargetBBox.min[2]) * 0.03))
        : 0.15;
    const marker = new THREE.Mesh(
        new THREE.SphereGeometry(r, 12, 12),
        new THREE.MeshBasicMaterial({ color: 0xff5533, depthTest: false, transparent: true, opacity: 0.95 }));
    marker.position.copy(worldPoint);
    marker.renderOrder = 30;
    viewer.scene.scene.add(marker);
    measureObjs.push(marker);

    if (measurePts.length > 1) {
        const a = measurePts[measurePts.length - 2];
        const b = worldPoint;
        const lineGeo = new THREE.BufferGeometry().setFromPoints([a, b]);
        const line = new THREE.Line(lineGeo,
            new THREE.LineBasicMaterial({ color: 0xffdd55, depthTest: false, transparent: true, opacity: 0.95 }));
        line.renderOrder = 30;
        viewer.scene.scene.add(line);
        measureObjs.push(line);

        const d3 = a.distanceTo(b);
        const dh = Math.hypot(b.x - a.x, b.y - a.y);
        const label = treeTextSprite(`d ${d3.toFixed(2)} m  (h ${dh.toFixed(2)})`, 1.1);
        label.position.copy(a).add(b).multiplyScalar(0.5);
        viewer.scene.scene.add(label);
        measureObjs.push(label);
    }
    viewer.render();
}
// ---- end bounding box + measurement ----

// Neighbor Stand Filter State (Exclusive to Forestry Tab)
let treeFilterActive = false;
let treeFilterMode = false;
let treeFilterSelectedIds = new Set();
let treeActiveFilteredData = null;

function getStandTreeList() {
    if (!treeRawData) return [];
    const { count, targetCount, neighborCount, scalars, ff3dTrees, isSegmented } = treeRawData;
    const tidArr = scalars ? (scalars.itc || scalars.tree_id || scalars.treeID) : null;

    const treeCounts = new Map();
    const isSeg = Boolean(isSegmented || (ff3dTrees && ff3dTrees > 0));

    if (isSeg && tidArr) {
        for (let i = 0; i < count; i++) {
            const tid = Math.round(tidArr[i]);
            treeCounts.set(tid, (treeCounts.get(tid) || 0) + 1);
        }
    } else {
        // Target tree points:
        treeCounts.set(treeOverlayItc, targetCount);

        // Neighbor points:
        if (tidArr && neighborCount > 0) {
            for (let i = targetCount; i < count; i++) {
                const tid = Math.round(tidArr[i]);
                treeCounts.set(tid, (treeCounts.get(tid) || 0) + 1);
            }
        } else if (neighborCount > 0) {
            treeCounts.set(0, neighborCount);
        }
    }

    const list = [];
    for (const [tid, ptCount] of treeCounts.entries()) {
        if (isSeg && tid === treeOverlayItc && ptCount <= 0) {
            continue;
        }
        list.push({
            id: tid,
            isTarget: isSeg ? false : (tid === treeOverlayItc),
            count: ptCount
        });
    }
    list.sort((a, b) => {
        if (a.isTarget) return -1;
        if (b.isTarget) return 1;
        if (a.id <= 0) return 1;
        if (b.id <= 0) return -1;
        return b.count - a.count;
    });
    return list;
}

window.getStandTrees = getStandTreeList;

window.setTreeFilterMode = function (enabled) {
    treeFilterMode = !!enabled;
    // Entering filter mode starts from a clean slate: every tree unselected
    if (treeFilterMode) treeFilterSelectedIds = new Set();
    if (renderArea) {
        renderArea.style.cursor = treeFilterMode ? 'pointer' : 'default';
    }
    updateStandFilterHud();
    updateTreeColors();
    showToast(treeFilterMode ? 'Filter Mode ON: Click trees in 3D to select/deselect them' : 'Filter Mode OFF');
};

window.isTreeFilterMode = function () {
    return treeFilterMode;
};

window.setTreeFilterSelection = function (treeIds) {
    treeFilterSelectedIds = new Set((treeIds || []).map(Number));
    updateStandFilterHud();
    updateTreeColors();
};

window.toggleTreeFilterId = function (tid) {
    const num = Number(tid);
    if (treeFilterSelectedIds.has(num)) {
        treeFilterSelectedIds.delete(num);
    } else {
        treeFilterSelectedIds.add(num);
    }
    updateStandFilterHud();
    updateTreeColors();
    postJSON('/api/event', {
        event: 'tree_filter_selection_changed',
        selected_ids: Array.from(treeFilterSelectedIds),
        toggled_id: num,
        is_selected: treeFilterSelectedIds.has(num)
    }).catch(() => {});

    const label = num === 0 ? 'Canopy / Ground' : `Tree #${num}`;
    showToast(`${label} ${treeFilterSelectedIds.has(num) ? 'selected' : 'deselected'} (${treeFilterSelectedIds.size} trees selected for filter)`);
};

function pickTreeInStand(e) {
    if (!treeOverlayPoints || !treeRawData) return;
    const rect = renderArea.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const camera = viewer.scene.getActiveCamera();
    if (!camera) return;

    const mouse = new THREE.Vector2(
        (clickX / rect.width) * 2 - 1,
        -(clickY / rect.height) * 2 + 1
    );
    const raycaster = new THREE.Raycaster();
    const ptSize = getTreePointSize();
    raycaster.params.Points.threshold = Math.max(0.35, ptSize * 0.45);
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(treeOverlayPoints);
    if (intersects && intersects.length > 0) {
        // Project all candidates to screen coordinates to find the point closest to the mouse cursor
        let bestCandidate = null;
        let minScreenDist = Infinity;

        for (const hit of intersects) {
            const ndc = hit.point.clone().project(camera);
            const px = ((ndc.x + 1) / 2) * rect.width;
            const py = ((-ndc.y + 1) / 2) * rect.height;
            const sDist = Math.hypot(px - clickX, py - clickY);

            // Favor hit closest to cursor on screen. If two hits are within 2.5px, pick the one closer to the ray.
            if (sDist < minScreenDist - 2.5 ||
                (Math.abs(sDist - minScreenDist) <= 2.5 && hit.distanceToRay < (bestCandidate ? bestCandidate.hit.distanceToRay : Infinity))) {
                minScreenDist = sDist;
                bestCandidate = { hit, sDist };
            }
        }

        if (bestCandidate && minScreenDist <= 32) {
            const hitIdx = bestCandidate.hit.index;
            const activeData = treeActiveFilteredData || treeRawData;
            const tidArr = activeData.scalars ? (activeData.scalars.itc || activeData.scalars.tree_id || activeData.scalars.treeID) : null;
            let hitTid = 0;
            if (hitIdx < activeData.targetCount) {
                hitTid = treeOverlayItc;
            } else if (tidArr) {
                hitTid = Math.round(tidArr[hitIdx]);
            }
            window.toggleTreeFilterId(hitTid);
        } else {
            showToast('No tree point hit under cursor');
        }
    } else {
        showToast('No tree point hit under cursor');
    }
}

function zoomToTreeAt(e) {
    if (!treeOverlayPoints || !treeOverlayPoints.visible || !viewer.scene || !viewer.scene.view) return;
    const rect = renderArea.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const camera = viewer.scene.getActiveCamera();
    if (!camera) return;

    const mouse = new THREE.Vector2(
        (clickX / rect.width) * 2 - 1,
        -(clickY / rect.height) * 2 + 1
    );

    const raycaster = new THREE.Raycaster();
    const ptSize = getTreePointSize();
    raycaster.params.Points.threshold = Math.max(0.4, ptSize * 0.55);
    raycaster.setFromCamera(mouse, camera);

    const intersects = raycaster.intersectObject(treeOverlayPoints);
    if (!intersects || intersects.length === 0) return;

    let bestHit = null;
    let minScreenDist = Infinity;
    for (const hit of intersects) {
        const ndc = hit.point.clone().project(camera);
        const px = ((ndc.x + 1) / 2) * rect.width;
        const py = ((-ndc.y + 1) / 2) * rect.height;
        const sDist = Math.hypot(px - clickX, py - clickY);
        if (sDist < minScreenDist) {
            minScreenDist = sDist;
            bestHit = hit;
        }
    }

    if (!bestHit || minScreenDist > 40) return;

    const targetPos = bestHit.point;
    const view = viewer.scene.view;
    const currentRadius = view.radius || 15.0;
    const targetRadius = Math.max(1.5, Math.min(currentRadius * 0.4, 7.0));
    const d = view.direction.clone().multiplyScalar(-targetRadius);
    const endPosition = new THREE.Vector3().addVectors(targetPos, d);

    const startPos = view.position.clone();
    const startRadius = currentRadius;
    const animationDuration = 500;

    if (typeof TWEEN !== 'undefined') {
        const tweenObj = { t: 0 };
        const tween = new TWEEN.Tween(tweenObj)
            .to({ t: 1 }, animationDuration)
            .easing(TWEEN.Easing.Quartic.Out)
            .onUpdate(() => {
                const t = tweenObj.t;
                view.position.lerpVectors(startPos, endPosition, t);
                view.radius = (1 - t) * startRadius + t * targetRadius;
                if (viewer.setMoveSpeed) viewer.setMoveSpeed(view.radius);
                viewer.render();
            })
            .start();
    } else {
        view.position.copy(endPosition);
        view.radius = targetRadius;
        if (viewer.setMoveSpeed) viewer.setMoveSpeed(view.radius);
        viewer.render();
    }
}

window.applyTreeFilter = function (treeIds) {
    if (!treeOverlayPoints || !treeRawData) return;
    const allowed = new Set((treeIds ? treeIds : Array.from(treeFilterSelectedIds)).map(Number));
    if (allowed.size === 0) {
        showToast('Select at least one tree to keep.');
        return;
    }
    treeFilterSelectedIds = allowed;
    treeFilterActive = true;

    const { count, targetCount, neighborCount, xyz, rgb, intensity, classification, scalars } = treeRawData;
    const tidArr = scalars ? (scalars.itc || scalars.tree_id || scalars.treeID) : null;

    const keepIndices = [];
    for (let i = 0; i < count; i++) {
        const tid = (i < targetCount) ? treeOverlayItc : (tidArr ? Math.round(tidArr[i]) : 0);
        if (allowed.has(tid)) {
            keepIndices.push(i);
        }
    }

    if (keepIndices.length === 0) {
        showToast('No points match the selected trees.');
        return;
    }

    const filteredCount = keepIndices.length;
    const newXyz = new Float32Array(filteredCount * 3);
    for (let j = 0; j < filteredCount; j++) {
        const src = keepIndices[j];
        newXyz[j * 3] = xyz[src * 3];
        newXyz[j * 3 + 1] = xyz[src * 3 + 1];
        newXyz[j * 3 + 2] = xyz[src * 3 + 2];
    }

    const filteredScalars = {};
    if (scalars) {
        for (const [k, arr] of Object.entries(scalars)) {
            const newArr = new Float32Array(filteredCount);
            for (let j = 0; j < filteredCount; j++) {
                newArr[j] = arr[keepIndices[j]];
            }
            filteredScalars[k] = newArr;
        }
    }

    let filteredRgb = null;
    if (rgb) {
        filteredRgb = new Uint8Array(filteredCount * 3);
        for (let j = 0; j < filteredCount; j++) {
            const src = keepIndices[j];
            filteredRgb[j * 3] = rgb[src * 3];
            filteredRgb[j * 3 + 1] = rgb[src * 3 + 1];
            filteredRgb[j * 3 + 2] = rgb[src * 3 + 2];
        }
    }

    let filteredIntensity = null;
    if (intensity) {
        filteredIntensity = new Uint16Array(filteredCount);
        for (let j = 0; j < filteredCount; j++) {
            filteredIntensity[j] = intensity[keepIndices[j]];
        }
    }

    let filteredClassification = null;
    if (classification) {
        filteredClassification = new Uint8Array(filteredCount);
        for (let j = 0; j < filteredCount; j++) {
            filteredClassification[j] = classification[keepIndices[j]];
        }
    }

    const geom = treeOverlayPoints.geometry;
    geom.setAttribute('position', new THREE.BufferAttribute(newXyz, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(filteredCount * 3), 3));
    geom.computeBoundingSphere();
    geom.computeBoundingBox();

    treeActiveFilteredData = {
        count: filteredCount,
        targetCount: keepIndices.filter(idx => idx < targetCount).length,
        neighborCount: keepIndices.filter(idx => idx >= targetCount).length,
        xyz: newXyz,
        rgb: filteredRgb,
        intensity: filteredIntensity,
        classification: filteredClassification,
        scalars: filteredScalars,
        bboxMin: treeRawData.bboxMin,
        bboxMax: treeRawData.bboxMax
    };

    treeOverlayPoints.frustumCulled = false;
    updateTreeColors();
    viewer.render();
    updateStandFilterHud();
    showToast(`Filter applied: keeping ${allowed.size} tree(s) (${filteredCount.toLocaleString()} pts)`);
    postJSON('/api/event', {
        event: 'tree_filter_applied',
        filtered_count: filteredCount,
        allowed_trees: Array.from(allowed)
    }).catch(() => {});
};

window.resetTreeFilter = function () {
    if (!treeOverlayPoints || !treeRawData) return;
    treeFilterActive = false;
    treeActiveFilteredData = null;
    treeFilterSelectedIds = new Set(getStandTreeList().map(t => t.id));

    const { count, xyz } = treeRawData;
    const geom = treeOverlayPoints.geometry;
    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(xyz), 3));
    geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
    geom.computeBoundingSphere();
    geom.computeBoundingBox();
    treeOverlayPoints.frustumCulled = false;

    updateTreeColors();
    viewer.render();
    updateStandFilterHud();
    showToast(`Filter reset: all ${count.toLocaleString()} stand points restored`);
    postJSON('/api/event', {
        event: 'tree_filter_reset',
        count: count
    }).catch(() => {});
};

function updateStandFilterHud() {
    // HUD moved to application top menu bar
}

function getTreePointSize() {
    const userSize = (treePointSizeUser !== null && treePointSizeUser !== undefined)
        ? treePointSizeUser
        : (Number(viewerSettings.pointSize) || 1.0);
    return Math.max(0.04, userSize * 0.08);
}

window.setTreePointSize = function (size) {
    const val = Number(size);
    if (!isNaN(val) && val > 0) {
        treePointSizeUser = val;
        if (treeOverlayPoints && treeOverlayPoints.material) {
            treeOverlayPoints.material.size = getTreePointSize();
            treeOverlayPoints.material.needsUpdate = true;
            viewer.render();
        }
    }
};

window.setTreeColorMode = function (mode) {
    treeColorMode = mode || 'auto';
    updateTreeColors();
};

function getClassificationColor(cls) {
    if (window.Potree && Potree.ClassificationScheme && Potree.ClassificationScheme.DEFAULT) {
        const item = Potree.ClassificationScheme.DEFAULT[cls] || Potree.ClassificationScheme.DEFAULT.DEFAULT;
        if (item && item.color) {
            return [item.color[0], item.color[1], item.color[2]];
        }
    }
    switch (Number(cls)) {
        case 2: return [0.48, 0.27, 0.14]; // Ground (rich earthy brown)
        case 3: return [0.0, 1.0, 0.0];    // Low veg (light green)
        case 4: return [0.0, 0.8, 0.0];    // Medium veg (green)
        case 5: return [0.0, 0.5, 0.0];    // High veg (dark forest green)
        case 6: return [1.0, 0.66, 0.0];   // Building (orange)
        case 7: return [1.0, 0.0, 0.0];    // Noise (red)
        case 9: return [0.0, 0.4, 0.9];    // Water (blue)
        default: return [0.5, 0.5, 0.5];
    }
}

function elevationColorLUT(t) {
    const clamped = Math.min(1.0, Math.max(0.0, t));
    const h = (1.0 - clamped) * 0.66;
    let r, g, b;
    let i = Math.floor(h * 6);
    let f = h * 6 - i;
    let p = 0.95 * (1 - 0.9);
    let q = 0.95 * (1 - 0.9 * f);
    let tt = 0.95 * (1 - 0.9 * (1 - f));
    switch (i % 6) {
        case 0: r = 0.95; g = tt; b = p; break;
        case 1: r = q; g = 0.95; b = p; break;
        case 2: r = p; g = 0.95; b = tt; break;
        case 3: r = p; g = q; b = 0.95; break;
        case 4: r = tt; g = p; b = 0.95; break;
        case 5: r = 0.95; g = p; b = q; break;
    }
    return [r, g, b];
}

function hslToRgb(h, s, l) {
    let r, g, b;
    if (s === 0) {
        r = g = b = l;
    } else {
        const hue2rgb = (p, q, t) => {
            let val = t;
            if (val < 0) val += 1;
            if (val > 1) val -= 1;
            if (val < 1/6) return p + (q - p) * 6 * val;
            if (val < 1/2) return q;
            if (val < 2/3) return p + (q - p) * (2/3 - val) * 6;
            return p;
        };
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1/3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1/3);
    }
    return [r, g, b];
}

function getNeighborTreeColor(tid, isGround) {
    if (isGround || tid <= 0) return [0.48, 0.27, 0.14]; // Rich earthy soil brown
    // Golden ratio hue stepping over [45°, 375°] mod 360°, bypassing the brown band [15°-45°]
    const t = ((tid * 0.61803398875) % 1.0 + 1.0) % 1.0;
    const hue = ((45.0 + t * 330.0) % 360.0) / 360.0;
    return hslToRgb(hue, 0.88, 0.54);
}

function updateTreeColors() {
    if (!treeOverlayPoints || !treeRawData) return;
    const activeData = treeActiveFilteredData || treeRawData;
    const { count, targetCount, neighborCount, rgb, intensity, classification, xyz, bboxMin, bboxMax, scalars } = activeData;
    const colorAttr = treeOverlayPoints.geometry.attributes.color;
    if (!colorAttr) return;
    const colors = colorAttr.array;

    let mode = treeColorMode;
    if (!mode || mode === 'auto') {
        mode = viewerSettings.colorMode || 'rgba';
    }

    const normKey = (k) => String(k || '').toLowerCase().replace(/[\s_-]/g, '');
    const nMode = normKey(mode);

    // 1. Tree ID / ITC mode: color all trees (target tree and neighbor trees) by distinct Tree ID
    const isTreeIdMode = ['itc', 'treeid', 'tree_id', 'trees', 'tree'].includes(nMode);
    if (isTreeIdMode) {
        const isSeg = Boolean(activeData.isSegmented || (activeData.ff3dTrees && activeData.ff3dTrees > 0));
        const tCount = isSeg ? 0 : (targetCount || (count - (neighborCount || 0)));
        const tidArr = scalars ? (scalars.itc || scalars.tree_id || scalars.treeID) : null;
        for (let i = 0; i < count; i++) {
            let tid = 0;
            if (isSeg && tidArr) {
                tid = Math.round(tidArr[i]);
            } else if (i < tCount) {
                tid = (tidArr ? Math.round(tidArr[i]) : treeOverlayItc) || 1;
            } else {
                tid = tidArr ? Math.round(tidArr[i]) : 0;
            }
            // Ground determination:
            // Point is ground if ASPRS classification == 2 or tid <= 0 (unassigned / ground from FF3D)
            const isGround = (classification && classification[i] === 2) || (tid <= 0);
            if (!isGround && tid <= 0) {
                // Surrounding unsegmented canopy/vegetation (not ground, but has no discrete tree id):
                // render lush vegetation green so it is never mistaken for brown ground!
                colors[i * 3] = 0.18; colors[i * 3 + 1] = 0.65; colors[i * 3 + 2] = 0.28;
            } else {
                const c = getNeighborTreeColor(tid, isGround);
                colors[i * 3] = c[0]; colors[i * 3 + 1] = c[1]; colors[i * 3 + 2] = c[2];
            }
        }
    } else {
        // 2. Extra scalar field matching (e.g. return number, tree_score, ndvi, gps-time, semantic_pred, etc.)
        let matchedScalar = null;
        let matchedScalarName = null;
        if (scalars) {
            for (const [sname, sarr] of Object.entries(scalars)) {
                if (normKey(sname) === nMode) {
                    matchedScalar = sarr;
                    matchedScalarName = sname;
                    break;
                }
            }
        }

        if (matchedScalar && !['elevation', 'height', 'classification', 'intensity', 'intensitygradient', 'green', 'viridis', 'rgba', 'rgb', 'color'].includes(nMode)) {
            let lo = Infinity, hi = -Infinity;
            let globalAttr = null;
            if (currentPointCloud && currentPointCloud.getAttribute && matchedScalarName) {
                globalAttr = currentPointCloud.getAttribute(matchedScalarName);
            }
            if (globalAttr && Array.isArray(globalAttr.range) && globalAttr.range.length >= 2 && Number.isFinite(globalAttr.range[0]) && Number.isFinite(globalAttr.range[1])) {
                lo = globalAttr.range[0];
                hi = globalAttr.range[1];
            } else {
                for (let i = 0; i < count; i++) {
                    const v = matchedScalar[i];
                    if (v < lo) lo = v;
                    if (v > hi) hi = v;
                }
            }
            const span = Math.max(0.0001, hi - lo);
            for (let i = 0; i < count; i++) {
                const t = Math.min(1.0, Math.max(0.0, (matchedScalar[i] - lo) / span));
                const c = treeColorLUT(t);
                colors[i * 3] = c[0]; colors[i * 3 + 1] = c[1]; colors[i * 3 + 2] = c[2];
            }
        } else if (nMode === 'classification') {
            const clsArr = classification || (matchedScalar && nMode === 'classification' ? matchedScalar : null);
            if (clsArr) {
                for (let i = 0; i < count; i++) {
                    const c = getClassificationColor(clsArr[i]);
                    colors[i * 3] = c[0]; colors[i * 3 + 1] = c[1]; colors[i * 3 + 2] = c[2];
                }
            } else {
                const zSpan = Math.max(0.1, bboxMax[2] - bboxMin[2]);
                for (let i = 0; i < count; i++) {
                    const t = Math.min(1.0, Math.max(0.0, (xyz[i * 3 + 2] - bboxMin[2]) / zSpan));
                    colors[i * 3] = 0.10 + 0.14 * t;
                    colors[i * 3 + 1] = 0.38 + 0.36 * t;
                    colors[i * 3 + 2] = 0.12 + 0.16 * t;
                }
            }
        } else if (nMode === 'intensity' || nMode === 'intensitygradient') {
            const intArr = intensity || matchedScalar;
            if (intArr) {
                let lo = 65535, hi = 0;
                for (let i = 0; i < count; i++) { const v = intArr[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
                const span = Math.max(1, hi - lo);
                for (let i = 0; i < count; i++) {
                    const v = Math.min(1.0, Math.max(0.0, (intArr[i] - lo) / span));
                    colors[i * 3] = v; colors[i * 3 + 1] = v; colors[i * 3 + 2] = v;
                }
            } else {
                const zSpan = Math.max(0.1, bboxMax[2] - bboxMin[2]);
                for (let i = 0; i < count; i++) {
                    const t = Math.min(1.0, Math.max(0.0, (xyz[i * 3 + 2] - bboxMin[2]) / zSpan));
                    const c = elevationColorLUT(t);
                    colors[i * 3] = c[0]; colors[i * 3 + 1] = c[1]; colors[i * 3 + 2] = c[2];
                }
            }
        } else if (nMode === 'elevation' || nMode === 'height') {
            const zSpan = Math.max(0.1, bboxMax[2] - bboxMin[2]);
            for (let i = 0; i < count; i++) {
                const t = Math.min(1.0, Math.max(0.0, (xyz[i * 3 + 2] - bboxMin[2]) / zSpan));
                const c = elevationColorLUT(t);
                colors[i * 3] = c[0]; colors[i * 3 + 1] = c[1]; colors[i * 3 + 2] = c[2];
            }
        } else if (nMode === 'green') {
            const zSpan = Math.max(0.1, bboxMax[2] - bboxMin[2]);
            for (let i = 0; i < count; i++) {
                const t = Math.min(1.0, Math.max(0.0, (xyz[i * 3 + 2] - bboxMin[2]) / zSpan));
                colors[i * 3] = 0.10 + 0.14 * t;
                colors[i * 3 + 1] = 0.38 + 0.36 * t;
                colors[i * 3 + 2] = 0.12 + 0.16 * t;
            }
        } else if (nMode === 'viridis') {
            const zSpan = Math.max(0.1, bboxMax[2] - bboxMin[2]);
            for (let i = 0; i < count; i++) {
                const t = Math.min(1.0, Math.max(0.0, (xyz[i * 3 + 2] - bboxMin[2]) / zSpan));
                const c = treeColorLUT(t);
                colors[i * 3] = c[0]; colors[i * 3 + 1] = c[1]; colors[i * 3 + 2] = c[2];
            }
        } else if (['rgba', 'rgb', 'color'].includes(nMode) && rgb) {
            for (let i = 0; i < count * 3; i++) {
                colors[i] = rgb[i] / 255;
            }
        } else if (rgb) {
            for (let i = 0; i < count * 3; i++) {
                colors[i] = rgb[i] / 255;
            }
        } else if (classification) {
            for (let i = 0; i < count; i++) {
                const c = getClassificationColor(classification[i]);
                colors[i * 3] = c[0]; colors[i * 3 + 1] = c[1]; colors[i * 3 + 2] = c[2];
            }
        } else {
            const zSpan = Math.max(0.1, bboxMax[2] - bboxMin[2]);
            for (let i = 0; i < count; i++) {
                if (classification && classification[i] === 2) {
                    colors[i * 3] = 0.48; colors[i * 3 + 1] = 0.27; colors[i * 3 + 2] = 0.14; // Rich earthy brown ground
                } else {
                    const t = Math.min(1.0, Math.max(0.0, (xyz[i * 3 + 2] - bboxMin[2]) / zSpan));
                    colors[i * 3] = 0.10 + 0.14 * t;
                    colors[i * 3 + 1] = 0.38 + 0.36 * t;
                    colors[i * 3 + 2] = 0.12 + 0.16 * t;
                }
            }
        }
    }

    // In Filter mode, dim trees that are not selected to keep:
    if (treeFilterMode && treeRawData && treeRawData.neighborCount > 0) {
        const tCount = targetCount || (count - (neighborCount || 0));
        const tidArr = scalars ? (scalars.itc || scalars.tree_id || scalars.treeID) : null;
        for (let i = 0; i < count; i++) {
            let tid = 0;
            if (i < tCount) {
                tid = (tidArr ? Math.round(tidArr[i]) : treeOverlayItc) || 1;
            } else {
                tid = tidArr ? Math.round(tidArr[i]) : 0;
            }
            if (!treeFilterSelectedIds.has(tid)) {
                colors[i * 3] = colors[i * 3] * 0.18 + 0.04;
                colors[i * 3 + 1] = colors[i * 3 + 1] * 0.18 + 0.04;
                colors[i * 3 + 2] = colors[i * 3 + 2] * 0.18 + 0.04;
            }
        }
    }

    colorAttr.needsUpdate = true;
    viewer.render();
}

window.showMainCloudContext = function () {
    if (treeOverlayPoints) {
        treeOverlayPoints.visible = false;
    }
    if (pickedPointMarker) pickedPointMarker.visible = true;
    updateStandFilterHud();
    showTreeContext();
    viewer.render();
};

window.showTreeOverlayContext = function () {
    if (inspectCard) inspectCard.style.display = 'none';
    if (pickedPointMarker) pickedPointMarker.visible = false;

    // The Forestry tab owns the individual-tree overlay; the full-density main
    // cloud belongs to the Layers tab. Hide it whether or not a tree is loaded,
    // so closing a tree view leaves the Forestry tab empty instead of showing
    // the whole point cloud again.
    hideTreeContext();
    if (treeOverlayPoints) {
        treeOverlayPoints.visible = true;
        updateTreeColors();
    }
    updateStandFilterHud();
    viewer.render();
};

window.closeTreeOverlay = function () {
    if (treeOverlayPoints) {
        if (viewer.scene && viewer.scene.scene) {
            viewer.scene.scene.remove(treeOverlayPoints);
        }
        treeOverlayPoints.geometry.dispose();
        treeOverlayPoints.material.dispose();
        treeOverlayPoints = null;
        treeOverlayItc = null;
        treeRawData = null;
        treeActiveFilteredData = null;
        treeFilterActive = false;
        treeFilterMode = false;
        treeFilterSelectedIds.clear();
        treeOverlayTargetBBox = null;
        showTreeBBox();
        clearMeasure();
    }
    updateStandFilterHud();
    if (inspectCard) inspectCard.style.display = 'none';
    clearPickedPointHighlight();
    showTreeContext();
    showToast('Tree view closed — context layers restored');
    viewer.render();
};

window.hideTreeOverlay = window.showMainCloudContext;

const treeBufferCache = new Map();

async function prefetchTree(itcValue, hintX, hintY) {
    const num = Number(itcValue);
    if (!num || num <= 0 || treeBufferCache.has(num)) return;
    try {
        const payload = { itc: num };
        if (hintX !== undefined && hintX !== null) payload.x = Number(hintX);
        if (hintY !== undefined && hintY !== null) payload.y = Number(hintY);
        const res = await fetch('/api/get_tree', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (res.ok) {
            const buf = await res.arrayBuffer();
            treeBufferCache.set(num, buf);
            if (treeBufferCache.size > 80) {
                const firstKey = treeBufferCache.keys().next().value;
                treeBufferCache.delete(firstKey);
            }
        }
    } catch (_) {}
}
window.prefetchTree = prefetchTree;

window.invalidateTreeCache = function (itcValue) {
    if (itcValue === undefined || itcValue === null) {
        treeBufferCache.clear();
        console.log('[PotreeHybrid] Cleared all treeBufferCache entries');
        return;
    }
    const num = Number(itcValue);
    const prefix = `${num}_`;
    for (const key of Array.from(treeBufferCache.keys())) {
        if (key === num || (typeof key === 'string' && key.startsWith(prefix))) {
            treeBufferCache.delete(key);
        }
    }
    console.log(`[PotreeHybrid] Invalidated treeBufferCache for Tree #${num}`);
};

window.viewTreeById = async function (itcValue, hintX, hintY, radius, neighborMode, groundRadius, endpoint, keepTreeIds) {
    if (itcValue === null || itcValue === undefined) return;
    const num = Number(itcValue);
    if (!num || num <= 0) return;

    if (inspectCard) inspectCard.style.display = 'none';
    if (pickedPointMarker) pickedPointMarker.visible = false;

    const radVal = Number(radius) || 0.0;
    const nMode = neighborMode || 'trees_only';
    const gRad = Number(groundRadius) || 0.0;
    const ep = endpoint || '/api/get_tree';
    const seg = ep.indexOf('segment') !== -1;

    let activeKeep = keepTreeIds;
    if (!activeKeep && seg && treeFilterActive && treeFilterSelectedIds && treeFilterSelectedIds.size > 0) {
        activeKeep = Array.from(treeFilterSelectedIds);
    }
    const keepStr = (activeKeep && activeKeep.length > 0) ? activeKeep.slice().sort().join(',') : 'all';
    const cacheKey = `${num}_${radVal}_${nMode}_${gRad}_${seg ? 'seg' : 'itc'}_${keepStr}`;

    try {
        let buf = treeBufferCache.get(cacheKey);
        if (!buf) {
            const label = radVal > 0 ? `Tree #${num} + neighbors (${radVal}m)` : `Tree #${num}`;
            if (seg) {
                showToast('ForestFormer3D: segmenting cloud (first run loads the model, ~15 s)...', 8000);
            } else if (isIndexingActive) {
                showToast(`Fetching ${label} via streaming fallback (background indexing in progress)...`, 3500);
            } else {
                showToast(`Fetching ${label} — querying full density...`);
            }
            const payload = { itc: num, radius: radVal, neighbor_mode: nMode };
            if (activeKeep && activeKeep.length > 0) {
                payload.keep_tree_ids = activeKeep;
            }
            if (gRad > 0) payload.ground_radius = gRad;
            if (hintX !== undefined && hintX !== null) payload.x = Number(hintX);
            if (hintY !== undefined && hintY !== null) payload.y = Number(hintY);

            const res = await fetch(ep, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!res.ok) throw new Error(`${ep} -> HTTP ${res.status}`);
            buf = await res.arrayBuffer();
            treeBufferCache.set(cacheKey, buf);
        }

        const dv = new DataView(buf);
        const count = dv.getUint32(8, true);
        const flags = dv.getUint32(12, true);
        const hasRgb = (flags & 1) !== 0;
        const hasIntensity = (flags & 2) !== 0;
        const hasClassification = (flags & 4) !== 0;
        const hasScalars = (flags & 8) !== 0;
        const cx = dv.getFloat64(16, true), cy = dv.getFloat64(24, true), cz = dv.getFloat64(32, true);
        const bboxMin = [dv.getFloat32(40, true), dv.getFloat32(44, true), dv.getFloat32(48, true)];
        const bboxMax = [dv.getFloat32(52, true), dv.getFloat32(56, true), dv.getFloat32(60, true)];

        if (count === 0) {
            showToast(`Tree #${itcValue}: no points found in master file`);
            return;
        }

        let off = 64;
        let scalarMeta = [];
        let targetCount = count;
        let neighborCount = 0;
        let standRadius = radVal;
        let ff3dTrees = 0;
        let ff3dLas = null;
        if (hasScalars) {
            const metaLen = dv.getUint32(off, true);
            off += 4;
            const metaBytes = new Uint8Array(buf, off, metaLen);
            const metaStr = new TextDecoder().decode(metaBytes);
            try {
                const parsed = JSON.parse(metaStr);
                scalarMeta = parsed.scalars || [];
                if (parsed.target_count !== undefined) targetCount = parsed.target_count;
                if (parsed.neighbor_count !== undefined) neighborCount = parsed.neighbor_count;
                if (parsed.radius !== undefined) standRadius = parsed.radius;
                if (parsed.ff3d_trees !== undefined) ff3dTrees = parsed.ff3d_trees;
                if (parsed.ff3d_las) ff3dLas = parsed.ff3d_las;
            } catch (e) {
                console.warn('Failed to parse tree scalar metadata:', e);
            }
            off += metaLen;
            const padMeta = (4 - ((4 + metaLen) % 4)) % 4;
            if (padMeta > 0) off += padMeta;
        }

        const xyz = new Float32Array(buf, off, count * 3);
        off += count * 12;

        let rgb = null, intensity = null, classification = null;
        if (hasRgb) {
            rgb = new Uint8Array(buf, off, count * 3);
            off += count * 3;
            if (off % 2 !== 0) off += 1; // alignment padding
        }
        if (hasIntensity) {
            if (off % 2 !== 0) {
                intensity = new Uint16Array(buf.slice(off, off + count * 2));
            } else {
                intensity = new Uint16Array(buf, off, count);
            }
            off += count * 2;
        }
        let totalBytes = 0;
        if (hasClassification) {
            classification = new Uint8Array(buf, off, count);
            off += count;
            totalBytes += count;
        }

        const scalars = {};
        if (hasScalars && scalarMeta.length > 0) {
            const padScalars = (4 - (totalBytes % 4)) % 4;
            if (padScalars > 0) off += padScalars;
            for (const sname of scalarMeta) {
                let arr;
                if (off % 4 !== 0) {
                    arr = new Float32Array(buf.slice(off, off + count * 4));
                } else {
                    arr = new Float32Array(buf, off, count);
                }
                scalars[sname] = arr;
                off += count * 4;
            }
        }

        treeRawData = {
            count,
            targetCount,
            neighborCount,
            radius: standRadius,
            xyz,
            rgb,
            intensity,
            classification,
            scalars,
            cx, cy, cz,
            bboxMin, bboxMax,
            ff3dTrees,
            ff3dLas,
            isSegmented: seg
        };

        // Reset previous stand filter state immediately so the new cloud
        // renders cleanly with its updated points, scalars, and colors!
        treeFilterActive = false;
        treeFilterMode = false;
        treeActiveFilteredData = null;
        treeFilterSelectedIds = new Set();

        // Hide octree context now that valid tree points have arrived
        hideTreeContext();

        if (treeOverlayPoints) {
            if (viewer.scene && viewer.scene.scene) {
                viewer.scene.scene.remove(treeOverlayPoints);
            }
            treeOverlayPoints.geometry.dispose();
            treeOverlayPoints.material.dispose();
            treeOverlayPoints = null;
        }

        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
        geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(count * 3), 3));

        const posAttr = geom.attributes.position.array;
        for (let i = 0; i < count * 3; i++) posAttr[i] = xyz[i];
        geom.computeBoundingSphere();
        geom.computeBoundingBox();

        // Points material styled like Potree point clouds:
        // - sizeAttenuation: true for 3D perspective
        // - depthTest: true and depthWrite: true so foliage points properly occlude each other in 3D
        // - onBeforeCompile shader discards fragments outside the circle and adds 3D spherical shading (EDL look)
        const ptSize = getTreePointSize();

        const mat = new THREE.PointsMaterial({
            size: ptSize,
            vertexColors: true,
            sizeAttenuation: true,
            depthTest: true,
            depthWrite: true,
            transparent: false
        });

        mat.onBeforeCompile = (shader) => {
            shader.fragmentShader = shader.fragmentShader.replace(
                '#include <color_fragment>',
                `#include <color_fragment>
                 vec2 coord = gl_PointCoord - vec2(0.5);
                 float distSq = dot(coord, coord);
                 if (distSq > 0.25) discard;
                 float nz = sqrt(max(0.0, 0.25 - distSq)) * 2.0;
                 diffuseColor.rgb *= clamp(0.55 + 0.45 * nz, 0.0, 1.0);
                `
            );
        };

        const points = new THREE.Points(geom, mat);
        points.position.set(cx, cy, cz);   // geometry is centroid-relative
        points.name = `tree_overlay_${itcValue}`;
        points.renderOrder = 10;
        points.frustumCulled = false;
        viewer.scene.scene.add(points);
        const isNewTree = treeOverlayItc !== itcValue;
        treeOverlayPoints = points;
        treeOverlayItc = itcValue;

        if (seg) {
            treeColorMode = 'itc';
        }

        updateTreeColors();

        const standTrees = getStandTreeList();
        updateStandFilterHud();

        // Target-tree bbox (world): powers the bounding box + height label,
        // and the fixed eye-level pose on tree change.
        {
            let fx0 = Infinity, fy0 = Infinity, fz0 = Infinity;
            let fx1 = -Infinity, fy1 = -Infinity, fz1 = -Infinity;
            const nT = (targetCount > 0 && !seg) ? Math.min(targetCount, count) : count;
            for (let i = 0; i < nT * 3; i += 3) {
                if (xyz[i] < fx0) fx0 = xyz[i];       if (xyz[i] > fx1) fx1 = xyz[i];
                if (xyz[i+1] < fy0) fy0 = xyz[i+1];   if (xyz[i+1] > fy1) fy1 = xyz[i+1];
                if (xyz[i+2] < fz0) fz0 = xyz[i+2];   if (xyz[i+2] > fz1) fz1 = xyz[i+2];
            }
            treeOverlayTargetBBox = {
                min: [cx + fx0, cy + fy0, cz + fz0],
                max: [cx + fx1, cy + fy1, cz + fz1]
            };

            if (isNewTree || seg) {
                const center = new THREE.Vector3(
                    cx + (fx0 + fx1) / 2,
                    cy + (fy0 + fy1) / 2,
                    cz + (fz0 + fz1) / 2);
                const radius = Math.max(fx1 - fx0, fy1 - fy0, fz1 - fz0, 4) * 0.5;
                const fov = (viewer.getFOV ? viewer.getFOV() : 60) * Math.PI / 180;
                const dist = (radius / Math.tan(fov * 0.5)) * 1.3;
                viewer.scene.view.position.set(
                    center.x,
                    center.y - dist,
                    center.z + dist * 0.35);
                viewer.scene.view.lookAt(center);
            }
        }
        showTreeBBox();

        viewer.render();
        const savedNote = ff3dLas ? ` · saved ${ff3dLas.split(/[\\/]/).pop()}` : '';
        if (ff3dTrees > 1) {
            showToast(`ForestFormer3D: ${ff3dTrees} trees found — colours show the split. Inspect, then Merge.${savedNote}`, 7000);
        } else if (ff3dTrees === 1) {
            showToast(`ForestFormer3D: 1 tree — the cloud is a single tree (no split).${savedNote}`, 6000);
        } else if (isIndexingActive) {
            showToast(`Tree #${itcValue} stand rendered via streaming fallback (background indexing continues)...`, 4000);
        } else if (neighborCount > 0) {
            showToast(`Tree #${itcValue} (${targetCount.toLocaleString()} pts) + ${neighborCount.toLocaleString()} neighbor pts (${standRadius}m stand)`);
        } else {
            showToast(`Tree #${itcValue} — ${count.toLocaleString()} pts (full density)`);
        }

        if (seg) {
            postJSON('/api/event', {
                event: 'segmentation_step',
                step: 5, total: 5, stage: 'complete',
                category: 'COMPLETE',
                msg: `3D viewport rendered with ${ff3dTrees} tree instances (${count.toLocaleString()} points). Ready to confirm & split!`,
                itc: num, trees: ff3dTrees
            }).catch(() => {});
        }

        postJSON('/api/event', {
            event: 'tree_loaded',
            itc: itcValue,
            count: count,
            target_count: targetCount,
            neighbor_count: neighborCount,
            radius: standRadius,
            trees: standTrees,
            ff3d_trees: ff3dTrees,
            ff3d_las: ff3dLas,
            is_segmented: seg,
            keep_tree_ids: activeKeep || null
        }).catch(() => {});

        if (neighborCount > 0 || (seg && ff3dTrees > 0)) {
            postJSON('/api/event', {
                event: 'tree_stand_loaded',
                target_itc: itcValue,
                count: count,
                target_count: targetCount,
                neighbor_count: neighborCount,
                radius: standRadius,
                trees: standTrees
            }).catch(() => {});
        }
    } catch (e) {
        console.error('Tree view failed:', e);
        showToast('Tree view failed: ' + e.message);
        showTreeContext();
    }
};

// Segment the displayed tree/stand with ForestFormer3D: the overlay rebuilds
// coloured per detected instance (green = tree 1, 48-stop palette per tree).
window.segmentTreeById = function (itcValue, hintX, hintY, radius, neighborMode, groundRadius, keepTreeIds) {
    let keep = keepTreeIds;
    if (!keep && treeFilterActive && treeFilterSelectedIds && treeFilterSelectedIds.size > 0) {
        keep = Array.from(treeFilterSelectedIds);
    }
    return window.viewTreeById(itcValue, hintX, hintY, radius, neighborMode, groundRadius,
                               '/api/segment_tree', keep);
};

window.currentTreeViewItc = function () {
    return treeOverlayItc;
};
