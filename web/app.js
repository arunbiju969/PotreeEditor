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
        attributes: pc.getAttributes ? pc.getAttributes().attributes.map(a => a.name) : []
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
    if (pc && pc.material) pc.material.activeAttributeName = attr;
    if (pc && pc === currentPointCloud) viewerSettings.colorMode = attr;
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

viewer.addEventListener('render.pass.perspective_overlay', () => {
    const camera = viewer.scene.getActiveCamera();
    if (camera && overlayScene.children.length > 0) {
        viewer.renderer.render(overlayScene, camera);
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

function showToast(text, duration = 3000) {
    if (!toastMsg) return;
    toastMsg.textContent = text;
    toastMsg.style.display = 'block';
    setTimeout(() => {
        toastMsg.style.display = 'none';
    }, duration);
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
        showToast('Pick Point Mode: Click any point in the cloud to inspect and highlight');
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
            pc.material.activeAttributeName = 'rgba';
        } else if (viewerSettings.colorMode) {
            pc.material.activeAttributeName = viewerSettings.colorMode;
        } else {
            pc.material.activeAttributeName = 'color';
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

    fetch('/api/select_polygon', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ coordinates: coords2d, points_3d: coords3d })
    })
    .then(r => r.json())
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
        volume.addEventListener('insertion_finished', () => {
            const min = volume.boundingBox.min;
            const max = volume.boundingBox.max;
            activeSelectionMesh = volume;

            fetch('/api/select_box', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    min: [min.x, min.y, min.z],
                    max: [max.x, max.y, max.z]
                })
            })
            .then(r => r.json())
            .then(data => handleSelectionResult(data))
            .catch(err => console.error(err));

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
            { pickClipped: true }
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
async function pickPointAt(mouseEvent) {
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

    // Extract attributes from Potree GPU pick if immediately available
    const initialClass = extractAttr(pickResult.point, 'classification');
    const initialInt = extractAttr(pickResult.point, 'intensity');
    const initialITC = extractAttr(pickResult.point, 'itc', 'tree_id', 'treeID');

    if (inspClass) inspClass.textContent = formatClassName(initialClass);
    if (inspInt) inspInt.textContent = initialInt !== null ? initialInt : '-';
    if (inspITC) inspITC.textContent = initialITC !== null ? initialITC : '-';
    if (inspectCard) inspectCard.style.display = 'block';

    try {
        const res = await fetch('/api/pick_point', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x: pos.x, y: pos.y, z: pos.z })
        });
        const data = await res.json();
        const picked = data.picked;

        if (picked) {
            const finalClass = picked.classification ?? initialClass;
            const finalInt = picked.intensity ?? initialInt;
            if (inspClass) inspClass.textContent = formatClassName(finalClass);
            if (inspInt) inspInt.textContent = finalInt ?? '-';

            if (picked.staged_hit) {
                if (inspITC) inspITC.textContent = `${picked.staged_hit.label} (Staged)`;
                updatePickedPointLabel(picked.staged_hit.label, picked.staged_hit.color || '#10b981');
            } else if (picked.itc !== undefined && picked.itc !== null) {
                if (inspITC) inspITC.textContent = picked.itc !== 0 ? picked.itc : 'None';
                if (picked.itc !== 0) {
                    updatePickedPointLabel(`Tree #${picked.itc}`, '#10b981');
                } else if (finalClass !== null && finalClass !== undefined) {
                    updatePickedPointLabel(formatClassName(finalClass), '#00f5ff');
                }
            } else if (finalClass !== null && finalClass !== undefined) {
                updatePickedPointLabel(formatClassName(finalClass), '#00f5ff');
            }
        } else if (initialClass !== null) {
            updatePickedPointLabel(formatClassName(initialClass), '#00f5ff');
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
        fetch('/api/select_polygon', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ coordinates: [], points_3d: [] })
        });
        showToast('Selection cleared.');
    }
}

// ---------------------------------------------------------------------------
// Mouse & Click Event Listeners
// ---------------------------------------------------------------------------
renderArea.addEventListener('click', (e) => {
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
    } else if (activeTool === 'inspect') {
        pickPointAt(e);
    }
});

renderArea.addEventListener('dblclick', (e) => {
    if (activeTool === 'polygon') {
        e.preventDefault();
        finishPolygonSelection();
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

        fetch('/api/stage_edit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(edit)
        })
        .then(r => r.json())
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

        fetch('/api/stage_edit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(edit)
        })
        .then(r => r.json())
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
               splatQuality: 'hq', pointBudget: 15_000_000, minNodeSize: 5 }
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
    if (currentPointCloud) currentPointCloud.material.activeAttributeName = attr;
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
