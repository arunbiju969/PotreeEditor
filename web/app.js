import * as THREE from '/libs/three.js/build/three.module.js';

// Global state
let currentPointCloud = null;
let activeTool = 'nav'; // 'nav', 'polygon', 'box', 'inspect'
let activeSelectionMesh = null;
let currentSelectionStats = null;

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

viewer.setEDLEnabled(viewerSettings.edlEnabled);
viewer.setEDLRadius(viewerSettings.edlRadius);
viewer.setEDLStrength(viewerSettings.edlStrength);
viewer.setEDLOpacity(viewerSettings.edlOpacity);
viewer.setFOV(60);
viewer.setPointBudget(viewerSettings.pointBudget);
viewer.setMinNodeSize(viewerSettings.minNodeSize);
viewer.setBackground(viewerSettings.background);

// UI Elements - Top HUD
const btnNav = document.getElementById('btnNav');
const btnPolygon = document.getElementById('btnPolygon');
const btnBox = document.getElementById('btnBox');
const btnInspect = document.getElementById('btnInspect');
const btnSettings = document.getElementById('btnSettings');
const btnClear = document.getElementById('btnClear');
const toastMsg = document.getElementById('toastMsg');
const selectionBanner = document.getElementById('selectionBanner');
const selectionSummary = document.getElementById('selectionSummary');
const btnQuickNoise = document.getElementById('btnQuickNoise');
const inspectCard = document.getElementById('inspectCard');
const closeInspect = document.getElementById('closeInspect');

// UI Elements - Settings Card
const settingsCard = document.getElementById('settingsCard');
const closeSettings = document.getElementById('closeSettings');
const cfgPointType = document.getElementById('cfgPointType');
const cfgPointSizeType = document.getElementById('cfgPointSizeType');
const sldPointSize = document.getElementById('sldPointSize');
const lblPointSize = document.getElementById('lblPointSize');
const cfgSplatQuality = document.getElementById('cfgSplatQuality');
const cfgPointBudget = document.getElementById('cfgPointBudget');
const sldMinNodeSize = document.getElementById('sldMinNodeSize');
const lblMinNodeSize = document.getElementById('lblMinNodeSize');
const chkEDLEnabled = document.getElementById('chkEDLEnabled');
const edlControlsContainer = document.getElementById('edlControlsContainer');
const sldEDLRadius = document.getElementById('sldEDLRadius');
const lblEDLRadius = document.getElementById('lblEDLRadius');
const sldEDLStrength = document.getElementById('sldEDLStrength');
const lblEDLStrength = document.getElementById('lblEDLStrength');
const sldEDLOpacity = document.getElementById('sldEDLOpacity');
const lblEDLOpacity = document.getElementById('lblEDLOpacity');
const cfgBackground = document.getElementById('cfgBackground');

function showToast(text, duration = 3000) {
    toastMsg.textContent = text;
    toastMsg.style.display = 'block';
    setTimeout(() => {
        toastMsg.style.display = 'none';
    }, duration);
}

function setActiveTool(tool) {
    activeTool = tool;
    [btnNav, btnPolygon, btnBox, btnInspect].forEach(b => b.classList.remove('active'));

    if (tool === 'nav') btnNav.classList.add('active');
    else if (tool === 'polygon') btnPolygon.classList.add('active');
    else if (tool === 'box') btnBox.classList.add('active');
    else if (tool === 'inspect') btnInspect.classList.add('active');
}

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

function applyMaterialSettings(pc = currentPointCloud) {
    if (!pc || !pc.material) return;
    const mat = pc.material;
    mat.shape = getPointShapeEnum(viewerSettings.pointType);
    mat.pointSizeType = getPointSizeTypeEnum(viewerSettings.pointSizeType);
    mat.size = Number(viewerSettings.pointSize);
}

function updateAllPointClouds() {
    if (viewer.scene && viewer.scene.pointclouds) {
        viewer.scene.pointclouds.forEach(applyMaterialSettings);
    }
}

// ---------------------------------------------------------------------------
// Point Cloud Loading
// ---------------------------------------------------------------------------
window.loadPointCloudUrl = async function(url) {
    console.log('[PotreeHybrid] Loading point cloud:', url);
    showToast('Loading point cloud: ' + url.split('/').pop());

    try {
        const e = await Potree.loadPointCloud(url);
        const pc = e.pointcloud;

        // If a cloud is already loaded, remove it
        if (currentPointCloud) {
            viewer.scene.scenePointCloud.remove(currentPointCloud);
        }

        currentPointCloud = pc;

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
// Settings Menu Handlers & Bindings
// ---------------------------------------------------------------------------
btnSettings.addEventListener('click', () => {
    const isShowing = settingsCard.style.display === 'block';
    settingsCard.style.display = isShowing ? 'none' : 'block';
    if (!isShowing) btnSettings.classList.add('active');
    else btnSettings.classList.remove('active');
});

closeSettings.addEventListener('click', () => {
    settingsCard.style.display = 'none';
    btnSettings.classList.remove('active');
});

// 1. Point Type (Shape)
cfgPointType.addEventListener('change', (e) => {
    window.setPointShape(e.target.value);
});

// 2. Point Sizing Type
cfgPointSizeType.addEventListener('change', (e) => {
    window.setPointSizeType(e.target.value);
});

// 3. Point Size
sldPointSize.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    lblPointSize.textContent = val.toFixed(2);
    window.setPointSize(val);
});

// 4. Splat Quality
cfgSplatQuality.addEventListener('change', (e) => {
    window.setSplatQuality(e.target.value);
});

// 5. Point Budget
cfgPointBudget.addEventListener('change', (e) => {
    window.setPointBudget(parseInt(e.target.value, 10));
});

// 6. Min Node Size
sldMinNodeSize.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    lblMinNodeSize.textContent = val + ' px';
    viewerSettings.minNodeSize = val;
    viewer.setMinNodeSize(val);
});

// 7. Eye-Dome Lighting (EDL)
chkEDLEnabled.addEventListener('change', (e) => {
    window.setEDLEnabled(e.target.checked);
});

sldEDLRadius.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    lblEDLRadius.textContent = val.toFixed(1);
    window.setEDLRadius(val);
});

sldEDLStrength.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    lblEDLStrength.textContent = val.toFixed(1);
    window.setEDLStrength(val);
});

sldEDLOpacity.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    lblEDLOpacity.textContent = val.toFixed(2);
    window.setEDLOpacity(val);
});

// 8. Background
cfgBackground.addEventListener('change', (e) => {
    viewerSettings.background = e.target.value;
    viewer.setBackground(e.target.value);
});

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
    viewer.scene.scene.add(polygonLine);
}

function finishPolygonSelection() {
    if (polygonPoints.length < 3) {
        showToast('Polygon cancelled: requires at least 3 points.');
        clearSelection();
        setActiveTool('nav');
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
    showToast(`Querying ${coords2d.length}-point polygon in Python backend...`);

    fetch('/api/select_polygon', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ coordinates: coords2d })
    })
    .then(r => r.json())
    .then(data => {
        handleSelectionResult(data);
    })
    .catch(err => {
        console.error('Polygon query failed:', err);
        showToast('Query error: ' + err.message);
    });

    setActiveTool('nav');
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

            setActiveTool('nav');
        });
    }
}

// ---------------------------------------------------------------------------
// Inspection / Pick Point Tool
// ---------------------------------------------------------------------------
async function pickPointAt(mouseEvent) {
    const rect = renderArea.getBoundingClientRect();
    const x = mouseEvent.clientX - rect.left;
    const y = mouseEvent.clientY - rect.top;

    const pickResult = viewer.scene.pick(viewer, x, y);
    if (!pickResult || !pickResult.position) {
        return;
    }

    const pos = pickResult.position;
    document.getElementById('inspX').textContent = pos.x.toFixed(3);
    document.getElementById('inspY').textContent = pos.y.toFixed(3);
    document.getElementById('inspZ').textContent = pos.z.toFixed(3);

    try {
        const res = await fetch('/api/pick_point', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x: pos.x, y: pos.y, z: pos.z })
        });
        const data = await res.json();
        const picked = data.picked;

        if (picked) {
            document.getElementById('inspClass').textContent = picked.classification ?? '-';
            document.getElementById('inspInt').textContent = picked.intensity ?? '-';
            document.getElementById('inspITC').textContent = picked.itc !== null ? picked.itc : 'None';
        } else {
            document.getElementById('inspClass').textContent = pickResult.point?.classification ?? '-';
            document.getElementById('inspInt').textContent = pickResult.point?.intensity ?? '-';
            document.getElementById('inspITC').textContent = '-';
        }
        inspectCard.style.display = 'block';
    } catch (e) {
        inspectCard.style.display = 'block';
    }
}

// ---------------------------------------------------------------------------
// Selection Result Handler
// ---------------------------------------------------------------------------
function handleSelectionResult(data) {
    currentSelectionStats = data;
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
    if (polygonLine) {
        viewer.scene.scene.remove(polygonLine);
        polygonLine = null;
    }
    if (activeSelectionMesh) {
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
    selectionBanner.style.display = 'none';

    if (notifyServer) {
        fetch('/api/select_polygon', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ coordinates: [] })
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
        const pick = viewer.scene.pick(viewer, e.clientX - rect.left, e.clientY - rect.top);
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

// UI Event Handlers
btnNav.addEventListener('click', () => setActiveTool('nav'));
btnPolygon.addEventListener('click', () => {
    setActiveTool('polygon');
    startPolygonSelection();
});
btnBox.addEventListener('click', () => {
    setActiveTool('box');
    startBoxSelection();
});
btnInspect.addEventListener('click', () => setActiveTool('inspect'));
btnClear.addEventListener('click', () => clearSelection());
closeInspect.addEventListener('click', () => {
    inspectCard.style.display = 'none';
});

btnQuickNoise.addEventListener('click', () => {
    fetch('/api/edit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'mark_noise', value: 9999 })
    })
    .then(r => r.json())
    .then(data => {
        showToast(`Marked ${data.modified_count.toLocaleString()} points as Noise!`);
    });
});

// ---------------------------------------------------------------------------
// Public API for Python/Qt & Web Bridge
// ---------------------------------------------------------------------------
window.setPointShape = (shape) => {
    viewerSettings.pointType = shape;
    cfgPointType.value = shape;
    updateAllPointClouds();
    showToast(`Point Type: ${shape}`);
};

window.setPointSizeType = (type) => {
    viewerSettings.pointSizeType = type;
    cfgPointSizeType.value = type;
    updateAllPointClouds();
    showToast(`Point Sizing: ${type}`);
};

window.setPointSize = (size) => {
    viewerSettings.pointSize = Number(size);
    sldPointSize.value = size;
    lblPointSize.textContent = Number(size).toFixed(2);
    updateAllPointClouds();
};

window.setSplatQuality = (quality) => {
    viewerSettings.splatQuality = quality;
    cfgSplatQuality.value = quality;
    const isHQ = (quality === 'hq' || quality === true);
    viewer.useHQ = isHQ;
    showToast(`Splat Quality: ${isHQ ? 'High Quality (HQ)' : 'Standard'}`);
};

window.setPointBudget = (budget) => {
    viewerSettings.pointBudget = Number(budget);
    cfgPointBudget.value = String(budget);
    viewer.setPointBudget(Number(budget));
    showToast(`Point Budget: ${(budget / 1_000_000).toFixed(1)}M points`);
};

window.setEDLEnabled = (enabled) => {
    const isEnabled = Boolean(enabled);
    viewerSettings.edlEnabled = isEnabled;
    chkEDLEnabled.checked = isEnabled;
    edlControlsContainer.style.opacity = isEnabled ? '1' : '0.4';
    viewer.setEDLEnabled(isEnabled);
    showToast(`EDL: ${isEnabled ? 'Enabled' : 'Disabled'}`);
};

window.setEDLRadius = (radius) => {
    const val = Number(radius);
    viewerSettings.edlRadius = val;
    sldEDLRadius.value = val;
    lblEDLRadius.textContent = val.toFixed(1);
    viewer.setEDLRadius(val);
};

window.setEDLStrength = (strength) => {
    const val = Number(strength);
    viewerSettings.edlStrength = val;
    sldEDLStrength.value = val;
    lblEDLStrength.textContent = val.toFixed(1);
    viewer.setEDLStrength(val);
};

window.setEDLOpacity = (opacity) => {
    const val = Number(opacity);
    viewerSettings.edlOpacity = val;
    sldEDLOpacity.value = val;
    lblEDLOpacity.textContent = val.toFixed(2);
    viewer.setEDLOpacity(val);
};

window.setColorAttribute = (attr) => {
    viewerSettings.colorMode = attr;
    if (currentPointCloud) currentPointCloud.material.activeAttributeName = attr;
};

window.toggleSettingsMenu = (show) => {
    if (show === undefined) {
        const isShowing = settingsCard.style.display === 'block';
        settingsCard.style.display = isShowing ? 'none' : 'block';
        if (!isShowing) btnSettings.classList.add('active');
        else btnSettings.classList.remove('active');
    } else {
        settingsCard.style.display = show ? 'block' : 'none';
        if (show) btnSettings.classList.add('active');
        else btnSettings.classList.remove('active');
    }
};

window.clearSelection = clearSelection;
