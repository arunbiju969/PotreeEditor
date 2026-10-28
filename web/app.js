import * as THREE from '/libs/three.js/build/three.module.js';

// Global state
let currentPointCloud = null;
let activeTool = 'nav'; // 'nav', 'polygon', 'box', 'inspect'
let activeSelectionMesh = null;
let currentSelectionStats = null;

// Initialize Potree Viewer
const renderArea = document.getElementById('potree_render_area');
const viewer = new Potree.Viewer(renderArea);
window.viewer = viewer;

viewer.setEDLEnabled(true);
viewer.setFOV(60);
viewer.setPointBudget(2_500_000);
viewer.setMinNodeSize(10);
viewer.setBackground('black');

// UI Elements
const btnNav = document.getElementById('btnNav');
const btnPolygon = document.getElementById('btnPolygon');
const btnBox = document.getElementById('btnBox');
const btnInspect = document.getElementById('btnInspect');
const btnClear = document.getElementById('btnClear');
const toastMsg = document.getElementById('toastMsg');
const selectionBanner = document.getElementById('selectionBanner');
const selectionSummary = document.getElementById('selectionSummary');
const btnQuickNoise = document.getElementById('btnQuickNoise');
const inspectCard = document.getElementById('inspectCard');
const closeInspect = document.getElementById('closeInspect');

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

    // Remove any in-progress measurement insertion
    if (viewer.measuringTool) {
        // cancels pending tool
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

        // Material configuration
        pc.material.size = 1;
        pc.material.pointSizeType = Potree.PointSizeType.ADAPTIVE;

        const hasRGBA = pc.getAttributes().attributes.some(a => a.name === 'rgba');
        if (hasRGBA) {
            pc.material.activeAttributeName = 'rgba';
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
// Polygon Selection Tool
// ---------------------------------------------------------------------------
let polygonPoints = [];
let polygonLine = null;

function startPolygonSelection() {
    clearSelection(false);
    polygonPoints = [];
    showToast('Polygon: Left-click points to trace boundary. Double-click or Right-click to finish.');

    // Temporary line object for visual feedback
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

    // Close polygon visually
    const flatPoints = [];
    for (const pt of polygonPoints) {
        flatPoints.push(pt.x, pt.y, pt.z);
    }
    flatPoints.push(polygonPoints[0].x, polygonPoints[0].y, polygonPoints[0].z);

    const positions = new Float32Array(flatPoints);
    polygonLine.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    polygonLine.geometry.attributes.position.needsUpdate = true;
    activeSelectionMesh = polygonLine;

    // Send 2D coordinates [[x, y], ...] to Python server
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

    // Raycast/pick using Potree's GPU picking
    const pickResult = viewer.scene.pick(viewer, x, y);
    if (!pickResult || !pickResult.position) {
        return;
    }

    const pos = pickResult.position;
    document.getElementById('inspX').textContent = pos.x.toFixed(3);
    document.getElementById('inspY').textContent = pos.y.toFixed(3);
    document.getElementById('inspZ').textContent = pos.z.toFixed(3);

    // Query Python for full LAS attributes (ITC, raw classification, intensity)
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

            // Update line geometry
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
// External API for Python/Qt
// ---------------------------------------------------------------------------
window.setPointBudget = (budget) => viewer.setPointBudget(Number(budget));
window.setPointSize = (size) => {
    if (currentPointCloud) currentPointCloud.material.size = Number(size);
};
window.setEDLEnabled = (enabled) => viewer.setEDLEnabled(Boolean(enabled));
window.setColorAttribute = (attr) => {
    if (currentPointCloud) currentPointCloud.material.activeAttributeName = attr;
};
window.clearSelection = clearSelection;
