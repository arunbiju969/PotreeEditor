// Headless check of the Forestry/Layers viewport context switch in web/app.js.
// Extracts the real showTreeOverlayContext / showMainCloudContext /
// hideTreeContext / showTreeContext + the treeContextHidden state they share,
// runs them against stubbed Potree objects, and asserts:
//   1. switching to Forestry with NO tree overlay hides the main cloud (empty viewport)
//   2. switching back to Layers restores it
//   3. Forestry with a tree overlay keeps the overlay visible
import fs from "node:fs";
import vm from "node:vm";

const src = fs.readFileSync(new URL("./web/app.js", import.meta.url), "utf8");
const grab = (re) => {
    const m = src.match(re);
    if (!m) throw new Error(`pattern not found: ${re}`);
    return m[0];
};

const code = [
    grab(/let treeContextHidden = \[\];/),
    grab(/function hideTreeContext\(\) \{[\s\S]*?\n\}/),
    grab(/function showTreeContext\(\) \{[\s\S]*?\n\}/),
    grab(/window\.showTreeOverlayContext = function \(\) \{[\s\S]*?\n\};/),
    grab(/window\.showMainCloudContext = function \(\) \{[\s\S]*?\n\};/),
].join("\n");

const ctx = {
    window: {},
    viewer: { renderCount: 0, render() { this.renderCount++; }, scene: { pointclouds: [], scene: {} } },
    stagedVisuals: [],
    inspectCard: { style: { display: "" } },
    pickedPointMarker: { visible: true },
    treeOverlayPoints: null,
    updateTreeColors() {},
    updateStandFilterHud() {},
    console,
};
vm.createContext(ctx);
vm.runInContext(
    code + "\nthis.__api = { showTreeOverlayContext: window.showTreeOverlayContext, showMainCloudContext: window.showMainCloudContext };",
    ctx);

const seed = () => {
    ctx.viewer.scene.pointclouds = [{ visible: true, name: "octree" }, { visible: true, name: "octree2" }];
    ctx.stagedVisuals = [{ visible: true, name: "staged-42" }];
    ctx.treeContextHidden = [];
};
const visibleCount = () =>
    ctx.viewer.scene.pointclouds.filter(p => p.visible).length + ctx.stagedVisuals.filter(g => g.visible).length;

seed();
ctx.__api.showTreeOverlayContext();                       // Forestry, no tree loaded
const empty = visibleCount() === 0;
console.log(`${empty ? "PASS" : "FAIL"}  Forestry tab with no tree overlay -> empty viewport ` +
            `(${visibleCount()} visible, treeContextHidden=${ctx.treeContextHidden.length})`);

ctx.__api.showMainCloudContext();                         // back to Layers
const restored = visibleCount() === 3 && ctx.treeOverlayPoints === null;
console.log(`${restored ? "PASS" : "FAIL"}  Layers tab restores the full context (${visibleCount()} visible)`);

ctx.treeOverlayPoints = { visible: false, geometry: {}, material: {} };  // a tree is loaded
ctx.__api.showTreeOverlayContext();
const overlayOnly = ctx.treeOverlayPoints.visible === true && visibleCount() === 0;
console.log(`${overlayOnly ? "PASS" : "FAIL"}  Forestry tab with a tree overlay -> overlay only ` +
            `(${visibleCount()} visible)`);

process.exit(empty && restored && overlayOnly ? 0 : 1);
