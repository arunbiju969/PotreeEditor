# Direct Potree 2.0 Octree Editing — Feature Status

_Date: 2026-10-30 · Module: [`potree2_patch.py`](potree2_patch.py) · Status: **core tool working, not yet wired into the editor**_

Editing a Potree 2.0 cloud's **attribute values in place** — writing straight into
`octree.bin` — so a relabel takes seconds instead of a full re-conversion.

## Why this path

Potree 2.0 stores points as fixed-stride records, so an attribute is addressable by byte
offset. A re-conversion of a cloud this size costs minutes of CPU and gigabytes of peak
memory; a patch touches only the records whose value changes. Potree's own layer model
keeps per-cloud state on the material, and the hierarchy/octree pair is stable under
value-only edits, so nothing else needs to be rewritten.

Decision recorded during the format survey: **Potree 2.0 tiles load and pan faster than
COPC tiles**, so 2.0 is the editing format and the COPC writer stays in the repo off the
roadmap.

## What a Potree 2.0 cloud looks like on disk

| File | Contents |
| --- | --- |
| `metadata.json` | version, encoding, attribute layout, scale/offset, per-attribute min/max and histogram |
| `hierarchy.bin` | flat table of 22-byte node records: `struct.Struct("<BBiqq")` = type, childMask, numPoints, byteOffset, byteSize |
| `octree.bin` | the points, one fixed-stride record each (39 bytes on the reference cloud) |

Two invariants the patcher depends on:

* A non-proxy node's `byteSize` is exactly `numPoints × stride`.
* **Type 2 (proxy) nodes address `hierarchy.bin`, not `octree.bin`** — validating their
  ranges against the point stride is a false-positive source (e.g. `8250 + 176` sits just
  past `firstChunkSize` 8228, not on a record boundary).

And one quirk: roughly **0.29 % of points are stored twice** — under an ancestor node and
again under a descendant (48 overlapping ranges on the reference cloud). Patching per node
therefore double-counts; the patcher reduces to the **unique record set**.

## How the patcher works

1. Read `metadata.json`; guard on version 2.0 and the DEFAULT encoding; derive the record
   stride from the attribute table (a multi-element attribute such as `rgb` becomes
   `rgb_0…rgb_3`). `size % stride` must be zero.
2. Read the hierarchy table, validating proxy ranges against `hierarchy.bin` length and
   ordinary ranges against the stride.
3. Select the records to patch — all, by attribute equality/inequality/range/`in`, or by
   world-space bbox — and reduce to unique node ranges.
4. `numpy.memmap` `octree.bin` as a structured array and write only records whose value
   differs from the new one.
5. Refresh `min`/`max` and the histogram for the touched attribute, then replace
   `metadata.json` atomically (write temp, `os.replace`).

A value may be a constant **or another attribute name**, which copies that attribute across
the selection.

### Command line

```bat
python potree2_patch.py C:\path\to\converted --info

python potree2_patch.py C:\path\to\converted ^
    --set treeID=42 --where classification=eq:2 --bbox 599900,4117900,10,600400,4118400,60 ^
    --dry-run --expect 120000 --save-metadata
```

`--where NAME=op:value` accepts `eq`, `ne`, `ge`, `le`, `in` (comma list). Ask for an
attribute the cloud does not have and it stops with the cloud's real attribute list.

## Demonstrated

On `clipped_Area#1_20260407_LLR_L2_normalized_converted` — **46,417,021 points**, 39 B/pt,
`octree.bin` 1,810,263,819 bytes, `hierarchy.bin` 425,458 bytes (19,339 nodes, 839 of them
proxies), attributes including `treeID` at byte 35 and `classification` 2/4:

* Hierarchy walk, selection and in-place patch complete in **seconds**, against re-conversion
  on the order of several minutes plus multi-GB peak memory.
* A demo patch wrote **43,079 records**; it was later restored from the sibling
  `…_normalized.laz` — 5,871 positions matched exactly and 37,208 matched within ±1 mm,
  with no ambiguous, missing or zeroed points left. The ±1 mm matching was necessary
  because PotreeConverter re-quantises positions, so only ~12 % of source positions land on
  the tree's grid.

## Not in scope for in-place editing

Attribute values of fixed width — including positions, intensity and rgb — are all editable
in place. Changing **point counts per node**, the octree structure, or the attribute layout
requires rewriting `octree.bin`.

## Outstanding

| Item | State |
| --- | --- |
| Undo / redo | Not built. The demo had to reconstruct previous values from the source LAZ because nothing recorded them — an undo file storing the old value per changed record is the fix |
| Wire into the editor | Not started. Staged tree edits (`treeID`, `classification`) still round-trip through PotreeConverter; committing them through the patcher is the next piece of work |
| Recolour confirmation | Unverified — patch a cell, then check the viewer shows it (the page loads `app.js` once, so restart between changes) |
| Colour-mode lists | Fixed five entries plus attribute names appended on demand (`treeID` included), not a full attribute-aware list |
| Picking / measure / commit routing | All act on the active layer, but potree's own picking has one `pointclouds[0]` path (`viewer.js:989`); route explicitly where it matters |
| Point budget | Viewer-wide, so two loaded clouds share it — roughly half each |

## Known defects in surrounding code

* `pointcloud_manager.py:232-239` — `ValueError: field 'itc' occurs more than once` when a
  cloud repeats the attribute (P0).
* `server.py:260` — `bytes=-N` suffix ranges not handled.
* Staged `delete` commits are silent no-ops.
* No path containment on the file API, CORS `*`, and `/api/save_las` is unauthenticated.
* The viewer is stale after a commit until refreshed.
* `window.toggleSettingsMenu` is referenced but not defined.

## Related

* Renderer fix carried upstream: `57473258` on `arunbiju969/potree` `develop` clamps the
  visibility-texture upload so an aggressive LOD cannot throw `RangeError` out of the render
  loop; the editor pins that revision in [`potree.version`](potree.version).
* Copied from Potree's own layer manager: the scene's point-cloud list **is** the layer list,
  and a cloud's **name** is its identity.
