"""Edit attributes in place inside a Potree 2.0 point cloud.

A converted Potree 2.0 cloud is a folder of ``metadata.json``, ``hierarchy.bin`` and
``octree.bin``.  ``octree.bin`` is nothing but fixed-size point records: the stride is
the sum of the attribute sizes declared in ``metadata.json`` (39 bytes/point in a
46,417,021-point cloud, 41 in another), and ``metadata.json`` states each attribute's
byte offset inside a record.  ``hierarchy.bin`` is a flat table of 22-byte node
records -- type, childMask, numPoints, byteOffset, byteSize -- exactly as the shipped
``OctreeLoader.parseHierarchy`` reads them, so a node's byte range is *written down*
rather than recomputed, and ``byteSize == numPoints * stride`` holds for every node.

That makes an attribute-only edit a local operation: rewrite the attribute bytes,
refresh the statistics in ``metadata.json``, and ``hierarchy.bin`` plus both file
sizes stay put, so streaming behaviour and node identity are untouched.  Geometry
edits are deliberately out of scope -- moving, adding or removing points changes
which node a point belongs to, which shifts every later byteOffset and means
rebuilding the tree.

Two measured quirks drive the design:

* 0.29 % of the points in a real cloud are stored twice, once under an ancestor and
  once under a descendant, and 48 node ranges overlap.  Edits therefore address the
  *unique set of records* rather than iterating node by node, otherwise the viewer
  can show the stale copy of a half-patched point.
* ``metadata.json`` caches per-attribute statistics, including a classification
  histogram that drives potree's colour-by-classification UI, so they are recomputed
  after a patch instead of left lying.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

NODE = struct.Struct("<BBiqq")       # type, childMask, numPoints, byteOffset, byteSize
PROXY = 2                            # node whose subtree continues in a later chunk
DTYPE_OF = {"int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
            "int32": "<i4", "uint32": "<u4", "int64": "<i8", "uint64": "<u8",
            "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}


def record_dtype(attributes: list) -> np.dtype:
    """Structured dtype for one point record, built from metadata.json attributes.

    Multi-element attributes become one field per element, so ``position`` turns into
    ``x``, ``y``, ``z`` and ``rgb`` into ``rgb_0`` and friends.
    """
    names, formats, offsets, cur = [], [], [], 0
    for a in attributes:
        base = DTYPE_OF[a["type"]]
        if a["size"] != a["numElements"] * a["elementSize"]:
            raise ValueError(f"attribute {a['name']!r} size {a['size']} does not match "
                             f"{a['numElements']} x {a['elementSize']}")
        if a["numElements"] > 1:
            labels = (["x", "y", "z"][:a["numElements"]] if a["name"] == "position"
                      else [f"{a['name']}_{e}" for e in range(a["numElements"])])
            for e, label in enumerate(labels):
                names.append(label)
                formats.append(base)
                offsets.append(cur + e * a["elementSize"])
        else:
            names.append(a["name"])
            formats.append(base)
            offsets.append(cur)
        cur += a["size"]
    return np.dtype({"names": names, "formats": formats, "offsets": offsets,
                     "itemsize": cur, "aligned": False})


class Potree2Cloud:
    """Read and patch view over a converted cloud directory."""

    def __init__(self, path: Path):
        self.dir = Path(path)
        meta_path = self.dir / "metadata.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta.get("version") != "2.0":
            raise ValueError(f"{self.dir.name}: metadata says version "
                             f"{self.meta.get('version')!r}, this tool handles 2.0")
        if self.meta.get("encoding") not in (None, "DEFAULT"):
            raise NotImplementedError(
                f"{self.dir.name}: octree encoding is {self.meta.get('encoding')!r}; "
                f"in-place patching only applies to uncompressed (DEFAULT) records")
        self.attributes = self.meta["attributes"]
        self.stride = sum(a["size"] for a in self.attributes)
        self.dtype = record_dtype(self.attributes)
        self.offset = list(self.meta["offset"])
        self.scale = list(self.meta["scale"])
        self.octree_path = self.dir / "octree.bin"
        self.hierarchy_path = self.dir / "hierarchy.bin"
        size = self.octree_path.stat().st_size
        if size % self.stride:
            raise ValueError(f"{self.octree_path.name} is {size:,} bytes, not a multiple "
                             f"of the {self.stride}-byte record stride")
        self.record_count = size // self.stride
        self.nodes = self._read_nodes()
        self._ranges = None

    # ---- structure -----------------------------------------------------------
    def _read_nodes(self) -> list:
        raw = self.hierarchy_path.read_bytes()
        if len(raw) % NODE.size:
            raise ValueError(f"{self.hierarchy_path.name} is {len(raw):,} bytes, not a "
                             f"multiple of the {NODE.size}-byte node record")
        nodes = [NODE.unpack_from(raw, i * NODE.size) for i in range(len(raw) // NODE.size)]
        # A proxy node's two offset fields address *hierarchy.bin* (OctreeLoader reads
        # them as hierarchyByteOffset/Size when type is 2), so only ordinary nodes have
        # to land on whole point records inside octree.bin.
        for t, _cm, num, off, size in nodes:
            if t == PROXY:
                if off + size > len(raw):
                    raise ValueError(f"proxy node points at {off:,}+{size:,}, past the "
                                     f"{len(raw):,}-byte hierarchy.bin")
            elif size % self.stride:
                raise ValueError(f"node byte range {off:,}+{size:,} is not a whole "
                                 f"number of {self.stride}-byte records")
        return nodes

    def record_ranges(self) -> np.ndarray:
        """(first_record, count) for every data-bearing node, duplicates included.

        Proxy nodes (type 2) only point at more hierarchy and hold no records.  Some
        points appear under both an ancestor and a descendant, which is why callers go
        through :meth:`records_to_patch` for the unique set.
        """
        if self._ranges is None:
            self._ranges = np.array([(off // self.stride, size // self.stride)
                                      for t, _cm, _n, off, size in self.nodes
                                      if t != PROXY and size > 0], dtype=np.int64)
        return self._ranges

    def all_records(self) -> np.ndarray:
        """Every record index the tree references, deduplicated."""
        return np.unique(self.records_to_patch(None))

    def records_to_patch(self, selection: np.ndarray | None) -> np.ndarray:
        """Sorted unique record indices covered by a boolean selection (all if None)."""
        ranges = self.record_ranges()
        blocks = []
        for first, count in ranges:
            idx = first + np.arange(count, dtype=np.int64)
            if selection is not None:
                idx = idx[selection[idx]]
            blocks.append(idx)
        if not blocks:
            return np.zeros(0, np.int64)
        return np.unique(np.concatenate(blocks))

    def records(self, mode: str = "r") -> np.ndarray:
        """The whole octree.bin as structured records (numpy memmap, so patching is a
        strided pwrite and the file never has to fit in memory)."""
        return np.memmap(self.octree_path, dtype=self.dtype, mode=mode,
                         shape=(self.record_count,))

    def field_names(self) -> list:
        return list(self.dtype.names)

    def attribute(self, name: str) -> dict:
        for a in self.attributes:
            if a["name"] == name:
                return a
        raise KeyError(f"no attribute {name!r}; this cloud has "
                       f"{[a['name'] for a in self.attributes]}")

    def world_xyz(self, records: np.ndarray, mm: np.ndarray) -> np.ndarray:
        """Record indices -> world metres, using the cloud's own scale and offset."""
        sc = np.asarray(self.scale, np.float64)
        of = np.asarray(self.offset, np.float64)
        return np.column_stack([mm["x"][records], mm["y"][records],
                               mm["z"][records]]).astype(np.float64) * sc + of

    # ---- editing -------------------------------------------------------------
    def patch(self, name: str, value, selection: np.ndarray | None = None,
              mode: str = "r+", dry_run: bool = False) -> dict:
        """Write ``value`` into ``name`` for the selected records.

        ``value`` may be a constant or the name of another attribute (a copy, which is
        how an id column gets seeded from an existing one).  ``selection`` is a boolean
        array over record indices.  Returns how many points changed, with the before
        and after statistics for the touched records.
        """
        mm = self.records("r+" if not dry_run else "r")
        try:
            target = np.asarray(mm[name])
            if isinstance(value, str):
                other = np.asarray(mm[value])
                new_values = lambda idx: other[idx]
                check_src = value
            else:
                cast = np.asarray(value, target.dtype)
                if cast.ndim != 0:
                    raise ValueError(f"value {value!r} for {name!r} must be a scalar")
                new_values = lambda idx: np.full(len(idx), cast, target.dtype)
                check_src = None
            before_unique = None
            changed_total = None
            touched = self.records_to_patch(selection)
            if touched.size:
                old = target[touched]
                new = new_values(touched)
                differs = old != new
                changed_total = int(differs.sum())
                before_unique = np.unique(old).tolist()
                if not dry_run and changed_total:
                    sel = touched[differs]
                    target[sel] = new[differs]
                    mm.flush()
            stats = {"attribute": name, "records": int(touched.size),
                     "changed": changed_total, "source_attribute": check_src,
                     "before_unique": before_unique[:16] if before_unique else None}
            if not dry_run:
                stats.update(self.refresh_statistics(name, mm))
            return stats
        finally:
            del mm

    def refresh_statistics(self, name: str, mm: np.ndarray | None = None) -> dict:
        """Recompute the cached min/max (and histogram, if the attribute has one).

        potree reads these from metadata.json -- the classification histogram in
        particular drives colour-by-classification -- so leaving them stale would make
        the viewer show colours that no longer match the data.
        """
        attr = self.attribute(name)
        own = mm is None
        if own:
            mm = self.records("r")
        try:
            values = np.asarray(mm[name])
            out = {"min": values.min(), "max": values.max()}
            if attr.get("histogram") is not None:
                counts = np.bincount(values.astype(np.int64), minlength=256)
                attr["histogram"] = counts[:len(attr["histogram"])].tolist()
                out["histogram_total"] = int(counts.sum())
            attr["min"] = [out["min"].item() if hasattr(out["min"], "item") else out["min"]]
            attr["max"] = [out["max"].item() if hasattr(out["max"], "item") else out["max"]]
            if own:
                del mm
            return out
        finally:
            if own:
                del mm

    def write_metadata(self) -> Path:
        """Rewrite metadata.json atomically, so a crash cannot leave it half written."""
        path = self.dir / "metadata.json"
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), prefix="metadata.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.meta, f, indent=1, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    def summary(self) -> str:
        ranges = self.record_ranges()
        proxies = sum(1 for n in self.nodes if n[0] == PROXY)
        return (f"{self.dir.name}: {self.meta['points']:,} points, {self.stride} B/pt, "
                f"{self.record_count:,} records, {len(self.nodes):,} nodes "
                f"({len(ranges):,} with records, {proxies} proxies), "
                f"scale {self.scale}, offset {self.offset}, encoding "
                f"{self.meta.get('encoding')}, attributes {[a['name'] for a in self.attributes]}")


# ---- command line -------------------------------------------------------------------
def parse_where(cloud: Potree2Cloud, spec: str, mm: np.ndarray):
    """``name=op:value`` filter; ops are eq ne ge le in bbox-less numeric tests."""
    name, _, rest = spec.partition("=")
    op, _, raw = rest.partition(":")
    if name not in cloud.dtype.names:
        raise KeyError(f"--where {name}: not an attribute of this cloud")
    values = np.asarray(mm[name])
    if op == "eq":
        return values == np.asarray(raw, values.dtype)
    if op == "ne":
        return values != np.asarray(raw, values.dtype)
    if op == "ge":
        return values >= np.asarray(raw, values.dtype)
    if op == "le":
        return values <= np.asarray(raw, values.dtype)
    if op in ("in", "any"):
        wanted = [np.asarray(p, values.dtype) for p in raw.split(",")]
        mask = np.zeros(len(values), bool)
        for w in wanted:
            mask |= values == w
        return mask
    raise ValueError(f"--where {spec!r}: unknown operator {op!r} (eq ne ge le in)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cloud", help="folder holding metadata.json / hierarchy.bin / octree.bin")
    ap.add_argument("--info", action="store_true", help="print the layout and exit")
    ap.add_argument("--set", dest="set_pair", metavar="NAME=VALUE",
                    help="attribute to write; VALUE is a constant or another attribute name")
    ap.add_argument("--where", action="append", default=[],
                    metavar="NAME=OP:VALUE",
                    help="keep only points matching OP (eq, ne, ge, le, in) -- repeatable, ANDed")
    ap.add_argument("--bbox", metavar="X0,Y0,Z0,X1,Y1,Z1",
                    help="keep only points inside this world-coordinate box")
    ap.add_argument("--dry-run", action="store_true", help="count changes, write nothing")
    ap.add_argument("--save-metadata", action="store_true",
                    help="rewrite metadata.json with refreshed statistics")
    ap.add_argument("--expect", type=int, metavar="N",
                    help="fail if the number of changed points is not exactly N")
    args = ap.parse_args(argv)

    cloud = Potree2Cloud(Path(args.cloud))
    print(cloud.summary())
    if args.info or not args.set_pair:
        for a in cloud.attributes:
            field = "x" if a["name"] == "position" else a["name"]
            if field not in cloud.dtype.fields:
                field = f"{a['name']}_0"          # multi-element attributes, e.g. rgb
            at = cloud.dtype.fields[field][1] if field in cloud.dtype.fields else "?"
            hist = f"  histogram total {sum(a['histogram']):,}" if a.get("histogram") else ""
            print(f"  {a['name']:<20} byte offset {at:>3}  {a['type']:<8} "
                  f"x{a['numElements']}  min {a.get('min')} max {a.get('max')}{hist}")
        return 0

    name, _, raw = args.set_pair.partition("=")
    if name not in cloud.dtype.names:
        raise KeyError(f"--set {name}: not an attribute of this cloud")
    value = None
    name_for = None
    if raw in cloud.dtype.names:
        value = raw
        name_for = f"copy of {raw}"
    else:
        try:
            value = float(raw) if "." in raw or "e" in raw.lower() else int(raw)
        except ValueError:
            raise SystemExit(f"--set {name}={raw!r}: not a number, and not an attribute "
                             f"of this cloud either; its attributes are "
                             f"{[a['name'] for a in cloud.attributes]}")
        name_for = repr(value)

    mm = cloud.records("r")
    try:
        selection = None
        for spec in args.where:
            part = parse_where(cloud, spec, mm)
            selection = part if selection is None else (selection & part)
        if args.bbox:
            lo, hi = np.asarray(args.bbox.split(",")[:3], np.float64), \
                     np.asarray(args.bbox.split(",")[3:6], np.float64)
            xyz = np.column_stack([np.asarray(mm["x"]), np.asarray(mm["y"]),
                                   np.asarray(mm["z"])]).astype(np.float64) \
                  * np.asarray(cloud.scale, np.float64) + np.asarray(cloud.offset, np.float64)
            inside = ((xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0]) &
                      (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1]) &
                      (xyz[:, 2] >= lo[2]) & (xyz[:, 2] <= hi[2]))
            selection = inside if selection is None else (selection & inside)
        if selection is not None:
            print(f"selection keeps {int(selection.sum()):,} of {len(selection):,} records")
        stats = cloud.patch(name, value, selection, dry_run=args.dry_run)
    finally:
        del mm

    print(f"  target {name} <- {name_for}")
    print(f"  records in scope: {stats['records']:,}   changed: {stats['changed']}")
    if args.expect is not None and stats["changed"] != args.expect:
        print(f"  --expect {args.expect} FAILED", file=sys.stderr)
        return 1
    print(f"  new statistics: min {stats.get('min')} max {stats.get('max')}"
          + (f" histogram total {stats['histogram_total']:,}" if "histogram_total" in stats else ""))
    if args.dry_run:
        print("  dry run: nothing written")
        return 0
    if args.save_metadata:
        print(f"  wrote {cloud.write_metadata()}")
    else:
        print("  octree.bin updated in place; metadata.json statistics NOT rewritten "
              "(--save-metadata to do it)")
    print(f"  sizes unchanged: octree.bin {cloud.octree_path.stat().st_size:,} bytes, "
          f"hierarchy.bin {cloud.hierarchy_path.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
