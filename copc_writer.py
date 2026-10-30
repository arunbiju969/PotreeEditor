"""Write a COPC file (``.copc.laz``) from a LAS/LAZ using only laspy + lazrs.

Why this exists
---------------
The editor needs one artifact that is simultaneously (a) an ordinary LAZ the Python
side can read and edit, (b) an octree the shipped Potree renderer streams by HTTP
range requests, and (c) self-describing about its CRS.  Nothing available here can
produce such a file: PotreeConverter 2.x has no COPC output at all, and the PDAL
2.10 build shipped with QGIS silently drops custom point dimensions (verified),
which would destroy the Tree ID column.  laspy 2.7 can *read* COPC but its
``CopcInfoVlr.record_data_bytes()`` raises ``NotImplementedError``.

Layout facts below were measured off ``potree/pointclouds/lion_takanawa.copc.laz``
-- the file this application renders -- and cross-checked against laspy 2.7's own
COPC parser, rather than taken from the specification text:

* hierarchy entry, 32 B: ``<iiii`` (level, x, y, z) + ``<Qii`` (byte offset,
  byte size, point count).  ``point_count == -1`` means "children live in another
  page at this offset"; ``0`` means "no point data for this key".
* ``copc`` / record 1 (info) VLR, payload 160 B: five doubles
  (center_x, center_y, center_z, halfsize, spacing), two int64
  (root hierarchy offset, root hierarchy size), two doubles (gps max, gps min),
  then 88 reserved zero bytes.  The sample carries no magic/version prefix.
* point data is LAZ with **one chunk per node**, i.e. the laszip VLR must say
  compressor version 1027, ``minor_version = 3`` and ``chunk_size = 0xFFFFFFFF``
  so the chunk table records a byte size per chunk.  Readers decode a node from
  that byte range; a version-514 table makes laz-perf give up.
* the hierarchy page follows the point data and the LAZ chunk table, wrapped in a
  VLR: user_id ``copc``, record_id 1000, description "EPT Hierarchy".  The info
  VLR's ``root_hier_offset`` points at that VLR's payload, and the info VLR itself
  must be the first VLR in the file -- laspy rejects it otherwise.
* lazrs buffers, so ``tell()`` while compressing is useless: its chunk table is the
  only trustworthy record of where chunks landed.  Chunks tile
  ``offset_to_point_data + 8`` (the 8 bytes lazrs reserves for the chunk-table
  offset) through to the chunk table itself, and the writer asserts that.
* the ExtraBytes "options" byte is cleared: potree's copc.js reads no_data/min/max
  at offsets laspy never wrote, and refuses the 64-bit sentinels waiting there.

Known limits of this first version (all raise, never silently corrupt):
* the whole cloud is held in memory (numpy int64 keys: ~8 B/point plus the records);
* hierarchy pages hold at most 2047 entries (the largest payload a VLR can carry);
  bigger trees spill into child pages that ``point_count = -1`` entries point at, so
  any size cloud fits.  What *is* fixed is ``chunk_size`` <= 50,000: lazrs flushes a
  LAZ chunk every 50,000 points internally and COPC wants one chunk per node.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Iterable, Optional

import lazrs
import laspy
import numpy as np
from laspy.vlrs.known import LasZipVlr

# --- measured layout -----------------------------------------------------------
INFO_RECORD_LEN = 160
EXTRA_RECORD_LEN = 192                                # one ExtraBytes record
INFO_TAIL = b"\x00" * (INFO_RECORD_LEN - 72)          # 88 reserved bytes
ENTRY = struct.Struct("<iiiiqii")                     # 32 B, see module docstring
VLR_HEADER_LEN = 54                                   # no 'ifpr' signature
CRS_VLR_RECORDS = (2111, 2112)                        # WKT2 / WKT1 CRS
GLOBAL_ENCODING_WKT_BIT = 0x10
MAX_PAGE_ENTRIES = 2047                               # 65535-byte VLR payload cap; child
#   pages could legally be bigger, one cap keeps the layout easy to reason about.


def _pack_info(center: np.ndarray, halfsize: float, spacing: float,
               hier_offset: int, hier_size: int,
               gps_max: float = 0.0, gps_min: float = 0.0) -> bytes:
    """Payload of the copc/info VLR; offsets are patched after the tree is written."""
    head = struct.pack("<5dqq2d", float(center[0]), float(center[1]), float(center[2]),
                       float(halfsize), float(spacing), int(hier_offset), int(hier_size),
                       float(gps_max), float(gps_min))
    assert len(head) == 72, f"info head is {len(head)} bytes, expected 72"
    return head + INFO_TAIL


def pack_entries(nodes: Iterable[tuple]) -> bytes:
    """nodes: iterable of (level, x, y, z, byte_offset, byte_size, point_count)."""
    out = bytearray()
    for lvl, x, y, z, off, size, count in nodes:
        if not -1 <= int(lvl) <= 62:
            raise ValueError(f"COPC level {lvl} out of range")
        out += ENTRY.pack(int(lvl), int(x), int(y), int(z), int(off), int(size), int(count))
    return bytes(out)


def _cell_indices(xyz: np.ndarray, origin: np.ndarray, cell: float, levels: int) -> np.ndarray:
    """Integer octant indices at ``levels`` subdivisions, clipped into [0, 2**levels)."""
    n = 1 << levels
    idx = np.floor((xyz - origin) / cell).astype(np.int64)
    return np.clip(idx, 0, n - 1)


def assign_levels(xyz: np.ndarray, center: np.ndarray, halfsize: float,
                  chunk_size: int, max_level: int) -> np.ndarray:
    """Per point, the level of the deepest node holding at most ``chunk_size`` points.

    Mirrors the shipped sample's shape: the root itself keeps the points of cells
    that were never subdivided, so no entry has a point count of zero for no reason.
    """
    n = len(xyz)
    side = 2.0 * halfsize
    levels = np.full(n, max_level, dtype=np.int32)
    pending = np.ones(n, dtype=bool)
    for lvl in range(0, max_level + 1):
        cell = side / (1 << lvl)
        idx = _cell_indices(xyz, center - halfsize, cell, lvl)
        flat = (idx[:, 0] * (1 << lvl) + idx[:, 1]) * (1 << lvl) + idx[:, 2]
        counts = np.bincount(flat)
        fits = counts[flat] <= chunk_size
        take = pending & fits
        levels[take] = lvl
        pending &= ~fits
        if not pending.any():
            break
    return levels


def build_tree(xyz: np.ndarray, center: np.ndarray, halfsize: float,
               chunk_size: int, max_level: int) -> tuple:
    """Return (entries_preorder, data_order, counts_by_node).

    ``entries_preorder`` is depth-first (parent before children, x fastest then y
    then z) exactly like the sample file; ``data_order`` is the post-order used to
    lay the chunks out, which also matches the sample (children's bytes first,
    the parent's own points last).
    """
    levels = assign_levels(xyz, center, halfsize, chunk_size, max_level)
    keys: dict[tuple, np.ndarray] = {}
    for lvl in np.unique(levels):
        cell = (2.0 * halfsize) / (1 << int(lvl))
        idx = _cell_indices(xyz[levels == lvl], center - halfsize, cell, int(lvl))
        trip = list(map(tuple, idx.tolist()))
        order = {}
        for i, t in enumerate(trip):
            order.setdefault(t, []).append(i)
        for t, rel in order.items():
            abs_idx = np.flatnonzero(levels == lvl)[np.asarray(rel, dtype=np.intp)]
            keys[(int(lvl),) + t] = abs_idx
    # structural ancestors of every node, so the tree is connected (point_count 0)
    for (lvl, x, y, z) in list(keys):
        step, cur = lvl, (lvl, x, y, z)
        while step > 0:
            step -= 1
            anc = (step, x >> (lvl - step), y >> (lvl - step), z >> (lvl - step))
            keys.setdefault(anc, np.empty(0, dtype=np.int64))

    def morton(node: tuple) -> int:
        _, x, y, z = node
        m = 0
        for bit in range(31):
            m |= ((x >> bit) & 1) << (2 * bit)
            m |= ((y >> bit) & 1) << (2 * bit + 1)
            m |= ((z >> bit) & 1) << (2 * bit + 2)
        return m

    children = {}
    for node in keys:
        if node[0] > 0:
            p = (node[0] - 1, node[1] >> 1, node[2] >> 1, node[3] >> 1)
            children.setdefault(p, []).append(node)

    preorder, postorder = [], []
    stack = [(min(keys)[0], 0, 0, 0)]
    seen = set()
    while stack:
        node = stack.pop()
        if node not in keys or node in seen:
            continue
        seen.add(node)
        preorder.append(node)
        postorder.append(node)
        kids = sorted(children.get(node, ()), key=morton)          # x fastest, z slowest
        # post-order: parent first in the list, children pushed after -> popped first
        for k in reversed(kids):
            stack.append(k)
    # The walk above already produces true DFS pre-order (a parent, then its
    # children in morton order).  Reversing it puts every parent after all of its
    # descendants, which is the byte layout the shipped sample uses.
    postorder = list(reversed(postorder))
    counts = {k: len(v) for k, v in keys.items()}
    return preorder, postorder, counts, keys


def paginate_tree(preorder: list, children: dict, max_entries: int) -> tuple:
    """Split a tree into hierarchy pages of at most ``max_entries`` entries.

    A subtree that no longer fits the page it would join gets a page of its own, and
    the parent entry marks it with ``point_count = -1`` plus the child page's file
    offset.  That is how COPC carries trees too big for a single VLR -- the browser's
    copc.js follows those links through ``Copc.loadHierarchyPage`` and laspy's
    CopcReader likewise -- so the only thing that bounds cloud size here is memory.
    """
    if max_entries < 2:
        raise ValueError("hierarchy pages need room for at least two entries")
    if not preorder:
        return [[]], {}, set()
    subtree: dict[tuple, int] = {}
    for node in reversed(preorder):
        subtree[node] = 1 + sum(subtree[c] for c in children.get(node, ()))
    if subtree[preorder[0]] != len(preorder):
        raise AssertionError(f"subtree sizes sum to {subtree[preorder[0]]} for "
                             f"{len(preorder)} nodes")

    pages: list[list] = [[preorder[0]]]
    subpages: dict[tuple, int] = {}
    links: set = set()          # (page index, node) rendered as a point_count = -1 link

    def walk(node, page_index: int) -> None:
        page = pages[page_index]
        for child in children.get(node, ()):
            if len(page) + subtree[child] <= max_entries:
                page.append(child)                    # whole subtree joins this page
                walk(child, page_index)
            elif len(page) < max_entries:
                # only the entry fits here: point_count -1 plus the offset of the page
                # that carries the subtree below it
                page.append(child)
                links.add((page_index, child))
                subpages[child] = len(pages)
                pages.append([child])
                walk(child, subpages[child])
            else:
                raise NotImplementedError(
                    f"hierarchy page {page_index} is full ({max_entries} entries) with "
                    f"subtrees left to place; this tree needs a larger page cap.")

    walk(preorder[0], 0)
    for page in pages:
        if not 1 <= len(page) <= max_entries:
            raise AssertionError(f"hierarchy page holds {len(page)} entries")
    return pages, subpages, links


def copc_laz_vlr_bytes(point_format_id: int, num_extra_bytes: int) -> bytes:
    """laszip VLR record data configured the way working COPC files have it.

    lazrs defaults to compressor version 514 with fixed 50k chunks.  Both the
    shipped sample and PDAL's output instead say version 1027, minor version 3
    and chunk_size 0xFFFFFFFF -- and laz-perf in the browser only decodes ranges
    from files spelled that way (a 514 file made it abort with
    "Cannot convert bigint to number").
    """
    vlr = lazrs.LazVlr.new_for_compression(point_format_id, num_extra_bytes)
    raw = bytearray(vlr.record_data())
    struct.pack_into("<H", raw, 4, 1027)              # compressor version (LAZ 1.4)
    struct.pack_into("<H", raw, 6, 3)                 # minor version 3 = variable chunks
    struct.pack_into("<I", raw, 12, 0xFFFFFFFF)       # variable chunk size
    check = lazrs.LazVlr(bytes(raw))
    if not check.uses_variable_size_chunks():
        raise RuntimeError("could not switch the lazrs VLR to variable-size chunks")
    return bytes(raw)


def _extra_bytes_option_bytes(header_blob: bytes) -> list:
    """Positions of the ExtraBytes "options" bytes that have to read as zero.

    laspy and potree's bundled copc.js disagree about the layout of an ExtraBytes
    record.  copc.js reads name at +4, description at +60, no_data at +40, min at
    +64, max at +88, scale at +112 and offset at +136, and pushes the integer ones
    through ``parseBigInt``, which refuses anything past +/-2^53-1.  laspy writes
    scales/offsets/no_data/min/max at its own offsets and sets the option bits that
    tell copc.js those slots exist -- so for an int32 dimension like TreeAIBox's
    ``itc`` the reader picks up 0x8000000000000000 where it expects ``max`` and the
    viewer dies with "Cannot convert bigint to number: -9223372036854775808", even
    though laspy itself reads the file back perfectly.

    The shipped COPC sidesteps the whole minefield by advertising no optional
    values: its options byte is 0, and every optional read is skipped.  Clearing
    that byte keeps the dimension itself (its type and length come from the type
    byte, and from the laszip VLR for the decoder) while removing metadata the
    viewer never uses.
    """
    needle = b"LASF_Spec".ljust(16, b"\x00") + struct.pack("<H", 4)
    pos = header_blob.find(needle)
    if pos < 0:
        return []
    length = struct.unpack_from("<H", header_blob, pos + len(needle))[0]
    payload = pos - 2 + 54                        # 2 bytes before user_id, then header
    return [payload + r * EXTRA_RECORD_LEN + 3
            for r in range(length // EXTRA_RECORD_LEN)]


def write_copc(src: Path, out: Path, chunk_size: int = 40_000, max_level: Optional[int] = None,
               carry_crs: bool = True, wkt: Optional[str] = None, verbose: bool = False,
               max_page_entries: int = MAX_PAGE_ENTRIES) -> dict:
    """Convert ``src`` (LAS/LAZ) into the COPC file ``out``.

    ``carry_crs`` copies CRS VLRs (WKT records 2111/2112) from the source and keeps
    the global-encoding WKT bit, which is what makes two datasets in the same CRS
    line up with vector overlays and basemaps.  ``wkt`` overrides the CRS text.
    Returns a dict of stats; raises rather than writing a doubtful file.
    """
    src, out = Path(src), Path(out)
    with laspy.open(src) as f:
        header, points = f.header, f.read()
    rec = points.points.array
    fmt = header.point_format
    if header.version.major != 1 or header.version.minor != 4:
        header.major_version, header.minor_version = 1, 4

    scale = np.asarray(header.scales, dtype=np.float64)
    offset = np.asarray(header.offsets, dtype=np.float64)
    xyz = np.asarray(points.xyz, dtype=np.float64)
    if len(xyz) == 0:
        raise ValueError(f"{src.name}: no points to write")

    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    half = float((maxs - mins).max() / 2.0) or 1.0
    half *= 1.000001                                   # keep every point inside the root cell
    spacing = float(np.abs(scale).max()) or 0.001
    if max_level is None:
        max_level = max(1, min(12, int(math.ceil(math.log2(max(len(xyz), 2) / max(chunk_size, 2)))) + 1))

    preorder, postorder, counts, keys = build_tree(xyz, center, half, chunk_size, max_level)
    placed = sum(c for c in counts.values())
    if placed != len(xyz):
        raise AssertionError(f"tree lost points: {placed:,} assigned vs {len(xyz):,} in source")

    # ---- header + VLRs --------------------------------------------------------
    laz_bytes = copc_laz_vlr_bytes(fmt.id, fmt.num_extra_bytes)
    header.vlrs.clear()
    # The copc/info VLR must be the FIRST VLR in the file: laspy rejects the file
    # otherwise ("it is not the first VLR as it should"), and laz-perf in the
    # browser reads the VLR block in order and mis-parses everything after it.
    header.vlrs.append(laspy.VLR(user_id="copc", record_id=1, description="COPC info VLR",
                                 record_data=_pack_info(center, half, spacing, 0, 0)))
    if carry_crs:
        with laspy.open(src) as f:
            keep = [v for v in f.header.vlrs
                    if v.user_id in ("LASF_Projection", "WKT", "OGC WKT")
                    or getattr(v, "record_id", 0) in CRS_VLR_RECORDS]
        for v in keep:
            header.vlrs.append(laspy.VLR(user_id=v.user_id, record_id=v.record_id,
                                         description=v.description,
                                         record_data=v.record_data_bytes()))
        if wkt is not None and not keep:
            header.vlrs.append(laspy.VLR(user_id="LASF_Projection", record_id=2112,
                                         description="OGC WKT Coordinate System",
                                         record_data=wkt.encode("utf-8")))
    if wkt is not None:
        header.vlrs = [v for v in header.vlrs if getattr(v, "record_id", 0) not in CRS_VLR_RECORDS]
        header.vlrs.append(laspy.VLR(user_id="LASF_Projection", record_id=2112,
                                     description="OGC WKT Coordinate System",
                                     record_data=wkt.encode("utf-8")))
    header.vlrs.append(LasZipVlr(laz_bytes))
    # laspy appends the ExtraBytesVlr at the end, which lands the VLRs in the same
    # order the shipped sample uses: copc/info, laszip, then LASF_Spec.
    try:
        header._sync_extra_bytes_vlr()                # laspy builds the ExtraBytesVlr
    except Exception:                                 # pragma: no cover - laspy internals
        pass
    header.point_count = len(xyz)

    probe = __import__("io").BytesIO()
    header.write_to(probe)
    header_blob = probe.getvalue()
    header_bytes = len(header_blob)
    header.offset_to_point_data = header_bytes
    info_payload = _find_info_payload(header_blob, header_bytes)
    # Zero the ExtraBytesVlr option bits: potree's copc.js reads those optional slots
    # at offsets laspy never wrote, and chokes on laspy's 64-bit sentinels.
    option_bytes = _extra_bytes_option_bytes(header_blob)

    out.parent.mkdir(parents=True, exist_ok=True)
    dest = open(out, "wb")
    header.write_to(dest)
    if dest.tell() != header_bytes:
        dest.close()
        raise AssertionError(f"header is {dest.tell()} bytes, expected {header_bytes}")
    for at in option_bytes:
        dest.seek(at)
        dest.write(b"\x00")
    dest.seek(header_bytes)

    # ---- point data: one LAZ chunk per node -----------------------------------
    comp = lazrs.LasZipCompressor(dest, lazrs.LazVlr(laz_bytes))
    # The compressor reserves the 8 bytes at the head of the point data for the
    # offset to the LAZ chunk table, so the first chunk starts 8 bytes further on.
    # (lazrs buffers, so its tell() lags behind that reservation.)
    data_start = header_bytes + 8
    offsets: dict[tuple, tuple[int, int]] = {}
    chunked, written = [], 0
    for node in postorder:
        idx = keys[node]
        if len(idx):
            batch = np.ascontiguousarray(rec[idx])
            comp.compress_many(np.frombuffer(batch, np.uint8))
            comp.finish_current_chunk()                  # one LAZ chunk per node
            chunked.append(node)
        else:
            offsets[node] = (data_start, 0)              # structural node: no chunk
        written += len(idx)
    comp.done()
    dest.close()
    if written != len(xyz):
        raise AssertionError(f"compressed {written:,} points, source has {len(xyz):,}")

    # lazrs buffers, so the chunk table it just wrote is the only trustworthy
    # record of where each chunk landed.  COPC wants one chunk per node, so table
    # entry i is node i in write order, and the ranges follow by summing sizes.
    table = _read_chunk_table(out, laz_bytes)
    while table and table[-1][0] == 0:
        table.pop()           # done() always closes one empty trailing chunk
    if len(table) != len(chunked):
        raise NotImplementedError(
            f"lazrs closed {len(table)} LAZ chunks for {len(chunked)} nodes, but COPC needs "
            f"one chunk per node.  chunk_size={chunk_size:,}: keep it at or below the lazrs "
            f"batch size (50,000) so no node is split across chunks.")
    at = data_start
    for node, (_npoints, nbytes) in zip(chunked, table):
        offsets[node] = (at, nbytes)
        at += nbytes
    point_data_end = at
    # Prove the layout: the chunk table the file advertises must start exactly
    # where the chunks end.  This is how the shipped sample is laid out too.
    with open(out, "rb") as fh:
        fh.seek(header_bytes)
        chunk_table_pos = struct.unpack("<q", fh.read(8))[0]
    if chunk_table_pos != point_data_end:
        raise AssertionError(
            f"chunks cover {data_start:,}..{point_data_end:,} but the file says the LAZ "
            f"chunk table starts at {chunk_table_pos:,}")

    # ---- hierarchy pages, root page in a VLR, then patch the info VLR ---------
    children: dict[tuple, list] = {}
    for node in preorder:
        lvl, x, y, z = node
        if lvl:
            children.setdefault((lvl - 1, x >> 1, y >> 1, z >> 1), []).append(node)
    pages, subpages, links = paginate_tree(preorder, children, max_page_entries)

    # Every position is knowable before writing anything: a page is just entries, the
    # root page sits in the VLR payload at the current end of file, and child pages
    # follow it back to back.
    hier_vlr_start = out.stat().st_size
    vlr_header = (struct.pack("<2s16sHH", b"\x00\x00", b"copc", 1000,
                              ENTRY.size * len(pages[0]))
                  + b"EPT Hierarchy".ljust(32, b"\x00"))
    page_pos = [hier_vlr_start + len(vlr_header)]
    for i in range(1, len(pages)):
        page_pos.append(page_pos[-1] + ENTRY.size * len(pages[i - 1]))

    blobs = []
    for index, page in enumerate(pages):
        rows = []
        for node in page:
            lvl, x, y, z = node
            if (index, node) in links:      # children live in another page
                kid = subpages[node]
                rows.append((lvl, x, y, z, page_pos[kid], ENTRY.size * len(pages[kid]), -1))
            elif counts[node] > 0:
                off, size = offsets[node]
                rows.append((lvl, x, y, z, off, size, counts[node]))
            else:
                anc_off, anc_size = offsets.get(node, (0, 0))
                rows.append((lvl, x, y, z, anc_off, anc_size, 0))
        blobs.append(pack_entries(rows))

    # The root page lives inside a VLR: user_id 'copc', record_id 1000 (laspy's
    # CopcHierarchyVlr.official_record_ids) and description 'EPT Hierarchy'; the
    # shipped sample and PDAL's output both spell it that way.
    with open(out, "r+b") as f:
        f.seek(0, 2)
        if f.tell() != hier_vlr_start:
            raise AssertionError(f"hierarchy would land at {f.tell():,}, planned for "
                                 f"{hier_vlr_start:,}")
        f.write(vlr_header)
        for blob in blobs:
            f.write(blob)
        f.seek(info_payload + 40)
        f.write(struct.pack("<qq", page_pos[0], ENTRY.size * len(pages[0])))

    stats = {"points": len(xyz), "nodes": len(preorder), "levels": int(max(k[0] for k in keys)),
             "bytes": out.stat().st_size, "chunk_size": chunk_size,
             "hierarchy_offset": page_pos[0], "hierarchy_bytes": ENTRY.size * len(pages[0]),
             "pages": len(pages),
             "extra_dims": [d.name for d in fmt.extra_dimensions],
             "extra_bytes_options_cleared": len(option_bytes),
             "center": center.tolist(), "halfsize": half, "spacing": spacing,
             "crs_vlrs": [v.record_id for v in header.vlrs
                          if getattr(v, "record_id", 0) in CRS_VLR_RECORDS]}
    if verbose:
        for k, v in stats.items():
            print(f"  {k:<18} {v}")
    return stats


def _find_info_payload(header_blob: bytes, header_bytes: int) -> int:
    """Byte position of the copc/info payload inside the header just written."""
    needle = b"copc" + b"\x00" * 12 + struct.pack("<H", 1)
    pos = header_blob.find(needle)
    if pos < 0:
        raise AssertionError("the copc/info VLR was not written where expected")
    payload = pos + 52                                # uid 16 + rid 2 + len 2 + desc 32
    if payload + INFO_RECORD_LEN > header_bytes:
        raise AssertionError("copc/info payload runs past the header")
    return payload


def _read_chunk_table(path: Path, laz_bytes: bytes) -> list:
    """Per-chunk (points, compressed bytes) exactly as lazrs recorded them."""
    with laspy.open(path) as f:
        point_data_start = f.header.offset_to_point_data
    with open(path, "rb") as fh:
        fh.seek(point_data_start)         # lazrs reads the chunk-table offset from here
        return list(lazrs.read_chunk_table(fh, lazrs.LazVlr(laz_bytes)))
