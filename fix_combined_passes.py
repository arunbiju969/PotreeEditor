#!/usr/bin/env python3
"""
Fix Pass 1 / Pass 2 merge offset collision in combined.laz:
1. Points with user_data == 1 (Pass 1): untouched (IDs 1..12937).
2. Points with user_data == 2 (Pass 2):
   - Current itc == 12936 (was pass 2 unassigned '-1' that collided with pass 1 tree 12936):
     remap to itc = 0, tree_id = -1, user_data = 0, tree_score = 0.0.
   - Current itc >= 12937 (pass 2 trees):
     shift by +1 so that:
     pass 2 tree 0 (current 12937) -> 12938
     pass 2 tree 1 (current 12938) -> 12939
     ...
     pass 2 tree 5357 (current 18294) -> 18295.
3. Points with user_data == 0: untouched (ground, itc 0).
"""
import os
import sys
import time
import laspy
import numpy as np

def fix_combined_laz(src_path: str, dst_path: str, chunk_size: int = 10_000_000):
    t0 = time.time()
    print("=" * 65)
    print("COMBINED LAZ REPAIR: PASS 1 / PASS 2 COLLISION REMAPPING")
    print(f"Source:      {src_path}")
    print(f"Destination: {dst_path}")
    print("=" * 65)

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")

    with laspy.open(src_path) as reader:
        header = reader.header
        total_pts = header.point_count
        print(f"Total points in source: {total_pts:,}")
        print(f"Point format: {header.point_format.id}, Version: {header.version}")

        # Create destination header
        dst_hdr = laspy.LasHeader(point_format=header.point_format.id, version=header.version)
        dst_hdr.scales = header.scales
        dst_hdr.offsets = header.offsets
        for ed in header.point_format.extra_dimensions:
            dst_hdr.add_extra_dim(laspy.ExtraBytesParams(name=ed.name, type=ed.dtype))
        dst_hdr.vlrs = [v for v in header.vlrs if getattr(v, "user_id", "") != "copc"]

        total_processed = 0
        total_unassigned_fixed = 0
        total_p2_trees_shifted = 0

        with laspy.open(dst_path, mode="w", header=dst_hdr) as writer:
            for chunk_idx, chunk in enumerate(reader.chunk_iterator(chunk_size), start=1):
                t_chk = time.time()
                pts_count = len(chunk)

                ud = chunk.user_data
                itc = chunk.itc
                tid = chunk.tree_id

                is_p2 = (ud == 2)
                if np.any(is_p2):
                    # 1. Fix unassigned pass 2 points that were mapped to 12936
                    m_unassigned = is_p2 & (itc == 12936)
                    cnt_unassigned = int(np.sum(m_unassigned))
                    if cnt_unassigned > 0:
                        itc[m_unassigned] = 0
                        tid[m_unassigned] = -1
                        ud[m_unassigned] = 0
                        if "tree_score" in chunk.point_format.dimension_names:
                            chunk.tree_score[m_unassigned] = 0.0
                        total_unassigned_fixed += cnt_unassigned

                    # 2. Shift valid pass 2 trees by +1 (12937 -> 12938, etc.)
                    m_valid_trees = is_p2 & (itc >= 12937)
                    cnt_valid = int(np.sum(m_valid_trees))
                    if cnt_valid > 0:
                        itc[m_valid_trees] += 1
                        tid[m_valid_trees] += 1
                        total_p2_trees_shifted += cnt_valid

                    chunk.user_data = ud
                    chunk.itc = itc
                    chunk.tree_id = tid

                writer.write_points(chunk)
                total_processed += pts_count
                pct = (total_processed / total_pts) * 100.0
                dt = time.time() - t_chk
                print(f"  [Chunk {chunk_idx:02d}] {total_processed:,}/{total_pts:,} ({pct:5.1f}%) in {dt:.1f}s | "
                      f"unassigned fixed: {total_unassigned_fixed:,} | p2 trees shifted: {total_p2_trees_shifted:,}",
                      flush=True)

    elapsed = time.time() - t0
    print("=" * 65)
    print(f"REPAIR COMPLETE in {elapsed:.1f}s")
    print(f"Total points written:        {total_processed:,}")
    print(f"Pass 2 unassigned corrected: {total_unassigned_fixed:,} (detached from tree 12936)")
    print(f"Pass 2 tree points shifted:  {total_p2_trees_shifted:,}")
    print(f"Output saved -> {dst_path}")
    print("=" * 65)

if __name__ == "__main__":
    src = "/run/media/neo/Omen/Data/20260424_LLR_L3_entire_tract_stitched.potreeedit/segmented/combined.laz"
    dst = "/run/media/neo/Omen/Data/20260424_LLR_L3_entire_tract_stitched.potreeedit/segmented/combined_fixed.laz"
    if len(sys.argv) > 1:
        src = sys.argv[1]
    if len(sys.argv) > 2:
        dst = sys.argv[2]
    fix_combined_laz(src, dst)
