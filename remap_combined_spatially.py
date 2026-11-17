#!/usr/bin/env python3
"""
Spatially Align Pass 2 Points in combined.laz
============================================
Root cause of horizontal stripes:
When pass 2 (10,532,224 points) was segmented, it was sliced into 70m tiles.
When merged into `pass2_unassigned_entire_tract_stitched.las`, points were
concatenated in TILE ORDER.
The master point cloud in `combined.laz` has points in SCANLINE ORDER.
Blindly copying labels index-by-index mapped tile-ordered labels onto
horizontal flight scanlines, stretching single trees into 30m-70m horizontal bars!

Fix:
1. Re-align Pass 2 stitched labels back to scanline order via exact (X, Y, Z, gps_time) matching.
2. Apply clean non-colliding tree IDs: pass2_tree_id + 12938 for valid trees (>=0), and -1 for unassigned.
3. Stream write the master 213M cloud with spatially aligned Pass 2 labels.
"""
import os
import sys
import time
import laspy
import numpy as np

def run_spatial_alignment():
    t0 = time.time()
    pass2_unassigned_path = "/run/media/neo/Omen/Data/pass2_unassigned.laz"
    pass2_stitched_path = "/run/media/neo/Omen/Data/20260424_LLR_L3_entire_tract_stitched.potreeedit/segmented/pipeline2/pass2_unassigned_entire_tract_stitched.las"
    combined_src_path = "/run/media/neo/Omen/Data/20260424_LLR_L3_entire_tract_stitched.potreeedit/segmented/combined_corrupt_offset.laz.bak"
    combined_dst_path = "/run/media/neo/Omen/Data/20260424_LLR_L3_entire_tract_stitched.potreeedit/segmented/combined_spatially_aligned.laz"

    print("=" * 70)
    print("SPATIAL REALIGNMENT & CORRECTION OF MULTI-PASS COMBINED CLOUD")
    print("=" * 70)

    # 1. Load the 10.5M points from both representations to compute permutation
    print(f"[Step 1/4] Loading scanline-order points from: {pass2_unassigned_path}...")
    f_un = laspy.open(pass2_unassigned_path)
    p_un = f_un.read()
    n_p2 = len(p_un)
    print(f"  Loaded {n_p2:,} unassigned scanline points.")

    print(f"[Step 2/4] Loading tile-order segmented points from: {pass2_stitched_path}...")
    f_st = laspy.open(pass2_stitched_path)
    p_st = f_st.read()
    if len(p_st) != n_p2:
        raise ValueError(f"Point count mismatch: {len(p_st)} vs {n_p2}")
    print(f"  Loaded {len(p_st):,} segmented points.")

    print("[Step 3/4] Computing exact (X, Y, Z, gps_time) permutation mapping...")
    t_sort = time.time()
    idx_un = np.lexsort((p_un.gps_time, p_un.Z, p_un.Y, p_un.X))
    idx_st = np.lexsort((p_st.gps_time, p_st.Z, p_st.Y, p_st.X))

    inv_idx_un = np.empty_like(idx_un)
    inv_idx_un[idx_un] = np.arange(n_p2)
    p2_to_scanline = idx_st[inv_idx_un]
    print(f"  Permutation computed in {time.time() - t_sort:.1f}s.")

    # Extract and map Pass 2 fields into scanline order
    st_tid = np.asarray(p_st.tree_id)
    st_score = np.asarray(p_st.tree_score)
    st_sem = np.asarray(p_st.semantic_pred)
    st_cls = np.asarray(p_st.classification)

    aligned_raw_tid = st_tid[p2_to_scanline]
    aligned_score = st_score[p2_to_scanline]
    aligned_sem = st_sem[p2_to_scanline]
    aligned_cls = st_cls[p2_to_scanline]

    # Free memory of full unassigned/stitched reader objects
    p_un = p_st = idx_un = idx_st = inv_idx_un = p2_to_scanline = None

    # Compute final tree IDs:
    # Pass 1 max is 12,937. Pass 2 valid trees (>=0) become raw_id + 12,938
    # Pass 2 unassigned (-1) becomes itc=0, tree_id=-1, user_data=0
    aligned_final_itc = np.zeros(n_p2, dtype=np.int32)
    aligned_final_tid = np.full(n_p2, -1, dtype=np.int32)
    aligned_final_ud = np.zeros(n_p2, dtype=np.uint8)

    is_valid_tree = (aligned_raw_tid >= 0)
    aligned_final_itc[is_valid_tree] = aligned_raw_tid[is_valid_tree] + 12938
    aligned_final_tid[is_valid_tree] = aligned_raw_tid[is_valid_tree] + 12938
    aligned_final_ud[is_valid_tree] = 2

    # Reset score for unassigned
    aligned_score[~is_valid_tree] = 0.0

    print(f"  Aligned {int(np.sum(is_valid_tree)):,} valid tree points (Pass 2 trees 12938..{int(np.max(aligned_final_itc))}).")
    print(f"  Aligned {int(np.sum(~is_valid_tree)):,} unassigned points (returned to background pool).")

    # 2. Stream through master 213M cloud and apply aligned labels to Pass 2 points
    print(f"\n[Step 4/4] Streaming master cloud ({combined_src_path}) -> {combined_dst_path}...")
    f_src = laspy.open(combined_src_path)
    total_pts = f_src.header.point_count

    dst_hdr = laspy.LasHeader(point_format=f_src.header.point_format.id, version=f_src.header.version)
    dst_hdr.scales = f_src.header.scales
    dst_hdr.offsets = f_src.header.offsets
    for ed in f_src.header.point_format.extra_dimensions:
        dst_hdr.add_extra_dim(laspy.ExtraBytesParams(name=ed.name, type=ed.dtype))
    dst_hdr.vlrs = [v for v in f_src.header.vlrs if getattr(v, "user_id", "") != "copc"]

    p2_ptr = 0
    total_written = 0

    with laspy.open(combined_dst_path, mode="w", header=dst_hdr) as writer:
        for chunk_idx, chunk in enumerate(f_src.chunk_iterator(10_000_000), start=1):
            t_chk = time.time()
            pts_count = len(chunk)
            ud = chunk.user_data

            is_p2 = (ud == 2)
            n_p2_chunk = int(np.sum(is_p2))
            if n_p2_chunk > 0:
                p2_slice = slice(p2_ptr, p2_ptr + n_p2_chunk)

                # Assign aligned properties
                chunk.itc[is_p2] = aligned_final_itc[p2_slice]
                chunk.tree_id[is_p2] = aligned_final_tid[p2_slice]
                chunk.user_data[is_p2] = aligned_final_ud[p2_slice]
                chunk.tree_score[is_p2] = aligned_score[p2_slice]
                chunk.semantic_pred[is_p2] = aligned_sem[p2_slice]
                chunk.classification[is_p2] = aligned_cls[p2_slice]

                p2_ptr += n_p2_chunk

            writer.write_points(chunk)
            total_written += pts_count
            pct = (total_written / total_pts) * 100.0
            dt = time.time() - t_chk
            print(f"  [Chunk {chunk_idx:02d}] {total_written:,}/{total_pts:,} ({pct:5.1f}%) in {dt:.1f}s | p2 mapped: {p2_ptr:,}/{n_p2:,}", flush=True)

    if p2_ptr != n_p2:
        raise RuntimeError(f"Pass 2 point alignment mismatch: mapped {p2_ptr} of {n_p2} points!")

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"SPATIAL REALIGNMENT COMPLETE in {elapsed:.1f}s")
    print(f"Total points written: {total_written:,}")
    print(f"Output saved -> {combined_dst_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_spatial_alignment()
