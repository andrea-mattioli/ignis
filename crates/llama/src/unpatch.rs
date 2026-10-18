//! The inverse of `ignis_artifact::vision::patches::pack_patches`: the
//! processor's patch rows back into the normalized image they were cut from.
//!
//! The ignis processor decodes, `smart_resize`s and normalizes an image the
//! model contract's way and packs it into `[raw_patches, 1536]` BF16 rows.
//! llama.cpp's encoder takes the normalized image instead and cuts its own
//! patches (the same 16 px, the frame repeated as the temporal pair, the
//! same 2x2 merge). So the leaf undoes the packing exactly -- every pixel is
//! present once per temporal copy -- and the encoder sees the pixels the
//! processor chose, at the grid the prompt's placeholders were counted for.

/// Side of one patch, in pixels.
pub const PATCH: usize = 16;
/// Frames per temporal patch.
pub const TEMPORAL: usize = 2;
/// Patches per merged token along each axis.
pub const MERGE: usize = 2;
/// Width of one patch row.
pub const PATCH_FEATURES: usize = 3 * TEMPORAL * PATCH * PATCH;

fn from_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// `patches` (row-major `[gh * gw, PATCH_FEATURES]`) of a `gh x gw` patch grid
/// into a `(gw * 16) x (gh * 16)` image, row-major interleaved RGB floats.
/// `None` when the rows do not fill that grid.
pub fn unpatch(patches: &[u16], gh: usize, gw: usize) -> Option<Vec<f32>> {
    if gh % MERGE != 0 || gw % MERGE != 0 || patches.len() != gh * gw * PATCH_FEATURES {
        return None;
    }
    let width = gw * PATCH;
    let mut rgb = vec![0.0f32; gh * PATCH * width * 3];
    let mut at = 0;
    for block_y in 0..gh / MERGE {
        for block_x in 0..gw / MERGE {
            for merge_y in 0..MERGE {
                for merge_x in 0..MERGE {
                    let (grid_y, grid_x) = (block_y * MERGE + merge_y, block_x * MERGE + merge_x);
                    for channel in 0..3 {
                        for temporal in 0..TEMPORAL {
                            for y in 0..PATCH {
                                let row = ((grid_y * PATCH + y) * width + grid_x * PATCH) * 3 + channel;
                                for x in 0..PATCH {
                                    // the frame's second temporal copy carries the same pixels
                                    if temporal == 0 {
                                        rgb[row + x * 3] = from_bf16(patches[at]);
                                    }
                                    at += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Some(rgb)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    /// The processor's packing, restated from `patches.rs` so the inverse is
    /// held to the order it inverts.
    fn pack(rgb8: &[u8], width: usize, height: usize) -> Vec<u16> {
        let (gh, gw) = (height / PATCH, width / PATCH);
        let mut out = Vec::new();
        for block_y in 0..gh / MERGE {
            for block_x in 0..gw / MERGE {
                for merge_y in 0..MERGE {
                    for merge_x in 0..MERGE {
                        let (grid_y, grid_x) = (block_y * MERGE + merge_y, block_x * MERGE + merge_x);
                        for channel in 0..3 {
                            for _ in 0..TEMPORAL {
                                for y in 0..PATCH {
                                    let row = ((grid_y * PATCH + y) * width + grid_x * PATCH) * 3 + channel;
                                    for x in 0..PATCH {
                                        out.push(to_bf16(rgb8[row + x * 3] as f32 / 127.5 - 1.0));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn unpatch_returns_every_pixel_where_the_packing_took_it() {
        let (w, h) = (64, 32);
        let rgb8: Vec<u8> = (0..w * h * 3).map(|i| (i * 37 % 251) as u8).collect();
        let back = unpatch(&pack(&rgb8, w, h), h / PATCH, w / PATCH).unwrap();
        for (i, (&v, &got)) in rgb8.iter().zip(&back).enumerate() {
            assert_eq!(got, from_bf16(to_bf16(v as f32 / 127.5 - 1.0)), "pixel component {i}");
        }
    }

    #[test]
    fn rows_that_do_not_fill_the_grid_are_refused() {
        assert!(unpatch(&[0; PATCH_FEATURES * 3], 2, 2).is_none());
        assert!(unpatch(&[0; PATCH_FEATURES * 3], 1, 3).is_none());
    }
}
