"""Tests for leakage guards, label fidelity and paired geometry; no model training."""
from io import BytesIO
import unittest

import numpy as np
from PIL import Image

from retinal_data.fives import decode_label, paired_crop_transform, read_pair
from retinal_data.prepare import ROOT, build_selection


def png_bytes(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


class DataContractTests(unittest.TestCase):
    def test_selection_is_deterministic_and_group_separated(self):
        _, subsets, _ = build_selection(ROOT / "configs/data_trial_v1.json")
        _, repeated, _ = build_selection(ROOT / "configs/data_trial_v1.json")
        self.assertEqual(subsets, repeated)
        self.assertEqual(len(subsets["smoke_train"]), 32)
        self.assertEqual(len(subsets["smoke_val"]), 8)
        self.assertEqual(len(subsets["edge_cases"]), 20)
        train = {r["group_id"] for r in subsets["smoke_train"]}
        val = {r["group_id"] for r in subsets["smoke_val"]}
        self.assertFalse(train & val)
        self.assertTrue({r["key"] for r in subsets["overfit"]} <= {r["key"] for r in subsets["smoke_train"]})
        self.assertTrue(all(r["trial_split"] == "edge_only" for r in subsets["all_train"]
                            if r["group_size"] > 1 or r["label_is_empty"]))
        self.assertTrue(all(r["source_split"] == "train" for r in subsets["all_train"]))

    def test_all_geometries_preserve_pixel_alignment_and_are_invertible(self):
        yy, xx = np.indices((13, 17))
        mask = ((xx + 2 * yy) % 5 == 0).astype(np.uint8)
        image = np.stack((mask * 255, xx, yy), axis=-1).astype(np.uint8)
        original = image.copy()
        for h in (False, True):
            for v in (False, True):
                for turns in range(4):
                    transformed, label = paired_crop_transform(image, mask, top=2, left=3, size=7,
                                                               hflip=h, vflip=v, quarter_turns=turns)
                    self.assertTrue(np.array_equal(transformed[..., 0] // 255, label))
                    restored = np.rot90(transformed, -turns)
                    if v:
                        restored = restored[::-1]
                    if h:
                        restored = restored[:, ::-1]
                    self.assertTrue(np.array_equal(restored, original[2:9, 3:10]))
                    self.assertEqual(set(np.unique(label)), {0, 1})
                    self.assertTrue(transformed.flags.c_contiguous and label.flags.c_contiguous)
        self.assertTrue(np.array_equal(image, original))

    def test_empty_rgb_and_opaque_rgba_labels(self):
        self.assertFalse(decode_label(png_bytes(np.zeros((8, 8, 3), dtype=np.uint8))).any())
        label = np.zeros((8, 8, 4), dtype=np.uint8)
        label[..., 3] = 255
        label[3:5, 1:7, :3] = 255
        self.assertEqual(int(decode_label(png_bytes(label)).sum()), 12)

    def test_nonbinary_mismatched_channels_and_transparency_are_rejected(self):
        gray = np.zeros((8, 8, 3), dtype=np.uint8)
        gray[1, 1] = 127
        with self.assertRaisesRegex(ValueError, "0 and 255"):
            decode_label(png_bytes(gray))
        channels = np.zeros((8, 8, 3), dtype=np.uint8)
        channels[1, 1, 0] = 255
        with self.assertRaisesRegex(ValueError, "channels"):
            decode_label(png_bytes(channels))
        alpha = np.zeros((8, 8, 4), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "opaque"):
            decode_label(png_bytes(alpha))

    def test_crop_bounds_and_heldout_access_are_rejected(self):
        with self.assertRaises(ValueError):
            paired_crop_transform(np.zeros((8, 8, 3)), np.zeros((8, 8)), top=0, left=0, size=9)
        with self.assertRaisesRegex(ValueError, "Held-out"):
            read_pair({"source_split": "test"})
        with self.assertRaisesRegex(ValueError, "Unexpected source path"):
            read_pair({"source_split": "train", "filename": "1_A.png", "key": "train/1_A.png",
                       "image_path": "outside.png"})


if __name__ == "__main__":
    unittest.main()
