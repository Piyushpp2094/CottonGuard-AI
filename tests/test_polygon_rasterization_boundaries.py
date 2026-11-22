"""
Unit Tests for CVAT Polygon Coordinate Parsing, Rounding Policy,
and Boundary Rasterization Traceability.

Traceability Breakdown:
1. Production-Path Tests: They import and exercise the actual production parse_points_string()
   function and apply cv2.fillPoly using the same rasterization operation as the production export pipeline.
   (Note: These test the core parsing and rasterization operations; they do not execute the entire
   process_cvat_xml_export() pipeline end-to-end).
2. Rounding-Policy Tests: Verify np.round(pts).astype(np.int32) behavior on continuous float
   coordinates (100.49, 100.50, 101.50, 100.51, 800.0).
   NumPy uses deterministic round-to-nearest-even behavior for exact halfway values. This avoids always
   resolving exact ties upward. However, it does not guarantee zero geometric bias for every polygon or
   dataset. Rasterized polygon areas remain subject to quantization effects.
   For coordinates represented exactly as float32, values x >= 799.5 round to 800 under nearest-even rounding
   because 800 is even. The same applies to y >= 799.5.
3. Independent OpenCV Behavior Tests: Quantify geometric distortion caused by arbitrary
   clipping to W - 1 vs unclipped OpenCV scanline clipping.
"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_cvat_annotations import parse_points_string


def production_rasterize_polygon(points_str: str, canvas_shape=(800, 800)) -> np.ndarray:
    """
    Executes the exact production parsing and rasterization pipeline from
    `scripts/prepare_cvat_annotations.py` (lines 337, 343, 370).
    """
    pts = parse_points_string(points_str)
    mask = np.zeros(canvas_shape, dtype=np.uint8)
    if pts.shape[0] >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask


class TestProductionPipelineRasterization(unittest.TestCase):
    """
    Production-Path Tests: They import and exercise the actual production parse_points_string()
    function and apply cv2.fillPoly using the same rasterization operation as the production export pipeline.
    Tested on all 7 synthetic boundary configurations.
    """

    def setUp(self):
        self.H = 800
        self.W = 800

    def test_01_prod_polygon_completely_inside(self):
        """Production pipeline: Polygon fully interior to image."""
        pts_str = "100.0,100.0;300.0,100.0;300.0,300.0;100.0,300.0"
        mask = production_rasterize_polygon(pts_str, (self.H, self.W))
        self.assertEqual(mask.shape, (800, 800))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask[200, 200], 1)
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[799, 799], 0)
        self.assertEqual(mask[:, 799].sum(), 0)
        self.assertEqual(mask[799, :].sum(), 0)

    def test_02_prod_polygon_touching_right_boundary_x800(self):
        """Production pipeline: Polygon with outer border touching x = 800.0."""
        pts_str = "700.0,100.0;800.0,100.0;800.0,200.0;700.0,200.0"
        mask = production_rasterize_polygon(pts_str, (self.H, self.W))
        self.assertEqual(mask.shape, (800, 800))
        # Outer boundary pixel column (index 799) must be fully rendered
        self.assertEqual(mask[150, 799], 1)
        self.assertEqual(mask[100:201, 799].sum(), 101)
        # Left boundary untouched
        self.assertEqual(mask[:, 0].sum(), 0)

    def test_03_prod_polygon_touching_bottom_boundary_y800(self):
        """Production pipeline: Polygon with outer border touching y = 800.0."""
        pts_str = "100.0,700.0;200.0,700.0;200.0,800.0;100.0,800.0"
        mask = production_rasterize_polygon(pts_str, (self.H, self.W))
        self.assertEqual(mask.shape, (800, 800))
        # Outer boundary pixel row (index 799) must be fully rendered
        self.assertEqual(mask[799, 150], 1)
        self.assertEqual(mask[799, 100:201].sum(), 101)

    def test_04_prod_polygon_touching_both_right_and_bottom(self):
        """Production pipeline: Polygon touching both x = 800.0 and y = 800.0."""
        pts_str = "600.0,600.0;800.0,600.0;800.0,800.0;600.0,800.0"
        mask = production_rasterize_polygon(pts_str, (self.H, self.W))
        self.assertEqual(mask.shape, (800, 800))
        # Corner pixel [799, 799] must be filled
        self.assertEqual(mask[799, 799], 1)
        self.assertEqual(mask[650, 799], 1)
        self.assertEqual(mask[799, 650], 1)

    def test_05_prod_polygon_touching_top_left(self):
        """Production pipeline: Polygon touching top-left origin (0.0, 0.0)."""
        pts_str = "0.0,0.0;200.0,0.0;200.0,200.0;0.0,200.0"
        mask = production_rasterize_polygon(pts_str, (self.H, self.W))
        self.assertEqual(mask.shape, (800, 800))
        self.assertEqual(mask[0, 0], 1)
        self.assertEqual(mask[0, 100], 1)
        self.assertEqual(mask[100, 0], 1)

    def test_06_prod_polygon_spanning_complete_canvas(self):
        """Production pipeline: Polygon spanning complete canvas [0,0] to [800,800]."""
        pts_str = "0.0,0.0;800.0,0.0;800.0,800.0;0.0,800.0"
        mask = production_rasterize_polygon(pts_str, (self.H, self.W))
        self.assertEqual(mask.shape, (800, 800))
        self.assertEqual(mask[0, 0], 1)
        self.assertEqual(mask[0, 799], 1)
        self.assertEqual(mask[799, 0], 1)
        self.assertEqual(mask[799, 799], 1)
        self.assertEqual(mask.sum(), 640000)

    def test_07_prod_polygon_vertex_at_800_800(self):
        """Production pipeline: Triangle with explicit vertex at (800.0, 800.0)."""
        pts_str = "500.0,500.0;800.0,500.0;800.0,800.0"
        mask = production_rasterize_polygon(pts_str, (self.H, self.W))
        self.assertEqual(mask.shape, (800, 800))
        self.assertEqual(mask[799, 799], 1)
        self.assertEqual(mask[500, 799], 1)


class TestProductionRoundingPolicy(unittest.TestCase):
    """
    Verifies the exact rounding behavior of `scripts.prepare_cvat_annotations.parse_points_string`
    which executes `np.round(np.array(pts, dtype=np.float32)).astype(np.int32)`.

    NumPy uses deterministic round-to-nearest-even behavior for exact halfway values. This avoids always
    resolving exact ties upward. However, it does not guarantee zero geometric bias for every polygon or
    dataset. Rasterized polygon areas remain subject to quantization effects.

    For coordinates represented exactly as float32, values x >= 799.5 round to 800 under nearest-even rounding
    because 800 is even. The same applies to y >= 799.5.
    """

    def test_rounding_100_49(self):
        """100.49 rounds strictly down to 100."""
        pts_str = "100.49,200.0;150.0,200.0;150.0,250.0"
        pts = parse_points_string(pts_str)
        self.assertEqual(pts[0, 0], 100)

    def test_rounding_100_50_nearest_even(self):
        """
        100.50 is an exact half-way tie.
        NumPy implements IEEE 754 round-to-nearest-even:
        100 is even, 101 is odd -> rounds to 100.
        """
        pts_str = "100.50,200.0;150.0,200.0;150.0,250.0"
        pts = parse_points_string(pts_str)
        self.assertEqual(pts[0, 0], 100)

    def test_rounding_101_50_nearest_even(self):
        """
        101.50 is an exact half-way tie.
        101 is odd, 102 is even -> rounds to 102.
        Confirms nearest-even tie-breaking without directional bias.
        """
        pts_str = "101.50,200.0;150.0,200.0;150.0,250.0"
        pts = parse_points_string(pts_str)
        self.assertEqual(pts[0, 0], 102)

    def test_rounding_100_51(self):
        """100.51 rounds strictly up to 101."""
        pts_str = "100.51,200.0;150.0,200.0;150.0,250.0"
        pts = parse_points_string(pts_str)
        self.assertEqual(pts[0, 0], 101)

    def test_rounding_boundary_800_0(self):
        """Boundary coordinate 800.0 remains exactly 800 (not truncated to 799)."""
        pts_str = "800.0,800.0;700.0,800.0;700.0,700.0"
        pts = parse_points_string(pts_str)
        self.assertEqual(pts[0, 0], 800)
        self.assertEqual(pts[0, 1], 800)


class TestIndependentOpenCVBehavior(unittest.TestCase):
    """
    Independently isolates and audits OpenCV `cv2.fillPoly` scanline clipping
    and proves why arbitrary vertex clipping to W - 1 introduces geometric distortion.
    """

    def setUp(self):
        self.H = 800
        self.W = 800

    def test_opencv_native_boundary_scanline_clipping(self):
        """
        Verifies that cv2.fillPoly internally clips scanlines against the canvas
        dimensions without needing manual vertex clipping.
        """
        # Outer vertices at 800
        tri = np.array([[600, 200], [800, 400], [600, 600]], dtype=np.int32)
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        # Must execute without out-of-bounds error
        cv2.fillPoly(mask, [tri], 1)
        self.assertEqual(mask.shape, (800, 800))
        # Boundary column 799 receives scanline coverage
        self.assertEqual(mask[:, 799].sum(), 3)

    def test_distortion_audit_unclipped_vs_clipped(self):
        """
        Demonstrates that clipping vertices to (W - 1, H - 1) distorts diagonal geometry
        and causes silent loss of coverage along boundary scanlines.
        """
        tri = np.array([[600, 200], [800, 400], [600, 600]], dtype=np.int32)
        tri_clipped = np.clip(tri, 0, [self.W - 1, self.H - 1])

        m_unclipped = np.zeros((self.H, self.W), dtype=np.uint8)
        m_clipped = np.zeros((self.H, self.W), dtype=np.uint8)

        cv2.fillPoly(m_unclipped, [tri], 1)
        cv2.fillPoly(m_clipped, [tri_clipped], 1)

        # Unclipped version includes 3 pixels in col 799
        self.assertEqual(m_unclipped[:, 799].sum(), 3)
        # Clipped version artificially loses 2 out of 3 border pixels
        self.assertEqual(m_clipped[:, 799].sum(), 1)

        # Quantify total pixel loss caused by altering the slope of the triangle edges
        pixel_loss = int(m_unclipped.sum() - m_clipped.sum())
        self.assertEqual(pixel_loss, 200)
        self.assertGreater(pixel_loss, 0, "Clipping to W-1 caused 200 mask pixels of area difference!")


if __name__ == "__main__":
    unittest.main()
