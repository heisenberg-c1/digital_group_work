"""装载与导出测试：EXIF 方向、透明底压白、Unicode 路径。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from docscan.io.exporter import save_image
from docscan.io.loader import load_image


def _save(tmp_path: Path, im: PILImage.Image, name: str) -> Path:
    p = tmp_path / name
    im.save(p)
    return p


class TestLoadImage:
    @staticmethod
    def _save_with_exif(im: PILImage.Image, path: Path, orientation: int) -> Path:
        exif = im.getexif()
        exif[274] = orientation  # 0x0112 Orientation
        im.save(path, exif=exif.tobytes())
        return path

    def test_exif_orientation_6(self, tmp_path):
        """Orientation=6：存储为竖幅像素，读取时应转正为横幅。"""
        im = PILImage.new("RGB", (40, 80), (255, 0, 0))  # 存储尺寸 w=40, h=80
        p = self._save_with_exif(im, tmp_path / "rot6.jpg", 6)
        out = load_image(p)
        assert out.shape == (40, 80, 3)  # 转正后 numpy 形状 (h=40, w=80)

    def test_exif_orientation_8(self, tmp_path):
        """Orientation=8：需要逆时针转 90°。"""
        im = PILImage.new("RGB", (30, 90), (0, 255, 0))
        p = self._save_with_exif(im, tmp_path / "rot8.jpg", 8)
        out = load_image(p)
        assert out.shape == (30, 90, 3)

    def test_no_exif_untouched(self, tmp_path):
        im = PILImage.new("RGB", (40, 80), (0, 0, 255))
        p = _save(tmp_path, im, "plain.jpg")
        out = load_image(p)
        assert out.shape == (80, 40, 3)

    def test_rgba_composited_on_white(self, tmp_path):
        """透明底 PNG：透明区域必须变成白色，不能变黑。"""
        rgba = np.zeros((50, 50, 4), dtype=np.uint8)
        rgba[:, :, :3] = (10, 200, 30)  # 绿色内容
        p = tmp_path / "alpha.png"
        PILImage.fromarray(rgba).save(p)
        out = load_image(p)
        assert out.shape[2] == 3
        assert tuple(out[0, 0]) == (255, 255, 255)  # BGR 白

    def test_grayscale_to_bgr(self, tmp_path):
        gray = np.full((40, 60), 128, dtype=np.uint8)
        p = _save(tmp_path, PILImage.fromarray(gray), "g.png")
        out = load_image(p)
        assert out.shape == (40, 60, 3)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "nope.jpg")

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "doc.gif"
        p.write_bytes(b"x")
        with pytest.raises(ValueError):
            load_image(p)

    def test_unicode_path(self, tmp_path):
        """中文路径必须可用（这是选 Pillow 而不是 cv2.imread 的原因）。"""
        p = tmp_path / "中文 目录" / "页面.jpg"
        p.parent.mkdir(parents=True)
        PILImage.new("RGB", (20, 30), (255, 255, 255)).save(p)
        out = load_image(p)
        assert out.shape == (30, 20, 3)


class TestSaveImage:
    def test_roundtrip(self, tmp_path):
        img = np.full((30, 40, 3), (50, 100, 200), dtype=np.uint8)
        p = save_image(img, tmp_path / "out.png")
        assert p.exists()
        back = load_image(p)
        np.testing.assert_array_equal(back, img)

    def test_binary_gray(self, tmp_path):
        img = np.zeros((20, 20), dtype=np.uint8)
        img[5:10, 5:10] = 255
        p = save_image(img, tmp_path / "bw.png")
        back = load_image(p)
        assert back.ndim == 3
        assert back[7, 7, 0] == 255
