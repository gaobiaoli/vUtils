import sys
import os
import cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from vUtils.utils.RegisUtils import HomographyInit

if __name__ == "__main__":
    src = cv2.imread(os.path.join(ROOT,"tests/data/test_img.jpg"))
    dst = cv2.imread(os.path.join(ROOT,"tests/data/test_bim.png"))
    HomographyInit.run(src, dst, save_path=None, show=True)