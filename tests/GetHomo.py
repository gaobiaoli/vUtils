import sys
import os
import cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from vUtils.utils.RegisUtils import HomographyInit

if __name__ == "__main__":
    # 鼠标左键选取点，esc退出后自动计算单应性矩阵
    src = cv2.imread(os.path.join(ROOT,"tests/data/test_img.jpg"))
    dst = cv2.imread(os.path.join(ROOT,"tests/data/test_bim.png"))
    HomographyInit.run(src, dst, save_path=os.path.join(ROOT,"tests/data/H.npy"), show=True)