import numpy as np
import cv2
from typing import Union
from functools import reduce
class HomographyInit(object):
    """
    通过手动特征点匹配获取初始单应性矩阵
    """

    @staticmethod
    def run(
        src: np.ndarray,
        dst: np.ndarray,
        save_path: Union[None, str] = None,
        show: bool = False,
    ):
        # 备份用于可视化结果的图片
        src_for_show = src.copy()
        dst_for_show = dst.copy()
        # 选取点的列表
        list_src = []
        list_dst = []

        def get_srcpoint(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                list_xy = [x, y]

                cv2.circle(src, (x, y), 2, (255, 0, 0), thickness=-1)
                cv2.putText(
                    src,
                    "%d,%d" % (x, y),
                    (x, y),
                    cv2.FONT_HERSHEY_PLAIN,
                    2.0,
                    (255, 255, 0),
                    thickness=1,
                )
                cv2.imshow("original_img", src)
                list_src.append(list_xy)

        def get_dstpoint(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                list_xy = [x, y]
                cv2.circle(dst, (x, y), 2, (255, 0, 0), thickness=-1)
                cv2.putText(
                    dst,
                    "%d,%d" % (x, y),
                    (x, y),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.0,
                    (255, 255, 0),
                    thickness=1,
                )
                cv2.imshow("target", dst)
                list_dst.append(list_xy)

        cv2.namedWindow("original_img", cv2.WINDOW_NORMAL)
        cv2.namedWindow("target", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("original_img", 1280, 960)
        cv2.resizeWindow("target", 1280, 960)
        cv2.setMouseCallback("original_img", get_srcpoint)
        cv2.setMouseCallback("target", get_dstpoint)
        cv2.imshow("original_img", src)
        cv2.imshow("target", dst)
        cv2.waitKey()
        cv2.destroyAllWindows()
        src = np.float32(list_src)
        dst = np.float32(list_dst)
        if len(src) != len(dst) or len(src) < 4 or len(dst) < 4:
            raise ValueError("选取点有误")
        H, mask = cv2.findHomography(src, dst)
        H = H
        if save_path != None:
            np.save(save_path, H)
            print(f"homography has saved in {save_path}")
        if show:
            HomographyInit.show(src_for_show, dst_for_show, H)
        return H

    @staticmethod
    def show(src, dst, H, alpha=0.7, beta=0.3, borderValue=0):
        Trans = cv2.warpPerspective(
            src, H, dst.shape[:2][::-1], borderValue=borderValue
        )

        show_img = cv2.addWeighted(Trans, alpha, dst, beta, 1)

        target_size = (1280, 960)
        height, width, _ = show_img.shape
        aspect_ratio = width / height
        target_width = int(target_size[1] * aspect_ratio)
        show_img = cv2.resize(show_img, (target_width, target_size[1]))

        cv2.namedWindow("Fusion", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Fusion", show_img.shape[1], show_img.shape[0])
        cv2.imshow("Fusion", show_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return show_img

    

    @staticmethod
    def run_one(
        src: np.ndarray,
        dst:np.ndarray,
        save_path: Union[None, str] = None,
        show: bool = False,
    ):  
        """
        通过手动输入dst坐标进行配准
        """
        src_for_show = src.copy()
        dst_for_show = dst.copy()
        list_src = []
        
        def _input():
            pointStr=input("请输入对应的点坐标(以逗号分割):")
            point=[np.float32(i) for i in pointStr.split(",")]
            return point
        
        def get_srcpoint(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                list_xy = [x, y]

                cv2.circle(src, (x, y), 4, (255, 0, 0), thickness=-1)
                cv2.putText(
                    src,
                    "%d,%d" % (x, y),
                    (x, y),
                    cv2.FONT_HERSHEY_PLAIN,
                    3.0,
                    (255, 255, 255),
                    thickness=2,
                )
                cv2.imshow("original_img", src)
                list_src.append(list_xy)
                # list_dst.append(_input())
        
        cv2.namedWindow("original_img", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("original_img", 1280, 960)
        cv2.setMouseCallback("original_img", get_srcpoint)
        cv2.imshow("original_img", src)
        cv2.waitKey()
        cv2.destroyAllWindows()
        list_dst = [_input() for _ in range(len(list_src))]
        src = np.float32(list_src)
        dst = np.float32(list_dst)
        if len(src) != len(dst) or len(src) < 4 or len(dst) < 4:
            raise ValueError("选取点有误")
        H, mask = cv2.findHomography(src, dst)
        H = H
        if save_path != None:
            np.save(save_path, H)
            print(f"homography has saved in {save_path}")
        if show:
            HomographyInit.show(src_for_show,dst_for_show, H,alpha=0.99, beta=0.01)
        return H
    
    @staticmethod
    def solvePnP(
        targetPoints,
        objectPoints,
        mtx,
        dist,
    ):
        pass