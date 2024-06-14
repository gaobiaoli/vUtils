import cv2
import numpy as np
from typing import Union
from .FasterVideoCapture import FasterVideoCapture

class VibrationCalibrator:
    """实现newFrame向baseImg的单应性变换"""

    def __init__(self, baseImg=None, baseHomography=None):
        self.H_old2base = baseHomography  # 如果第一张图片无法配准，需要提供初始的单应性矩阵
        self.baseImg = baseImg  # 基准图片
        self.oldImg = baseImg

        self.detector = cv2.ORB_create(nfeatures=30000)
        self.threshold = 20000  # 特征点小于阈值则跳过
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.average_displacement_threshold = 20.0  # 设置平均位移阈值

    def getHomography(self):
        return self.H_old2base

    def getFeaturePoint(self, img):
        
        kp, des = self.detector.detectAndCompute(img, None)
        return kp, des

    def calHomography(self, old_img, new_img):
        """Find a Homography Matrix
        transfer new Image to old Image"""
        kp1, des1 = self.getFeaturePoint(old_img)
        kp2, des2 = self.getFeaturePoint(new_img)

        if len(kp2) < self.threshold:
            print("图像错误：跳过")
            return False, self.getHomography()
        
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append(m)
        old_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        new_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(new_pts, old_pts, cv2.RANSAC, 5.0)
        inliers_new_pts = new_pts[mask.ravel() == 1]
        inliers_old_pts = old_pts[mask.ravel() == 1]

        # 变换内点 new_pts 到 old_pts 的坐标系
        inliers_new_pts_transformed = cv2.perspectiveTransform(inliers_new_pts, H)

        # 计算所有有效点的欧几里得距离
        distances = np.linalg.norm(inliers_new_pts_transformed - inliers_old_pts, axis=1)

        # 计算平均位移
        average_displacement = np.mean(distances)

        # 检查平均位移是否过大
        if average_displacement > self.average_displacement_threshold:
            print(average_displacement)
            print("Transformation invalid due to large displacement, skipping homography.")
            return False, self.getHomography()
        return True, H

    def calibrate(self, newFrame):
        if self.H_old2base is None:
            ret, self.H_old2base = self.calHomography(
                old_img=self.baseImg, new_img=newFrame
            )
        else:
            ret, H_new2old = self.calHomography(old_img=self.oldImg, new_img=newFrame)
            if ret:
                self.H_old2base = np.dot(H_new2old, self.H_old2base)
        if ret:
            self.oldImg = newFrame
        return self.getHomography()

class DeVibVideoCapture(FasterVideoCapture):
    def __init__(
        self,
        videoPath: str,
        initStep: int = 0,
        interval: int = 0,
        mtx: Union[None, np.array] = None,
        dist: Union[None, np.array] = None,
        calibrator: Union[None, VibrationCalibrator] = None,
    ) -> None:
        super().__init__(videoPath=videoPath, interval=interval,initStep=initStep, mtx=mtx, dist=dist)
        self.calibrator = calibrator

    def read(self):
        """间隔interval帧读取"""
        ret,frame = super().read()
        if not ret:
            return False, None
        if self.calibrator is not None:
            Homography = self.calibrator.calibrate(frame)
            frame = cv2.warpPerspective(
                frame, Homography, frame.shape[:2][::-1], borderValue=0
            )
        return ret, frame