import cv2
from typing import Union, List
import numpy as np
import os

class BaseVideoCapture:
    """带畸变校正和间隔读取的视频播放器"""

    def __init__(
        self,
        videoPath: Union[List[str], str],
        initStep: int = 0,
        mtx: Union[None, np.array] = None,
        dist: Union[None, np.array] = None,
        interval: int = 1,
    ) -> None:
        self.videoPath = videoPath
        self.mtx = mtx
        self.dist = dist
        if isinstance(videoPath, str):
            self.name = self.videoPath
            self.capture = cv2.VideoCapture(self.videoPath)
        else:
            self.name = self.videoPath[0]
            self.capture = cv2.VideoCapture(self.videoPath.pop(0))
        self._capture_count = 0
        self._capture_count_video = 0
        self.initStep = initStep
        self.skip(step=initStep)
        self.interval = interval
        
    def skip(self, step):
        """跳过step帧"""
        if step<0:
            return True
        for _ in range(step):
            ret = self._grab()
            if not ret:
                return False
        return True

    def _grab(self):
        """跳帧"""
        ret = self.capture.grab()
        if ret:
            self._capture_count += 1
            self._capture_count_video += 1
        elif isinstance(self.videoPath, list) and len(self.videoPath):
            self._update(self.videoPath.pop(0))
            ret = self._grab()
        return ret

    def _read(self):
        """读帧"""
        ret, frame = self.capture.read()
        if ret:
            self._capture_count += 1
            self._capture_count_video += 1
        elif isinstance(self.videoPath, list) and len(self.videoPath):
            self.capture.release()
            self._update(self.videoPath.pop(0))
            ret, frame = self._read()
        return ret, frame

    def read(self, interval=None):
        """间隔interval帧读取"""
        if not interval:
            interval = self.interval

        ret = self.skip(interval - 1)
        if not ret:
            return False, None

        ret, frame = self._read()
        if not ret:
            return False, None

        if self.mtx is not None and self.dist is not None:
            frame = cv2.undistort(frame, self.mtx, self.dist)
        return ret, frame

    def _update(self, newVideoPath):
        print(f"Update Capture to {newVideoPath}")
        self.name = newVideoPath
        self.capture = cv2.VideoCapture(newVideoPath)
        self._capture_count_video = 0

    def release(self):
        self.capture.release()

    def count(self):
        return self._capture_count
    
    def getVideoID(self):
        return os.path.basename(self.name).split('.')[0]
    
    def getVideoFrameCount(self):
        return self._capture_count_video