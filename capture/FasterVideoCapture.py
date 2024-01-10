import cv2
import threading
import queue
from typing import Union, List
import numpy as np


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
            self.capture = cv2.VideoCapture(self.videoPath)
        else:
            self.capture = cv2.VideoCapture(self.videoPath.pop(0))
        self._capture_count = 0
        self.initStep = initStep
        self.skip(step=initStep)
        self.interval = interval

    def skip(self, step):
        """跳过step帧"""
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
        elif isinstance(self.videoPath, list) and len(self.videoPath):
            self._update(self.videoPath.pop(0))
            ret = self._grab()
        return ret

    def _read(self):
        """跳帧+读帧"""
        ret, frame = self.capture.read()
        if ret:
            self._capture_count += 1
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
        self.capture = cv2.VideoCapture(newVideoPath)

    def release(self):
        self.capture.release()

    def count(self):
        return self._capture_count


class FasterVideoCapture(BaseVideoCapture):
    VIDEO_END_FLAG = -1

    def __init__(
        self,
        videoPath: Union[List[str], str],
        initStep: int = 0,
        mtx: Union[None, np.array] = None,
        dist: Union[None, np.array] = None,
        interval: int = 1,
        buffer_size: int = 5,
    ) -> None:
        super().__init__(videoPath=videoPath, initStep=initStep, mtx=mtx, dist=dist,interval=interval)
        self.interval = interval
        self.buffer_size = buffer_size
        self._read_count = initStep
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.read_thread = threading.Thread(target=self._preload_frames, daemon=True)
        self.read_thread.start()

    def _preload_frames(self):
        while not self.stop_event.is_set():
            ret, frame = super().read()
            if ret:
                self.frame_buffer.put((self._capture_count, frame), block=True)
            else:
                self.frame_buffer.put(
                    (FasterVideoCapture.VIDEO_END_FLAG, None), block=True
                )

    def read(self):
        count, frame = self.frame_buffer.get(block=True)
        if count == FasterVideoCapture.VIDEO_END_FLAG:
            return False, None
        self._read_count += self.interval
        assert count == self._read_count
        if self.mtx is not None and self.dist is not None:
            frame = cv2.undistort(frame, self.mtx, self.dist)
        return True, frame

    def release(self):
        self.stop_event.set()
        self.read_thread.join()
        self.cap.release()

    def count(self):
        return self._read_count
