import cv2
import numpy as np
from functools import reduce
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    unary_from_labels,
    create_pairwise_bilateral,
    create_pairwise_gaussian,
    unary_from_softmax,
)


class SegUtils:

    @staticmethod
    def applyColorMap(mat: np.ndarray):
        colors = [
            (0, 0, 0),
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            (0, 128, 128),
            (128, 128, 128),
            (64, 0, 0),
            (192, 0, 0),
            (64, 128, 0),
            (192, 128, 0),
            (64, 0, 128),
            (192, 0, 128),
            (64, 128, 128),
            (192, 128, 128),
            (0, 64, 0),
            (128, 64, 0),
            (0, 192, 0),
            (128, 192, 0),
            (0, 64, 128),
            (128, 64, 12),
        ]
        seg_img = np.zeros((np.shape(mat)[0], np.shape(mat)[1], 3))
        for c in range(np.max(mat)):
            seg_img[:, :, 0] += (mat[:, :] == c) * (colors[c][0])
            seg_img[:, :, 1] += (mat[:, :] == c) * (colors[c][1])
            seg_img[:, :, 2] += (mat[:, :] == c) * (colors[c][2])
        return seg_img.astype("uint8")

    @staticmethod
    def DSFusion(matList):
        k = reduce(lambda x, y: x * y, matList)
        K = 1 - np.sum(k, axis=2)
        return k / np.expand_dims(K, axis=2)

    @staticmethod
    def meanFusion(matList):
        return reduce(lambda x, y: x + y, matList) / len(matList)

    @staticmethod
    def denseCRF(prior_image, matrix, parameter=[1, 3, 20, 30, 10]):

        h = matrix.shape[0]  # 高度
        w = matrix.shape[1]  # 宽度
        matrix = np.transpose(matrix, (2, 0, 1))  # 调整通道

        d = dcrf.DenseCRF2D(w, h, matrix.shape[0])
        U = unary_from_softmax(matrix, clip=1e-20)  # 这里的clip很重要
        U = np.ascontiguousarray(U)  # 返回一个地址连续的数组
        img = np.ascontiguousarray(prior_image)

        d.setUnaryEnergy(U)  # 设置一元势

        d.addPairwiseGaussian(
            sxy=parameter[0], compat=parameter[1]
        )  # 设置二元势中高斯情况的值
        d.addPairwiseBilateral(
            sxy=parameter[2], srgb=parameter[3], rgbim=img, compat=parameter[4]
        )  # 设置二元势众双边情况的值

        Q = d.inference(5)  # 迭代5次推理
        # Q = np.argmax(np.array(Q), axis=0).reshape((h, w)) #返回label
        Q = np.array(Q).reshape((-1, h, w)).transpose((1, 2, 0))  # 返回softmax

        return Q

    @staticmethod
    def denseCRF_from_labelmap(prior_image, label_map, unsure=[0], gt_prob=0.7, parameter=[1, 3, 20, 30, 10], n_labels=None):
        """
        使用 label_map 构建 softmax-like 概率图，并调用 denseCRF 进行精化。
        """
        h, w = label_map.shape
        if n_labels is None:
            n_labels = int(label_map.max()) + 1

        # 构造 softmax 概率矩阵 (H, W, C)
        matrix = np.zeros((h, w, n_labels), dtype=np.float32)

        for cls in range(n_labels):
            mask = label_map == cls
            if cls in unsure:
                matrix[mask, :] = 0.0
            else:
                matrix[mask, :] = (1.0 - gt_prob) / (n_labels - 1)
                matrix[mask, cls] = gt_prob

        # 调用已有 denseCRF 函数
        refined = SegUtils.denseCRF(prior_image, matrix, parameter=parameter)

        return refined.argmax(axis=-1).astype(np.uint8)
    
    @staticmethod
    def preprocess(imgList, prefunc=None, stack=False, postfunc=None):
        """
        BGR2RGB
        normalise
        channel: -1 -> 0
        stack(axis=0)
        """
        if prefunc is None:
            prefunc = lambda x: x / 255
        processedImgs = []
        for img in imgList:
            # rgb
            temp = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(
                np.float64
            )
            # normal
            temp = prefunc(temp)
            # trans
            temp = np.transpose(temp, (2, 0, 1))
            processedImgs.append(temp)
        if stack:
            processedImgs = np.stack(processedImgs, axis=0)
        if postfunc is not None:
            processedImgs = postfunc(processedImgs)
        return processedImgs

    @staticmethod
    def batchWarpPerspective(
        arrayList,
        H_List,
        shape,
        prefunc=None,
        prefunc_item=None,
        postfunc=None,
        postfunc_item=None,
    ):
        assert len(arrayList) == len(H_List)
        if prefunc is not None:
            arrayList = prefunc(arrayList)
        result = []
        for array, H in zip(arrayList, H_List):
            if prefunc_item is not None:
                array = prefunc_item(array)
            array = cv2.warpPerspective(array, H, shape)
            if postfunc_item is not None:
                array = postfunc_item(array)
            result.append(array)
        if postfunc is not None:
            result = postfunc(result)
        return result
