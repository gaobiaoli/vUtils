from typing import List


class MultiVideoCapture:
    def __init__(self, capList: List) -> None:
        self.capList = capList

    def read(self):
        frameList = []
        ret = True
        for cap in self.capList:
            subret, frame = cap.read()
            ret = subret & ret
            frameList.append(frame)
        return ret, frameList

    def count(self):
        countList = []
        for cap in self.capList:
            countList.append(cap.count())
        return countList
    
    def setInitStep(self,skipList):
        # TODO 
        assert len(skipList) == len(self.capList)
