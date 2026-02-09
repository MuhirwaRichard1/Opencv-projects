import numpy as np


class Helper:
    def __init__(self):
        pass

    @staticmethod
    def xyxy_2_xywh(x1: int, y1: int, x2: int, y2: int):

        x_min = int(x1)
        y_min = int(y1)

        width = int(abs(x2 - x1))
        height = int(abs(y2 - y1))

        return x_min, y_min, width, height
    
    @staticmethod
    def xywh_2_xyxy(x_min: int, y_min: int, width: int, height: int):

        x1 = int(x_min)
        y1 = int(y_min)
        x2 = int(x1 + width)
        y2 = int(y1 + height)

        return x1, y1, x2, y2
    
    @staticmethod
    def xyxy_2_xcycwh(x1: int, y1: int, x2: int, y2: int):

        width = int(abs(x2 - x1))
        height = int(abs(y2 - y1))

        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)

        return x_center, y_center, width, height
    
    @staticmethod
    def xywh_2_xcycwh(x_min: int, y_min: int, width: int, height: int):

        x_center = int(x_min + width/2)
        y_center = int(y_min + height/2)

        return x_center, y_center, width, height