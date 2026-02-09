from src.utils import TimeStamp
import numpy as np
from enum import Enum

class ImageType(Enum):
    RGB_IMAGE = 0
    GRAYSCALE_IMAGE = 1

class Image:
    def __init__(self,
                 frame_id: int,
                 time_stamp: TimeStamp,
                 data: np.ndarray):
        
        self.frame_id = frame_id
        self.time_stamp = time_stamp
        self.data = data
        # self._image_type = image_type

        self._width = None
        self._height = None
        self._channels = None
        self._image_type = None

    @property
    def frame_id(self):
        return self._frame_id
    
    @frame_id.setter
    def frame_id(self, value):
        if not isinstance(value, int):
            raise TypeError("Frame ID of the image must be int only")
        if not value > 0:
            raise ValueError("Frame ID must be positive value > 0")
        self._frame_id = value

    @property
    def time_stamp(self):
        return self._time_stamp
    
    @time_stamp.setter
    def time_stamp(self, value):
        if not isinstance(value, TimeStamp):
            raise TypeError("Time stamp of the image must be TimeStamp object only")
        self._time_stamp = value

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("Image data is invalid. It must be 2D numpy array with one or more channels")
        self._data = value

        # set other properties to none when new image is set
        self._width = None
        self._height = None
        self._channels = None
        self._image_type = None


    
    # only getter
    @property
    def width(self):
        if self._width is None:
            _, self._width, _ = self._data.shape
        return self._width
    
    @property
    def height(self):
        if self._height is None:
            self._height, _, _ = self._data.shape
        return self._height
    
    @property
    def channels(self):
        if self._channels is None:
            _, _, self._channels = self._data.shape
        return self._channels
    
     # only getter
    @property
    def image_type(self):
        if self.channels == 3:
            self._image_type = ImageType.RGB_IMAGE
            return ImageType.RGB_IMAGE
        elif self.channels == 1:
            self._image_type = ImageType.GRAYSCALE_IMAGE
            return ImageType.GRAYSCALE_IMAGE
        
    def __repr__(self) -> str:
        out = "---- Image----\n" \
            + "Frame ID: " + str(self.frame_id) + "\n" \
            + "size: [" + str(self.height) + ", " + str(self.width) + ", " + str(self.channels) + "]\n" \
            + "Time Stamp: " + str(self.time_stamp) \
            + "values: " + str(self.data) + "\n"
        return out
    