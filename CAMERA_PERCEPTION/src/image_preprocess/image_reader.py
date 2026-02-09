from src.image_preprocess import Image
from src.utils import TimeStamp, CameraPosition, CameraLensType
from pathlib import Path
import os
import cv2

class ImageReader:
    allowed_extensions = [".png", ".jpg"]

    def __init__(self, camera_position: CameraPosition, camera_lens_type: CameraLensType):
        self.camera_position = camera_position
        self.camera_lens_type = camera_lens_type

    # getter and setter for camera_lens_type
    @property
    def camera_lens_type(self):
        return self._camera_lens_type
    
    @camera_lens_type.setter
    def camera_lens_type(self, value):
        if not isinstance(value, CameraLensType):
            raise TypeError("Invalid Camera lens type given. It must be of type CameraLensType Enum")
        self._camera_lens_type = value
    
    # getter camera_position and setter camera_position
    @property
    def camera_position(self):
        return self._camera_position
    
    @camera_position.setter
    def camera_position(self, value):
        if not isinstance(value, CameraPosition):
            raise TypeError("Invalid Camera position given. It must be of type CameraPosition Enum")
        self._camera_position = value

    @staticmethod
    def read_image_from_disk(image_file_path: Path) -> Image:
        if not isinstance(image_file_path, Path):
            raise TypeError("File Path must be of type Path()")
        if not os.path.exists(image_file_path):
            raise FileNotFoundError("Image file path given does not exist on the disk")
        if image_file_path.suffix not in ImageReader.allowed_extensions:
            raise ValueError(f"Image file extension must be one of the following: {ImageReader.allowed_extensions}")
        
        # read image using cv2
        image_data = cv2.imread(str(image_file_path))

        image_file_name_without_extension = image_file_path.stem

        split_str = image_file_name_without_extension.split("__")

        frame_id = int(split_str[0])
        date_time_str = split_str[1]

        epoch_seconds = int(int(date_time_str)/1000)
        epoch_milliseconds = int(int(date_time_str)%1000)

        return Image(frame_id= frame_id,
                     time_stamp = TimeStamp(epoch_secs=epoch_seconds, epoch_milli_secs=epoch_milliseconds),
                     data=image_data)
    
    