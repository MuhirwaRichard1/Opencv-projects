from src.image_preprocess import Image, ImageType
import numpy as np
from src.utils import TimeStamp

img = Image(frame_id=1,
            time_stamp=TimeStamp(epoch_secs=1594815132, epoch_milli_secs=98),
            data= np.zeros((100,300,1)))

print(f"Image Type: {img.image_type}")
print(f"Is RGB?: {img.image_type == ImageType.RGB_IMAGE}")
print(f"Is GRAYSCALE?: {img.image_type == ImageType.GRAYSCALE_IMAGE}")
print(img.height)
print(img.width)
print(img.channels)

# print(img)
