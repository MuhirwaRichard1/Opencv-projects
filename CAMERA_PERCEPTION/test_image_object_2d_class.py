from src.object_detector import ObjectCategory, ImageObject2d, ImageObject2dList
from src.utils import TimeStamp

bbox2d = ImageObject2d(x_min= 100,
                     y_min= 200,
                     width= 500,
                     height= 300,
                     object_category= ObjectCategory.CAR,
                     confidence= 80)

bbox2d2 = ImageObject2d(x_min= 100,
                     y_min= 300,
                     width= 200,
                     height= 300,
                     object_category= ObjectCategory.PERSON,
                     confidence= 90)

bbox2d3 = ImageObject2d(x_min= 100,
                     y_min= 200,
                     width= 500,
                     height= 300,
                     object_category= ObjectCategory.TRUCK,
                     confidence= 70)
# print(bbox2d)
# print(bbox2d2)
# print(bbox2d3)

objects = ImageObject2dList(time_stamp= TimeStamp(epoch_secs=1594815132, epoch_milli_secs=98),
                            frame_id= 1, image_objects_2d=[bbox2d, bbox2d2, bbox2d3])

print(objects)