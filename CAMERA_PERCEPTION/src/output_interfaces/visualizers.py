from src.image_preprocess import Image
from src.utils import TimeStamp

import cv2

class Visualizer:
    def __init__(elf):
        pass

    @staticmethod
    def show_image(image: Image, print_frame_id_on_image: bool = False, print_time_on_image: bool = False, wait_time_msec: int = 50):

        if not isinstance(image, Image):
            raise TypeError("Input image must be of type Image() only")

        if print_time_on_image:
            label_font_type1 = cv2.FONT_HERSHEY_SIMPLEX
            x2 = 20
            y2 = 30

            time_1 = TimeStamp(epoch_secs=image.time_stamp.epoch_secs, epoch_milli_secs=image.time_stamp.epoch_milli_secs)
            time_and_date_label = f'time= {time_1.date_time}s'

            label_size1, _ = cv2.getTextSize(time_and_date_label, label_font_type1, 0.5, 2)
            # Draw filled rectangle for label background
            cv2.rectangle(image.data, (x2, y2 - (label_size1[1] + 10)), (x2 + label_size1[0]+90, y2), (0, 0, 0), cv2.FILLED)
            # Draw label text
            cv2.putText(image.data, time_and_date_label, (x2, y2 - 5), label_font_type1, 0.7, (255, 255, 255), 2)


        if print_frame_id_on_image:
            label_font_type = cv2.FONT_HERSHEY_SIMPLEX
            x1 = 1080
            y1 = 30

            label = f'frame id = {image.frame_id}'
           
            # get label size
            label_size, _ = cv2.getTextSize(label, label_font_type, 0.5, 2)

            # Draw filled rectangle for label background
            cv2.rectangle(image.data, (x1, y1 - (label_size[1] + 10)), (x1 + label_size[0]+50, y1), (0, 0, 0), cv2.FILLED)

            # Draw label text
            cv2.putText(image.data, label, (x1, y1 - 5), label_font_type, 0.7, (255, 255, 255), 2)
            # cv2.putText(image.data, time_and_date_label, (x1, y1 + 15), label_font_type, 0.7, (255, 0, 0), 2)

        cv2.imshow("image", image.data)
        cv2.waitKey(wait_time_msec)

