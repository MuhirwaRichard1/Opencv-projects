import cv2
import numpy as np

def draw_dashed_line(img, start_point, end_point, color, thickness=1, dash_length=10, gap_length=200):

    # Draws a dashed line between two points 
    distance = np.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
    num_dashes = int(distance / (dash_length + gap_length))
    
    if num_dashes == 0:
        cv2.line(img, start_point, end_point, color, thickness)
        return
    
    for i in range(num_dashes):
        # Calculate start and end points for the current dash
        start_r = i * (dash_length + gap_length) / distance
        end_r = (i * (dash_length + gap_length) + dash_length) / distance

        # Ensure the end point does not exceed the original end point
        if end_r > 1.0:
            end_r = 1.0

        # start_pt = (
        #     int(start_point[0] + (end_point[0] - start_point[0]) * start_r),
        #     int(start_point[1] + (end_point[1] - start_point[1]) * start_r)
        # )

        start_pt = (
        int(start_point[0] * (1 - start_r) + end_point[0] * start_r),
        int(start_point[1] * (1 - start_r) + end_point[1] * start_r)
        )

        end_pt = (
            int(start_point[0] * (1 - end_r) + end_point[0] * end_r),
            int(start_point[1] * (1 - end_r) + end_point[1] * end_r)
        )

        cv2.line(img, start_pt, end_pt, color, thickness)

def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_length=10, gap_length=10):
    # Draws a dashed rectangle using four dashed lines

    # define the four corners of the rectangle
    top_left = pt1
    top_right = (pt2[0], pt1[1])
    bottom_right = pt2
    bottom_left = (pt1[0], pt2[1])

    # Draw each side of the rectangle
    draw_dashed_line(img, top_left, top_right, color, thickness, dash_length, gap_length)
    draw_dashed_line(img, top_right, bottom_right, color, thickness, dash_length, gap_length)
    draw_dashed_line(img, bottom_right, bottom_left, color, thickness, dash_length, gap_length)
    draw_dashed_line(img, bottom_left, top_left, color, thickness, dash_length, gap_length)
   

# example usage
if __name__ == "__main__":
    # create a blank white image
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    # image = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # Define rectangle points and parameters
    start_point = (100, 100)
    end_point = (400, 400)
    color = (0, 255, 0)  # Green color (BGR format)
    thickness = 2
    dash_length = 40
    gap_length = 40

    # Draw the dashed rectangle
    draw_dashed_rect(image, start_point, end_point, color, thickness, dash_length, gap_length)

    # Display the result
    cv2.imshow("Dashed Rectangle", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()