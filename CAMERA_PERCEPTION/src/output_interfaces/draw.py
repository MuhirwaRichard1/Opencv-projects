import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')

cv.imshow("blank_image", blank)

# print different color
# blank[200:250, 300:350] = 0,255,0
# cv.imshow("Green", blank)

cv.rectangle(blank, (0,0), (400,400), (0, 255,0), thickness=cv.FILLED)
cv.putText(blank, "Opencv", (50, 250), cv.FONT_HERSHEY_SIMPLEX, 5.0, (250, 0, 0), 5)
cv.imshow("rectangle", blank)
cv.waitKey(2000)