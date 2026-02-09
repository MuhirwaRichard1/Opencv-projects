from src.utils import Helper

a = Helper.xyxy_2_xywh(x1=100, y1=100, x2= 400, y2 = 500)

print(a)

b = Helper.xywh_2_xyxy(x_min=100, y_min=100, width=300, height=400)

print(b)

c = Helper.xyxy_2_xcycwh(x1=100, y1=100, x2= 400, y2= 500)

print(c)

d = Helper.xywh_2_xcycwh(x_min=100, y_min=100, width=300, height=400)

print(d)