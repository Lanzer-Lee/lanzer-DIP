import cv2 

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        x_str = str(x)
        y_str = str(y)
        f = open("data\coordinate.txt", "a+")
        f.writelines(x_str + ' ' + y_str + '\n')

#maxsize = (928,696)    #B
maxsize=(1216,912)      #A
img = cv2.imread('Image A.jpg')
print(img.shape)
img = cv2.resize(img, maxsize, cv2.INTER_AREA)
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)

while (True):
    try:
        cv2.waitKey(100)
    except Exception:
        cv2.destroyAllWindows()
        break

cv2.waitKey(0)
cv2.destroyAllWindows()

