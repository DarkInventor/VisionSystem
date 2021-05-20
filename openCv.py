import cv2

video=cv2.VideoCapture(0)

check,frame = video.read(0)

print(check)
print(frame)

video.release()

