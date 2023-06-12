import cv2, os
import numpy as np 

img_save_path = './joaNNE/images/'
# if not 
onDown = False
xprev, yprev = None, None
def onmouse(event, x, y, flags, params):
    global onDown, img, xprev, yprev
    if event == cv2.EVENT_LBUTTONDOWN:
        print('DOWN : {0}, {1}'.format(x,y))
        onDown = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if onDown == True:
            print("MOVE : {0}, {1}".format(x,y))
            # cv2.line(img, (xprev, yprev), (x,y), (255, 255, 255), 10) # 선 흰
            cv2.line(img, (xprev, yprev), (x,y), (0, 0, 0), 10) # 선 검정
    elif event == cv2.EVENT_LBUTTONUP:
        print('UP : {0}, {1}'.format(x,y))
        onDown = False
    xprev, yprev = x,y
    
cv2.namedWindow('image')
cv2.setMouseCallback('image',onmouse)
width, height = 500, 500
# img = np.zeros((width, height, 3), np.uint8) # 배경 검정
img = np.ones((width, height, 3), np.uint8) * 255 # 배경 흰
figNum = 1

existing_files = os.listdir(img_save_path)
if existing_files:
    existing_numbers = [int(file_name[5:7]) for file_name in existing_files if file_name.startswith('image')]
    figNum = max(existing_numbers) + 1
    
while True:
    cv2.imshow('image', img)
    key = cv2.waitKey(1)
    if key == ord('c'): 
        # img = np.zeros((width, height, 3), np.uint8) # 배경 검정
        img = np.ones((width, height, 3), np.uint8) * 255 # 배경 흰
        print('Clear.')
    if key == ord('s'):
        img_save = cv2.resize(img, dsize=(500, 500), interpolation=cv2.INTER_AREA)
        cv2.imwrite("{0}image{1}.jpg".format(img_save_path, str(figNum).zfill(2)), img_save)
        figNum = figNum + 1
        print('Image saved')
    # if key == ord('q'): # q를 눌러야만 실행창이 닫힘 X버튼 눌러도 다시실행됨
    if key == ord('q') or cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1: # X 버튼을 눌러도 닫힘
        print('Good bye')
        break
cv2.destroyAllWindows()
