import cv2
import numpy as np



def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        if (cnt[0][0]>4 and cnt[0][1]>4):
                            squares.append(cnt)
    return squares

def liczenie(crop_img):

    while(True):
        #crop_img = cv2.imread("6.png",1)
        #gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lower_white = np.array([0,0,150], dtype=np.uint8)
        upper_white = np.array([200,100,255], dtype=np.uint8)
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, lower_white, upper_white)
        #img = cv2.erode(white, None, 1)
        #img1 = cv2.dilate(img,None,1)
        #mask = cv2.dilate(white,None,1)
        mask=white
        # Bitwise-AND mask and original image
        #res = cv2.bitwise_and(crop, crop, mask= mask)
        cv2.imshow('Kostka', mask)
        cv2.moveWindow('Kostka', 2200, 0)
        #image = cv2.Canny(res,50,50)
        #ret,thresh = cv2.threshold(imgg,200,200,0)

        contours ,  hierarchy  =  cv2 . findContours ( mask , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE )
        #cv2.drawContours(crop, contours, -1, (0,255,0), 1)
        liczba=0
        for i in contours[1:]:
            (x,y),radius = cv2.minEnclosingCircle(i)
            center = (int(x),int(y))
            radius = int(radius)
            #print(radius)
            #cv2.circle(crop_img,center,radius,(0,255,0),2)
            liczba=liczba+1

        #cv2.imshow("cropped", crop_img)
        #cv2.moveWindow('cropped', 1500, 0)
        return liczba


def wykrywanie_kwadratowych_kostek():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()

        squares = find_squares(frame)
        x=0
        for sq in squares:
            print(sq)
            ix= np.min([sq[0][0],sq[1][0],sq[2][0],sq[3][0]])
            iy= np.min([sq[0][1],sq[1][1],sq[2][1],sq[3][1]])
            ix1=np.max([sq[0][0],sq[1][0],sq[2][0],sq[3][0]])
            iy1=np.max([sq[0][1],sq[1][1],sq[2][1],sq[3][1]])
            crop_img = frame[iy:iy1, ix:ix1]
            x = liczenie(crop_img)
        cv2.drawContours( frame, squares, -1, (0, 255, 0), 2 )

        cv2.putText(frame,str(x),(10,50), cv2.FONT_HERSHEY_DUPLEX, 1,(0,255,0),2,)
        # Display the resulting frame
        cv2.imshow('Wykrywanie liczby oczek na kostce',frame)
        cv2.moveWindow('Wykrywanie liczby oczek na kostce', 1500, 200)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    wykrywanie_kwadratowych_kostek()
    #liczenie(0)
    print("END")