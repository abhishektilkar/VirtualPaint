import cv2
# our hand tracking module which
# we have implemented using google media-pipe
import module as htm
# imorting numpy
import numpy as np

# hardcoded thickness we will be use to paint
# drawing thickness
brushSize = 20

# erasing thickness
eraserSize = 90

# setting blue color at
# by default this color will be set when we start
selectedColor = (255, 0, 0)

# coordinates for drawing
# x previous and y previous value
prex, prey = 0, 0

# defining canvas up
imageCanvas = np.zeros((720, 1280, 3), np.uint8)

# color pallate
header = cv2.imread("IMAGE/1.jpg")

# calling web cam to capture
cap = cv2.VideoCapture(0)

# defining th size of video
# width
cap.set(3, 1280)

# height
cap.set(4, 720)

# making object using hand tracking module
# maxhands property
detector = htm.handlandmarksDetector(detectionCon=0.60, maxHands=1)

while True:

    # Importing image
    success, image = cap.read()

    # here neglecting mirror inversion
    image = cv2.flip(image, 1)

    # getting Hand Landmarks
    image = detector.AdrawHands(image)
    # using functions for connecting landmarks

    # using function to find specific landmark position,draw false means no circles on landmarks
    # geting all hand landmarks in variable lmList up
    lmList, bbox = detector.findPosition(image, draw=False)

    if len(lmList) != 0:
        # printing hand landmarks as it shows up a
        # print(lmList)

        # we will use index finger for painting and both///
        # fingers to pick bruches or eraser when it shows up//

        # tip of index finger
        x1, y1 = lmList[8][1], lmList[8][2]

        # tip of middle finger
        x2, y2 = lmList[12][1], lmList[12][2]

        # 3. Checking which fingers are up
        fingers = detector.fingersOpen()
        # print(fingers)

        # 4. If Selection Mode - Two finger are up
        # index for index and middle in list is 1 and two - Two finger are up
        if fingers[1] and fingers[2]:

            prex, prey = 0, 0

            # our header has 125 height so if y1 is less than  125 we will go

            if y1 < 125:

                # blue
                if 90 < x1 < 180:
                    selectedColor = (255, 0, 0)  # 255

                # if i m clicking at purple color
                elif 307 < x1 < 415:
                    selectedColor = (255, 0, 255)

                    # if i m clicking at red color
                elif 590 < x1 < 700:
                    selectedColor = (0, 0, 255)  # 255

                # if i m clicking at green color
                elif 854 < x1 < 960:
                    selectedColor = (0, 255, 0)

                    # if i m clicking at eraser(white)
                elif 1081 < x1 < 1200:
                    selectedColor = (0, 0, 0)

                    # rectangle to represent selection mode value
            cv2.rectangle(image, (x1, y1 - 25), (x2, y2 + 25), selectedColor, cv2.FILLED)
            # cv2.circle(image, ((x1+x2)//2, (y1+y2)//2), 100, selectedColor, cv2.FILLED)

        # drawing mode
        if fingers[1] and fingers[2] == False:

            # drawing mode is represented as circle
            cv2.circle(image, (x1, y1), 15, selectedColor, cv2.FILLED)
            # print("Drawing Mode")

            # initially prex and prey will be at 0,0 so it will draw a line from 0,0 to whichever point our tip is at
            if prex == 0 and prey == 0:
                # so to avoid that we set prex=x1 and prey=y1
                prex, prey = x1, y1

                # till now we are creating our drawing but it gets removed as everytime our frames are updating so we have
            # to define our canvas where we can draw and show also

            # eraser code up
            if selectedColor == (0, 0, 0):
                cv2.line(image, (prex, prey), (x1, y1), selectedColor, eraserSize)
                cv2.line(imageCanvas, (prex, prey), (x1, y1), selectedColor, eraserSize)
            else:

                # we will gonna draw lines from previous coodinates to new positions x1,y1  prex,prey
                cv2.line(image, (prex, prey), (x1, y1), selectedColor, brushSize)

                # on canvas draw up
                cv2.line(imageCanvas, (prex, prey), (x1, y1), selectedColor, brushSize)

                # Updating values to prex,prey everytime loop
            prex, prey = x1, y1

            # merging two windows into one image canvas and image

    # 1 converting image to gray
    imageGray = cv2.cvtColor(imageCanvas, cv2.COLOR_BGR2GRAY)
    # it will give grayscale in black canvas if called show
    # cv2.imshow("Image", imageGray)

    # 2 converting into binary image and thn inverting from white to
    # color back because of inverse call
    # taking it imageGray

    # canvas ka color black hai aur jo image draw kare hain usko gray me convert kar diye hain imageGray mane se ab
    # binary me convert karne se canvas black rahega par draw kia part white ho jayega aur isko inverse karne pe
    # inke colors bhi inverse ho jayenge

    _, imageInv = cv2.threshold(imageGray, 50, 255, cv2.THRESH_BINARY_INV)
    #
    # # # converting again to BGR we have to and in a BGR image i.e image
    imageInv = cv2.cvtColor(imageInv, cv2.COLOR_GRAY2BGR)
    # #
    # # # and original image with imageInv ,by doing this we get our drawing only in black color
    image = cv2.bitwise_and(image, imageInv)
    #
    # # # add image and imageCanvas,by doing this we get colors on image
    image = cv2.bitwise_or(image, imageCanvas)
    #
    # # # setting the header image
    # # on our frame we are setting our JPG image acc to H,W of jpg images
    image[0:125, 0:1280] = header

    cv2.imshow("Image", image)
    # cv2.imshow("Canvas", imageCanvas)
    # cv2.imshow("Inv", imageInv)
    cv2.waitKey(1)

