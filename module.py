import cv2
import mediapipe as mp

class handlandmarksDetector():

    def __init__(self,mode=False, maxHands = 2 , detectionCon = 0.5 , trackCon = 0.5 ): # constructor
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands=mp.solutions.hands
        # initializing hands module for the instance

        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        # object for Hands for a particular instance

        # object for Drawing
        self.mpDraw = mp.solutions.drawing_utils

        # this is the tip of fingers up
        # Thumb and 4 fingers total 5 TiP
        self.tipIds = [4, 8, 12, 16, 20]

    def AdrawHands(self,img,draw=True):

        # BGR to RGB
        # As it works only on in rgb imaGE
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        # Processing Now ImaGE
        self.results=self.hands.process(imgRGB)

        # gives x,y,z of every landmark or if no hand than NONE
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks: # each hand landmarks in results
                if draw:

                    # joining points on our hand all
                     self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)

        return img


########################################################################################################################


    def findPosition(self,img,handNo=0,draw=True):

        xList = []

        yList = []

        bbox=[]
        self.lmlist=[]

        # gives x,y,z of landmark we got
        if self.results.multi_hand_landmarks:

            # Gives result for particular hand
            myHand = self.results.multi_hand_landmarks[handNo]

            # gives id and lm(x,y,z)
            for id,lm in enumerate(myHand.landmark):

                # getting h,w for converting decimals x,y into pixels
                h,w,c=img.shape

                # pixels coordinates for landmarks
                cx,cy=int(lm.x*w),int(lm.y*h)

                # print(id, cx, cy)
                xList.append(cx)
                yList.append(cy)
                self.lmlist.append([id,cx,cy])

                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

            xmin,xmax=min(xList),max(xList)
            ymin,ymax=min(yList),max(yList)
            bbox=xmin,ymin,xmax,ymax

            if draw:
                cv2.rectangle(img,(bbox[0]-20,bbox[1]-20),(bbox[2]+20,bbox[3]+20),(0,255,0),2)

        return self.lmlist,bbox


########################################################################################################################


    # checking which finger is open
    def fingersOpen(self):

        # storing final result
        fingers = []

        # checking thumbs tip is in right or left if open 1 else push 0 value

        # Thumb < sign only when  we use flip function to avoid mirror inversion else > sign

        # checking x position of 4 is in right to x position of 3
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1]:

            # open push 1 value
            fingers.append(1)
        else:

            # else push 0 value
            fingers.append(0)

        # we will check tip is in top up
        # Fingers
        # checking tip point is below tippoint-2 (only in Y direction)
        for id in range(1, 5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # totalFingers = fingers.count(1)
        # its size will be total no of fingers which is five
        return fingers


########################################################################################################################