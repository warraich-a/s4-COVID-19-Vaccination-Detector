import cv2
import numpy as np
import face_recognition
import os

# step 1
#  to get the images automatically and their names
path = 'Resources'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    # to add the path of the image with the image, e.g. to get the full path of the image
    curImage = cv2.imread(f'{path}/{cl}')
    # to add the image in the images list
    images.append(curImage)
    # to add the name of the image in the list of classNames without extension by spliting the text
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# step 2
# a method to find the encodings
def findEncodings(imageList):
    encodeList = []
    # to loop through the given list
    for img in imageList:
        # to get the colors of the pictures
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # encode all the pictures in the given list, however here it will only get the first face becuse
        #  of given [0]
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('Encoding complete')
#

# step 3
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    # to read each frame and decrease the size to 0.25 so that program runs fast
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    # to get colors of the frames
    imgS = cv2.cvtColor(imgS, cv2.COLOR_RGB2BGR)

    # to get the current frame list by each frame, e.g. imgS is one frame
    facesCurrentFrame = face_recognition.face_locations(imgS)
    #  to get the encoding of each frame
    encodesCurrentFrame = face_recognition.face_encodings(imgS, facesCurrentFrame)

#    to loop through both of the lists, that is why we use zip,
#     encodeFace will grab from encodesCurrentFrame and faceLoc will grab from facesCurrentFrame one by one
    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        # to compare the found faces with the list of stored images, for example with the list of faces already in Resources
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        #  to get the distance between the stored faces and new camera faces
        #  however the first element in the list will be the one that has the lowest value, means matching the most
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDist)

        #  to get the lowest found distance value's distance
        matchIndex =np.argmin(faceDist)
        print(matchIndex)

        # to furthur deal with the matched face, for example if you want to put a rectangle around the face
        # for example if match index is in matches list e.g. if 3 is in the list which is index of lets say 0.36....
        if matches[matchIndex]:
            # to get the name correct name of the image from list.
            name = classNames[matchIndex].upper()
            print(name)
            # here the faceloc is the one that is coming from the loop e.g. facesCurrentFrame is the list and faceLoc is one instance
            #  in for loop
            y1,x2,y2,x1 = faceLoc
            # the reason why multiplying is because the original size of the frame was decreasd above but now I want increase it again
            y1, x2, y2, x1 =y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1), (x2,y2),(255.155,100), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0.255, 100), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)
        else:
            y1, x2, y2, x1 = faceLoc
            # the reason why multiplying is because the original size of the frame was decreasd above but now I want increase it again
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, "Not", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow('Webcame', img)
    cv2.waitKey(1)



# # step 2
# faceLocation = face_recognition.face_locations(imgMain)[0]
# encodeMainImage = face_recognition.face_encodings(imgMain)[0]
# # print(faceLocation)
# cv2.rectangle(imgMain, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (0, 0, 255), 2)
#
# # in the square bracket you write what number of face you want to detect, lets say
# #   you have three faces in one picture then you can write 0,1 or 2 but if you write 3 you
# #  will get an error. because that index is out of range
# faceLocationTest = face_recognition.face_locations(imgTest)[0]
# encodeTestImage = face_recognition.face_encodings(imgTest)[0]
# # print(faceLocation)
# cv2.rectangle(imgTest, (faceLocationTest[3], faceLocationTest[0]), (faceLocationTest[1], faceLocationTest[2]),
#               (0, 0, 255), 2)
#
# # step 3
# # to compare faces
# results = face_recognition.compare_faces([encodeMainImage],encodeTestImage)
# #  to get the distance between images, for example it will tell you how much is the
# # difference such as if you have the pictures of the same person it will say 0.36.... or more,
# # but if you have the pictures of the different persons then it will have higher value e.g. 0.67.. or more or less
# faceDistance = face_recognition.face_distance([encodeMainImage], encodeTestImage)
