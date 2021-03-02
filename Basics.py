import cv2
import numpy as nm
import face_recognition

# step 1
imgMain = face_recognition.load_image_file('Resources/Anas.jpg')
# this is must otherwise you will get inverted colors
imgMain = cv2.cvtColor(imgMain, cv2.COLOR_RGB2BGR)
imgTest = face_recognition.load_image_file('Resources/Group_photo.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_RGB2BGR)

# width first then height
# imgMainResized= cv2.resize(imgMain, (600, 500))
# imgTestResized = cv2.resize(imgTest, (600, 600))

# step 2
faceLocation = face_recognition.face_locations(imgMain)[0]
encodeMainImage = face_recognition.face_encodings(imgMain)[0]
# print(faceLocation)
cv2.rectangle(imgMain, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (0, 0, 255), 2)

# in the square bracket you write what number of face you want to detect, lets say
#   you have three faces in one picture then you can write 0,1 or 2 but if you write 3 you
#  will get an error. because that index is out of range
faceLocationTest = face_recognition.face_locations(imgTest)[0]
encodeTestImage = face_recognition.face_encodings(imgTest)[0]
# print(faceLocation)
cv2.rectangle(imgTest, (faceLocationTest[3], faceLocationTest[0]), (faceLocationTest[1], faceLocationTest[2]),
              (0, 0, 255), 2)

# step 3
# to compare faces
results = face_recognition.compare_faces([encodeMainImage],encodeTestImage)
#  to get the distance between images, for example it will tell you how much is the
# difference such as if you have the pictures of the same person it will say 0.36.... or more,
# but if you have the pictures of the different persons then it will have higher value e.g. 0.67.. or more or less
faceDistance = face_recognition.face_distance([encodeMainImage], encodeTestImage)
print(results, faceDistance)

# step 4
#  to put text
#  first parameter is the image on which you want to put the text
#  second parameter is actual text, but below im puting the result e.g. true or false with distance( 2 is decimal numbers I want)
#  third parameter is the location or area where I want to put the text
#  4th parameter is the font
#  5th font size
#  6th is the color of the font
#  7th thickness of the text
cv2.putText(imgTest,f'{results} {round(faceDistance[0], 2)}', (50,50), cv2.FONT_ITALIC, 1, (255,0,255), 2)
cv2.imshow('Main', imgMain)
cv2.imshow('Test', imgTest)

cv2.waitKey(0)
