import face_recognition
import os
import cv2

KNOWN_FACES_DIR = "Resources"

TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog" #hog

video = cv2.VideoCapture(0)
print("Loading known faces")

known_faces = []
known_names = []


for name in os.listdir(KNOWN_FACES_DIR):
    # to add the path of the image with the image, e.g. to get the full path of the image
    # to load the image from the file with image name
    image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}")
    #  to find the encoding of the found image
    encoding = face_recognition.face_encodings(image)[0]
    # to add the encoding of the face in the list so that later we can compare
    known_faces.append(encoding)
    # to add the names of the images for example the ids in the list
    known_names.append(os.path.splitext(name)[0])

print("Processing unknown faces")
while True:
    success, image = video.read()
    # to add the location of the face in the each frame in locations list so that later we can compare faces
    locations = face_recognition.face_locations(image, model=MODEL)
    #  once the locations of the faces are found then provide the list of known faces locations in the parameter below
    # to get the exact encodings
    encodings = face_recognition.face_encodings(image, locations)

    #    to loop through both of the lists, that is why we use zip,
    #     encodeFace will grab from encodesCurrentFrame and faceLoc will grab from facesCurrentFrame one by one
    for encodeFace, faceLoc in zip(encodings, locations):
        # to compare faces. first we have to provide the known faces list, for example here the stored images
        # then you have to provide with what encodings you want to compare
        # then tolerance – How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance
        results = face_recognition.compare_faces(known_faces, encodeFace, TOLERANCE)
        # compare_faces returns of a list of true false, for example when it compares with the list
        # of known_faces(encodings) then it compare the found encoded faces with each of the known_faces and then once the result matches, that index becomes true
        # those which doesnt match will be false
        match = None
        if True in results:
            #  to get the name of the image from the list of images by matched face from results.
            match = known_names[results.index(True)]
            print(f"match found: {results}")
            top_left = (faceLoc[3], faceLoc[0])
            bottom_left = (faceLoc[1], faceLoc[2])
            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_left, color, FRAME_THICKNESS)

            top_left = (faceLoc[3], faceLoc[2])
            bottom_left = (faceLoc[1], faceLoc[2]+22)
            cv2.rectangle(image, top_left, bottom_left, color, cv2.FILLED)
            cv2.putText(image, match, (faceLoc[3] + 10, faceLoc[2] + 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        else:
            top_left = (faceLoc[3], faceLoc[0])
            bottom_left = (faceLoc[1], faceLoc[2])
            color = [0, 0, 255]
            cv2.rectangle(image, top_left, bottom_left, color, FRAME_THICKNESS)

            top_left = (faceLoc[3], faceLoc[2])
            bottom_left = (faceLoc[1], faceLoc[2] + 22)
            cv2.rectangle(image, top_left, bottom_left, color, cv2.FILLED)
            cv2.putText(image, "Not", (faceLoc[3]+10, faceLoc[2]+15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Result", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


