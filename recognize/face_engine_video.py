# To capture face recognition using a VIDEO file.
import os 
import cv2
import face_recognition

# Import your video file
video_file = cv2.VideoCapture(os.path.abspath("recognize/videos/new_kids_turbo.mp4"))

# Capture the length based on the frame.
length = int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))

# We need to add all the faces that we want our code to recognize
image_Barrie_1 = face_recognition.load_image_file(os.path.abspath("recognize/images/Barrie.png"))
image_Gerrie_1 = face_recognition.load_image_file(os.path.abspath("recognize/images/Gerrie.png"))
image_Richard_1 = face_recognition.load_image_file(os.path.abspath("recognize/images/Richard.png"))
image_Rikket_1 = face_recognition.load_image_file(os.path.abspath("recognize/images/Rikket.png"))
image_Robbie_1 = face_recognition.load_image_file(os.path.abspath("recognize/images/Robbie.png"))

# Generate the face encoding for the image that has been passed.
Barrie_face_1 = face_recognition.face_encodings(image_Barrie_1)[0]
Gerrie_face_1 = face_recognition.face_encodings(image_Gerrie_1)[0]
Richard_face_1 = face_recognition.face_encodings(image_Richard_1)[0]
Rikket_face_1 = face_recognition.face_encodings(image_Rikket_1)[0]
Robbie_face_1 = face_recognition.face_encodings(image_Robbie_1)[0]

# Make a list of all the known faces that we want to be recognized based on the 
# encoding.
known_faces = [
Barrie_face_1, Gerrie_face_1, Richard_face_1, Rikket_face_1, Robbie_face_1
]

facial_points = []
face_encodings = []
facial_number = 0

while True:
    return_value, frame = video_file.read()
    facial_number = facial_number + 1
    
    if not return_value:
        break
    rgb_frame = frame[:, :, ::-1]

    facial_points = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, facial_points)

    facial_names = []
    for encoding in face_encodings:
        match = face_recognition.compare_faces(known_faces, encoding, tolerance=0.50)
        # alex_face_1, badguy_face_1, farah_face_1, laswell_face_1, price_face_1
        # match = [False, True, True, False , False]

        name = ""
        if match[0]:
            name = "Barrie"
        if match[1]:
            name = "Gerrie"
        if match[2]:
            name = "Richard"
        if match[3]:
            name = "Rikket"
        if match[4]:
            name = "Robbie"

        facial_names.append(name)

    for (top, right, bottom, left), name in zip(facial_points, facial_names):
        # Enclose the face with the box - Red color 
        # top, right, bottom, left - 129, 710, 373, 465
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Name the characters in the Box created above
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    codec = int(video_file.get(cv2.CAP_PROP_FOURCC))
    fps = int(video_file.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_movie = cv2.VideoWriter("output_{}.mp4".format(facial_number), codec, fps, (frame_width,frame_height))
    print("Writing frame {} / {}".format(facial_number, length))
    output_movie.write(frame)

video_file.release()
output_movie.release()
cv2.destroyAllWindows()
