import cv2
import os


def generate_dataset(user_name):
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    if face_classifier.empty():
        print("Error loading Haar Cascade XML file!")
        return

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None

        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
            return cropped_face

    user_folder = f"cropped_faces/{user_name}"
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    img_id = 0
    max_images = 10

    print("Press Enter to capture each image.")

    while img_id < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Display the frame
        cv2.imshow("Capture Image", frame)
        key = cv2.waitKey(1)

        if key == 13:
            cropped = face_cropped(frame)
            if cropped is not None:
                img_id += 1
                face = cv2.resize(cropped, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                file_name_path = f"{user_folder}/user_{img_id}.jpg"
                cv2.imwrite(file_name_path, face)

                print(f"Captured image {img_id}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Dataset collection completed. {img_id} samples were saved.")


user_name = input("Enter the user's name: ").strip()
generate_dataset(user_name)
