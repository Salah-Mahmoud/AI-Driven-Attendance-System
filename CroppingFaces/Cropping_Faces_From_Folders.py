import cv2
import os
import json


def generate_dataset_from_annotations(train_folder, annotations_file, output_folder):
    # Load the Haar Cascade for face detection (optional, can be removed if bounding box is sufficient)
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    if face_classifier.empty():
        print("Error loading Haar Cascade XML file!")
        return

    # Create the base output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the annotations from the JSON file
    annotations_path = os.path.join(train_folder, annotations_file)
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    # Extract images, categories, and annotations
    images = {img['id']: img['file_name'] for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    annotations = data['annotations']

    # Iterate through each annotation
    for annotation in annotations:
        image_id = annotation['image_id']
        image_file = images.get(image_id)
        bounding_box = annotation['bbox']  # Bounding box for the face [x, y, width, height]
        class_label = categories.get(annotation['category_id'])  # Class label of the face

        if not image_file:
            print(f"Error: Image ID {image_id} not found")
            continue

        # Read the image from the train folder
        image_path = os.path.join(train_folder, image_file)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Error: Could not read image {image_path}")
            continue

        # Convert bounding box coordinates to integers
        x, y, w, h = map(int, bounding_box)

        # Crop the face using the bounding box
        cropped_face = img[y:y + h, x:x + w]

        if cropped_face is None or cropped_face.size == 0:
            print(f"Error: Cropped face is empty or invalid in {image_file}")
            continue

        # Resize and convert to grayscale (optional)
        face = cv2.resize(cropped_face, (200, 200))
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Create a directory for the class if it doesn't exist
        class_folder = os.path.join(output_folder, class_label)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        # Save the cropped face image in the class-specific folder
        output_path = os.path.join(class_folder,
                                   f"{os.path.splitext(image_file)[0]}_{annotations.index(annotation)}.jpg")
        cv2.imwrite(output_path, face_gray)

        print(f"Saved cropped face: {output_path}")

    print("Dataset processing completed.")


# Set the folders
train_folder = 'valid'
annotations_file = '_annotations.coco.json'  # Name of the JSON file containing annotations
output_folder = 'cropped_test'

# Call the function to start processing the dataset
generate_dataset_from_annotations(train_folder, annotations_file, output_folder)
