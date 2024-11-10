import os
import cv2
import numpy as np

def remove_dark_background(image, threshold=30, min_area_ratio=0.1, padding=10):
    # Create initial binary mask for screen detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Clean up the binary mask to better detect screen boundaries
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find screen contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image, image.copy()

    # Find the largest bright region (presumably the screen)
    valid_contours = []
    total_area = image.shape[0] * image.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > total_area * min_area_ratio:
            valid_contours.append((area, cnt))

    if not valid_contours:
        return image, image.copy()

    # Get the screen contour
    screen_contour = max(valid_contours, key=lambda x: x[0])[1]

    # Approximate the contour to get a polygon
    epsilon = 0.02 * cv2.arcLength(screen_contour, True)
    approx = cv2.approxPolyDP(screen_contour, epsilon, True)

    # Get corners for perspective transform
    if len(approx) == 4:
        corners = approx
    else:
        # If we don't get exactly 4 points, use the bounding rectangle
        rect = cv2.minAreaRect(screen_contour)
        corners = np.int0(cv2.boxPoints(rect))

    # Order points in [top-left, top-right, bottom-right, bottom-left] order
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")

        # Sum the x+y coordinates - top-left will have smallest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Diff the coordinates - top-right will have smallest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    corners = order_points(corners.reshape(4, 2))

    # Calculate the width and height of the new image
    width_a = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) + ((corners[2][1] - corners[3][1]) ** 2))
    width_b = np.sqrt(((corners[1][0] - corners[0][0]) ** 2) + ((corners[1][1] - corners[0][1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((corners[1][0] - corners[2][0]) ** 2) + ((corners[1][1] - corners[2][1]) ** 2))
    height_b = np.sqrt(((corners[0][0] - corners[3][0]) ** 2) + ((corners[0][1] - corners[3][1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Define destination points for perspective transform
    dst = np.array([
        [padding, padding],
        [max_width - padding, padding],
        [max_width - padding, max_height - padding],
        [padding, max_height - padding]
    ], dtype="float32")

    # Calculate perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    # Create debug image showing the detected corners
    debug_image = image.copy()
    for i in range(4):
        cv2.circle(debug_image, tuple(corners[i].astype(int)), 5, (0, 255, 0), -1)
        cv2.line(debug_image,
                tuple(corners[i].astype(int)),
                tuple(corners[(i+1)%4].astype(int)),
                (0, 255, 0), 2)

    # Remove any remaining black borders
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_warped, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        final_image = warped[y:y+h, x:x+w]
    else:
        final_image = warped

    return final_image, debug_image

def process_dataset(input_path, output_path):
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Process each class folder
    for class_name in ['good', 'horizontal_band', 'horizontal_line', 'vertical_band', 'vertical_line']:
        class_input_path = os.path.join(input_path, class_name)
        class_output_path = os.path.join(output_path, class_name)
        
        # Create class directory in output path
        os.makedirs(class_output_path, exist_ok=True)
        
        # Process each image in the class folder
        for image_name in os.listdir(class_input_path):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            # Read the image
            image_path = os.path.join(class_input_path, image_name)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue
            
            # Process the image
            try:
                processed_image, _ = remove_dark_background(image)
                
                # Create new filename with _cropped suffix
                name_without_ext = os.path.splitext(image_name)[0]
                extension = os.path.splitext(image_name)[1]
                new_image_name = f"{name_without_ext}_cropped{extension}"
                
                # Save the processed image
                output_path_full = os.path.join(class_output_path, new_image_name)
                cv2.imwrite(output_path_full, processed_image)
                print(f"Successfully processed: {image_name} -> {new_image_name}")
                
            except Exception as e:
                print(f"Error processing {image_name}: {str(e)}")

if __name__ == "__main__":
    input_dataset_path = r"D:\rlogic_traning_dataset\rlogic_traning_dataset"
    output_dataset_path = r"D:\rlogic_traning_dataset\rlogic_training_dataset_cropped"
    
    process_dataset(input_dataset_path, output_dataset_path)
    print("Dataset processing completed!")