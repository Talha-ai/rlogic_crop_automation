import os
import cv2
import numpy as np

def remove_dark_background(image, threshold=30, min_area_ratio=0.1, padding=10):
    """
    Detects screen area and applies perspective correction
    """
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
        # Fix: Replace np.int0 with np.array(..., dtype=np.int32)
        box_points = cv2.boxPoints(rect)
        corners = np.array(box_points, dtype=np.int32)

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

def process_failed_images(input_path, output_path, failed_images):
    """
    Process only the failed images from the previous run
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    for class_name, image_names in failed_images.items():
        class_input_path = os.path.join(input_path, class_name)
        class_output_path = os.path.join(output_path, class_name)
        
        # Create class directory in output path
        os.makedirs(class_output_path, exist_ok=True)
        
        # Process each failed image
        for image_name in image_names:
            image_path = os.path.join(class_input_path, image_name)
            
            # Check if the image exists
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            
            # Read the image
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
    
    # Dictionary of failed images by class
    failed_images = {
        'vertical_line': [f'vertical_line_{x}.jpg' for x in [143, 1, 95]],
        'vertical_band': [f'vertical_band_{x}.jpg' for x in [94, 93, 88, 80, 8, 76, 77, 78, 74, 75, 71, 7, 70, 67, 65, 6, 60, 61, 62, 63, 55, 56, 57, 48, 49, 5, 43, 44, 45, 46, 40, 41, 32, 33, 34, 35, 36, 37, 38, 39, 3, 23, 2, 185, 180, 175, 176, 177, 178, 179, 170, 171, 172, 173, 167, 168, 169, 16, 161, 15, 145, 137, 13, 130, 131, 132, 133, 127, 128, 124, 122, 120, 12, 119, 117, 111]],
        'horizontal_line': [f'horizontal_line_{x}.jpg' for x in [98, 97, 94, 8, 6, 3, 151, 146, 139, 137]]
    }
    
    process_failed_images(input_dataset_path, output_dataset_path, failed_images)
    print("Failed images processing completed!")