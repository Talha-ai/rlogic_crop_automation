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

    # Create a mask for the screen area
    screen_mask = np.zeros_like(gray)
    cv2.drawContours(screen_mask, [screen_contour], -1, (255, 255, 255), -1)

    # Get screen boundaries
    x, y, w, h = cv2.boundingRect(screen_contour)

    # Add padding to ensure clean edges
    x += padding
    y += padding
    w -= 2 * padding
    h -= 2 * padding

    # Ensure we don't exceed image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    # Create debug image showing the detected screen area
    debug_image = image.copy()
    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Create the output image
    # output = np.full_like(image, [255, 255, 255])  # White background

    # Copy the screen area exactly as is
    screen_area = image[y:y+h, x:x+w]
    # output[y:y+h, x:x+w] = screen_area

    return screen_area, debug_image
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