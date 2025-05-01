import cv2
import os
import pickle
import math
import numpy as np


def annotate_image(image_path, keypoints):
    """
    Annotate the rendered image with 2D keypoint labels using OpenCV.
    :param image_path: Path to the rendered image.
    :param keypoints: List of projected keypoints with IDs and 2D coordinates.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Font settings for OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0  # Adjust font scale as needed
    font_thickness = 3
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    alpha = 1.0  # Transparency factor (0.0 = fully transparent, 1.0 = fully opaque)

    # Adjust the rectangle background padding
    padding_x = 5  # Horizontal padding
    padding_y = 10  # Vertical padding

    for keypoint in keypoints:
        x, y = map(int, keypoint['coords_2d'])  # Convert to integers for OpenCV
        keypoint_id = str(int(keypoint['name']))

        # Get text size
        text_size, _ = cv2.getTextSize(keypoint_id, font, font_scale, font_thickness)
        text_width, text_height = text_size

        # get bounding box
        top_left = (x - padding_x, y - text_height - padding_y)
        bottom_right = (x + text_width + padding_x, y)

        # Create a copy of the image to draw the transparent rectangle
        overlay = image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, bg_color, thickness=cv2.FILLED)

        # Blend the overlay with the original image
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Draw text on top of the rectangle
        cv2.putText(image, 
                    keypoint_id, 
                    (x, y - int(padding_y / 2)), 
                    font, 
                    font_scale, 
                    text_color, 
                    thickness=font_thickness)

        if image.shape[2] > 3:
            for i in range(max(0, top_left[1]), min(bottom_right[1], image.shape[1])):
                for j in range(max(top_left[0], 0), min(bottom_right[0], image.shape[0])):
                    if np.sum(image[i, j, :3]) > 100:
                        image[i, j, 3] = 255

    # Save the annotated image
    annotated_image_path = image_path.replace(".png", "_annotated.png")
    cv2.imwrite(annotated_image_path, image)
    print(f"Annotated image saved to {annotated_image_path}")


def main(args):
    for file in os.listdir(args.input_data_dir):
        if file.endswith('.pkl'):
            pkl_file_path = os.path.join(args.input_data_dir, file)
            with open(pkl_file_path, 'rb') as f:
                keypoints = pickle.load(f)

            if args.no_3d_keypoints:
                image_path = os.path.join(args.input_data_dir, args.output_file_name)
                annotate_image(image_path, keypoints)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str)
    parser.add_argument('--output_file_name', type=str)
    parser.add_argument("--no_3d_keypoints", action="store_true", default=False)
    args = parser.parse_args()
    main(args)