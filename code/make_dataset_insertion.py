import os
import cv2
import numpy as np

# Define paths
input_folder = "/home/bonggeeun/team_project/ceti-glove-main/cetiglove/dataset"
output_dir = "/home/bonggeeun/team_project/diffusion_model/Imitating-Human-Behaviour-w-Diffusion-main/code/dataset_insertion"
#input_folder = "D:\TUD\Team Project Robot\code\dataset"
#output_dir = "dataset_insertion"
output_file_robot_hand = os.path.join(output_dir, "actions.npy")
output_file_images = os.path.join(output_dir, "images.npy")
output_file_images_small = os.path.join(output_dir, "images_small.npy")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Create lists to store all image arrays
robot_hand_data = []
images = []
images_small = []

def scale_image(img, scale_factor):
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

def rotate_image(img, angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return img

# Iterate over each subdirectory in the input folder to find robot_hand.npy files and .png images
for subdir in sorted(os.listdir(input_folder)):
    subdir_path = os.path.join(input_folder, subdir)
    if os.path.isdir(subdir_path):
        # Path to the robot_hand.npy file in the subdirectory
        npy_file_path = os.path.join(subdir_path, "robot_action.npy")
        
        # Check if the file exists
        if os.path.exists(npy_file_path):
            # Load the robot_hand.npy file
            robot_hand = np.load(npy_file_path)
            # Append the loaded data to the list
            robot_hand_data.append(robot_hand)
            #print(f"Loaded {npy_file_path}")
        else:
            print(f"robot_hand.npy not found in {subdir_path}")
        
        # Process .png files within the same subdirectory
        for filename in sorted(os.listdir(subdir_path)):
            if filename.lower().endswith(".png"):
                img_path = os.path.join(subdir_path, filename)
                image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                # If the image was successfully loaded
                if image is not None:
                    # Process the image (optional scaling and rotation)
                    image = scale_image(image, scale_factor=1.0)
                    image = rotate_image(image, angle=0)

                    # Create a 32x32 small version for quick training
                    image_small = cv2.resize(image, (32, 32))

                    # Append the processed images to the lists
                    images.append(image)
                    images_small.append(image_small)
                    #print(f"Image {filename} processed successfully")
                else: 
                    print(f"Failed to load image {img_path}")

# Combine the lists into .npy files
combined_robot_hand = np.concatenate(robot_hand_data, axis=0) if robot_hand_data else np.array([])
combined_images = np.array(images)
combined_images_small = np.array(images_small)

# Save the combined images as .npy files in the dataset_insertion folder
np.save(output_file_robot_hand, combined_robot_hand)
np.save(output_file_images, combined_images)
np.save(output_file_images_small, combined_images_small)

#print(f"Saved combined robot_hand data to {output_file_robot_hand}")
#print(f"Saved combined images to {output_file_images}")
#print(f"Saved combined small images to {output_file_images_small}")
