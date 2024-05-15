from rembg import remove
import os

# input and options
in_dir = '/home/heitor/USP/IC/FAPESP/code_dataset/dataset/WB_dataset/'
out_dir = '/home/heitor/USP/IC/FAPESP/code_dataset/dataset/WB_dataset_segmented/'

imgfiles = []
valid_images = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG")

# Collect all valid image files
print("All directories:")
classes = os.listdir(in_dir)
for class_dir in classes:
    class_path = os.path.join(in_dir, class_dir)
    if os.path.isdir(class_path):  # Make sure it's a directory
        for f in os.listdir(class_path):
            if f.lower().endswith(valid_images):
                imgfiles.append(os.path.join(class_path, f))

# Create the output directory if it doesn't exist
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Process each image
for in_img in imgfiles:
    print("Processing image: " + in_img)
    separated_path = os.path.normpath(in_img).split(os.sep)
    file_name = separated_path[-1]
    class_dir = separated_path[-2]

    class_out_dir = os.path.join(out_dir, class_dir)
    if not os.path.exists(class_out_dir):
        os.mkdir(class_out_dir)

    out_img_path = os.path.join(class_out_dir, file_name)

    # Read, process, and write the image
    with open(in_img, 'rb') as i:
        input_data = i.read()
        output_data = remove(input_data)
        with open(out_img_path, 'wb') as o:
            o.write(output_data)

print("Processing complete.")