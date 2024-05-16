from rembg import remove
import os
import argparse


def main(in_dir, out_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    imgfiles = []
    valid_images = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG")
    
    print("All directories: ")
    classes = os.listdir(in_dir)
    for class_dir in classes:
        class_path = os.path.join(in_dir, class_dir)
        if os.path.isdir(class_path):
            for f in os.listdir(class_path):
                if f.lower().endswith(valid_images):
                    imgfiles.append(os.path.join(class_path, f))

    # Process each image
    for in_img in imgfiles:
        print("Processing image: " + in_img)
        separated_path = os.path.normpath(in_img).split(os.sep)
        file_name = separated_path[-1]
        class_dir = separated_path[-2]

        class_out_dir = os.path.join(out_dir, class_dir)
        os.makedirs(class_out_dir, exist_ok=True)

        out_img_path = os.path.join(class_out_dir, file_name)

        # Read, process, and write the image
        with open(in_img, 'rb') as i:
            input_data = i.read()
            output_data = remove(input_data)
            with open(out_img_path, 'wb') as o:
                o.write(output_data)

    print("Processing complete.")


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Remove backgrounds from images and save them to a specified output directory.')
    
    # Add the arguments
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the input images organized by class.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed images.')
    
    # Parse the arguments
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
