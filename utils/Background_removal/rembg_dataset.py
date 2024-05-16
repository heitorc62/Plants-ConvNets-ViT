from rembg import remove
import os
import argparse


def main(in_dir, out_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    imgfiles = []
    valid_images = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG")
    imgfiles = []
    valid_images = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG")
    print("All directories: ")
    classes = os.listdir(in_dir)
    for class_dir in classes:
        for f in os.listdir(os.path.join(in_dir, class_dir)):
            if f.lower().endswith(valid_images):
                imgfiles.append(os.path.join(in_dir, class_dir, f))



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


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument('--input_dir', type=str, help='')
    parser.add_argument('--output_dir', type=str, help='')
    # Parse the arguments
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)

    
    
    