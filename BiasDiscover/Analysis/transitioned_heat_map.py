import argparse
import pandas as pd
import torch
from torchvision import models
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import os
import numpy as np


def get_label_mappings(data_dir):
    # Use a dummy transform since we just need the class names and indices
    transform = transforms.Compose([transforms.ToTensor()])
    # Load the dataset to access the class_to_idx attribute
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    # Return the mapping
    # Return the reversed mapping
    return {v: k for k, v in dataset.class_to_idx.items()}


def get_model(weights_path, num_classes=39, device='cuda:1'):
    model = models.vgg16_bn()
    print(f"num_classes = {num_classes}")
    num_ftrs = model.classifier[6].in_features
    print(f"num_ftrs = {num_ftrs}")
    model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        
    # Load model weights if a path is provided
    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage.cuda(device))
    model.load_state_dict(checkpoint)
    
    return model


class TransitionedImagesDataset(Dataset):
    def __init__(self, image_data):
        self.image_data = image_data
        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __len__(self):
        return len(self.image_data)
    
    def extract_identifier(self, regular_img_path):
        # Extract the base name of the file
        file_name = os.path.basename(regular_img_path)  # 'image (986).JPG'

        # Split the name from the extension and replace spaces with underscores
        name, _ = os.path.splitext(file_name)  # 'image (986)'
        formatted_name = name.replace(' ', '_')  # 'image_(986)'

        # Extract the directory name and replace underscores with spaces
        dir_name = os.path.basename(os.path.dirname(regular_img_path))  # 'Tomato___healthy'
        
        return f"{dir_name}_{formatted_name}"

    def __getitem__(self, idx):
        img_info = self.image_data[idx]
        identifier = self.extract_identifier(img_info["regular"]["regular_img_path"])
        regular_image = Image.open(img_info["regular"]["regular_img_path"]).convert('RGB')
        seg_wb_image = Image.open(img_info["seg_wb"]["seg_wb_img_path"]).convert('RGB')
        
        regular_image_tensor = self.transform(regular_image)
        seg_wb_image_tensor = self.transform(seg_wb_image)

        return  regular_image_tensor, img_info["regular"]["regular_pred"], \
                seg_wb_image_tensor, img_info["seg_wb"]["seg_wb_pred"], \
                identifier
    


def get_transitioned_images(path, directories):
    df = pd.read_csv(path)
    
    transitioned_imgs = []
    
    for index in range(0, len(df), 2):
        row_regular = df.iloc[index]
        row_seg_wb = df.iloc[index + 1] if index + 1 < len(df) else None
        
        # process regular images
        transitioned_img = {}
        if row_regular is not None:
            transitioned_img["regular"] = {
                "regular_img_path": directories["regular_dir"] + "/" + row_regular["image_path"],
                "regular_pred": row_regular["pred"]
            }

        # process seg_wb images
        if row_seg_wb is not None:
            transitioned_img["seg_wb"] = {
                "seg_wb_img_path": directories["seg_wb_dir"] + "/" + row_seg_wb["image_path"],
                "seg_wb_pred": row_seg_wb["pred"]
            }
            
        # Assuming the true label is the same for both regular and seg_wb images and is stored in the first of the two rows
        transitioned_img["true_label"] = row_regular["label"]

        transitioned_imgs.append(transitioned_img)
    
    return transitioned_imgs

def create_dir(output_dir, id):
    label_dir = os.path.join(output_dir, id)
    os.makedirs(label_dir, exist_ok=True)
    return label_dir

def get_visuals(img_tensors, cams):
    convert_to_image = transforms.ToPILImage()
    visuals = []
    for i in range(cams.shape[0]):
        cam = cams[i, :]
        img = np.array(convert_to_image(img_tensors[i]))
        
        # Normalize the image to be in the range [0, 1]
        img = img.astype(np.float32) / 255.0
        
        visual = show_cam_on_image(img, cam, use_rgb=True)
        visuals.append(visual)
        
    return visuals

def main(
         directories, transitioned_imgs, regular_model_path, seg_wb_model_path, output_dir, batch_size=8, 
         dataset_path="/home/heitorc62/PlantsConv/dataset/Plant_leave_diseases_dataset_without_augmentation"
        ):
    
    label_mappings = get_label_mappings(dataset_path)
    
    print("Producing dataset's DF...")
    transitioned_imgs = get_transitioned_images(transitioned_imgs, directories)

    print("Creating dataset...")
    dataset = TransitionedImagesDataset(transitioned_imgs)
    print("Dataset created successfully.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Carregar modelos
    regular_model = get_model(regular_model_path)
    seg_wb_model = get_model(seg_wb_model_path)
    print("Models loaded successfully.")
    
    regular_target_layer = [regular_model.features[42]]
    seg_wb_target_layer = [seg_wb_model.features[42]]
    
    
    # Instanciar CAM (GradCam)
    regular_cam = GradCAM(model=regular_model, target_layers=regular_target_layer, use_cuda=True)
    seg_wb_cam = GradCAM(model=seg_wb_model, target_layers=seg_wb_target_layer, use_cuda=True)
    
    print("Cams loaded successfully.")
    
        
    for batch in dataloader:
        regular_tensors, regular_preds, seg_wb_tensors, seg_wb_preds, identifier = batch

        regular_grayscale_cam = regular_cam(input_tensor=regular_tensors)
        regular_visualization = get_visuals(regular_tensors, regular_grayscale_cam)
        
        seg_wb_grayscale_cam = seg_wb_cam(input_tensor=seg_wb_tensors)
        seg_wb_visualization = get_visuals(seg_wb_tensors, seg_wb_grayscale_cam)
        
        # Iterate through the batch and save each result
        for (id, regular_vis, seg_wb_vis, regular_pred, seg_wb_pred) in zip(identifier, regular_visualization, seg_wb_visualization, regular_preds, seg_wb_preds):
            # Salvar resultados em uma pasta com o nome da imagem (true label) e duas imagens dentro com o nome de cada predição (regular e seg_wb)
            # Create the output directory based on true_label
            label_dir = create_dir(output_dir, id)

            # Process and save regular visualization
            regular_image_path = os.path.join(label_dir, f"regular_pred_{label_mappings[regular_pred.item()]}.jpg")
            Image.fromarray(regular_vis).save(regular_image_path)

            # Process and save seg_wb visualization
            seg_wb_image_path = os.path.join(label_dir, f"seg_wb_pred_{label_mappings[seg_wb_pred.item()]}.jpg")
            Image.fromarray(seg_wb_vis).save(seg_wb_image_path)
            
            break
        
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parses argument for VGG explainability program.")
    parser.add_argument("--regular_dir", type=str, help='')
    parser.add_argument("--seg_wb_dir", type=str, help='')
    parser.add_argument("--transitioned_imgs", type=str, help='')
    parser.add_argument("--regular_model", type=str, help='')
    parser.add_argument("--seg_wb_model", type=str, help='')
    parser.add_argument("--output_dir", type=str, help='')
    
    args = parser.parse_args()
    
    main({"regular_dir": args.regular_dir, "seg_wb_dir": args.seg_wb_dir}, args.transitioned_imgs, args.regular_model, args.seg_wb_model, args.output_dir)
    