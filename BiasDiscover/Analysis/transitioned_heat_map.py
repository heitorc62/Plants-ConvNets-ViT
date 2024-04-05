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
    def __init__(self, image_data, transform=None):
        self.image_data = image_data
        self.transform = transform

    def __len__(self):
        return len(self.image_data)
    
    def extract_identifier(self, regular_img_path):
        return regular_img_path.replace("/image ", "")

    def __getitem__(self, idx):
        img_info = self.image_data[idx]
        identifier = self.extract_identifier(img_info["regular"]["regular_img_path"])
        regular_image = Image.open(img_info["regular_img_path"]).convert('RGB')
        seg_wb_image = Image.open(img_info["seg_wb_img_path"]).convert('RGB')
        
        if self.transform:
            regular_image = self.transform(regular_image)
            seg_wb_image = self.transform(seg_wb_image)

        return regular_image, img_info["regular"]["pred"], seg_wb_image, img_info["seg_wb"]["pred"], identifier
    


def get_transitioned_images(path, directories):
    df = pd.read_csv(path)
    
    transtioned_imgs = []
    transitioned_img = {}
    # loop through the rows using iterrows()
    for index, row in df.iterrows():
        tmp_dict = {}
        if index % 2 == 0: # regular images
            tmp_dict["regular_img_path"] = directories["regular_dir"] + row["image_path"]
            tmp_dict["regular_pred"] = row["pred"]
            transitioned_img["regular"] = tmp_dict
        else: # seg_wb images
            tmp_dict["seg_wb_img_path"] = directories["seg_wb_dir"] + row["image_path"].replace(".JPG", "_marked.JPG")
            tmp_dict["seg_wb_pred"] = row["pred"]
            transitioned_img["seg_wb"] = tmp_dict
            
        transitioned_img["true_label"] = row["label"]
        transtioned_imgs.append(transitioned_img)
    
    return transtioned_imgs

def create_dir(id, output_dir):
    label_dir = os.path.join(output_dir, id)
    os.makedirs(label_dir, exist_ok=True)
    return label_dir

def main(
         directories, transitioned_imgs, regular_model_path, seg_wb_model_path, output_dir, batch_size=8, 
         dataset_path="/home/heitorc62/PlantsConv/dataset/Plant_leave_diseases_dataset_without_augmentation"
        ):
    
    label_mappings = get_label_mappings(dataset_path)
    
    print("Producing dataset's DF...")
    transitioned_imgs = get_transitioned_images(transitioned_imgs, directories)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Creating dataset...")
    dataset = TransitionedImagesDataset(transitioned_imgs, transform=transform)
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
        regular_images, regular_preds, seg_wb_images, seg_wb_preds, identifier = batch

        regular_grayscale_cam = regular_cam(input_tensor=regular_images)
        regular_visualization = show_cam_on_image(regular_images, regular_grayscale_cam, use_rgb=True)
        
        seg_wb_grayscale_cam = seg_wb_cam(input_tensor=regular_images)
        seg_wb_visualization = show_cam_on_image(seg_wb_images, seg_wb_grayscale_cam, use_rgb=True)
        
        # Iterate through the batch and save each result
        for (id, regular_vis, seg_wb_vis, regular_pred, seg_wb_pred) in zip(identifier, regular_visualization, seg_wb_visualization, regular_preds, seg_wb_preds):
            # Salvar resultados em uma pasta com o nome da imagem (true label) e duas imagens dentro com o nome de cada predição (regular e seg_wb)
            # Create the output directory based on true_label
            label_dir = create_dir(output_dir, id)

            # Process and save regular visualization
            regular_image_path = os.path.join(label_dir, f"regular_pred_{regular_pred.map(label_mappings)}.jpg")
            Image.fromarray(regular_vis).save(regular_image_path)

            # Process and save seg_wb visualization
            seg_wb_image_path = os.path.join(label_dir, f"seg_wb_pred_{seg_wb_pred.map(label_mappings)}.jpg")
            Image.fromarray(seg_wb_vis).save(seg_wb_image_path)

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
    