import sys
import os
import torch
from utils.model import download_url
from model import VGG
from data.dataloader import load_data
from train_and_eval import train, evaluate
from utils.model import get_model_size


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_url = "https://hanlab18.mit.edu/files/course/labs/vgg.cifar.pretrained.pth"
    checkpoint = torch.load(download_url(checkpoint_url), map_location="cpu")
    print("Using device:", device)
    
    model = VGG().to(device)
    print(f"=> loading checkpoint '{checkpoint_url}'")
    model.load_state_dict(checkpoint['state_dict'])
    
    dataset, dataloader = load_data()
    
    fp32_model_accuracy = evaluate(device, model, dataloader['test'])
    # fp32_model_size = get_model_size(model)
    print(f"fp32 model has accuracy={fp32_model_accuracy:.2f}%")
    # print(f"fp32 model has size={fp32_model_size/MiB:.2f} MiB")

if __name__ == '__main__':
    main()
    