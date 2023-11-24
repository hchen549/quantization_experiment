import sys
import os
import torch
from model_util import download_url, get_model_size
from model import VGG
from dataloader import load_data
from train_and_eval import train, evaluate
# from utils.model import 


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_url = "https://hanlab18.mit.edu/files/course/labs/vgg.cifar.pretrained.pth"
    checkpoint = torch.load(download_url(checkpoint_url), map_location="cpu")
    print("Using device:", device)
    
    model = VGG().to(device)
    print(f"=> loading checkpoint '{checkpoint_url}'")
    model.load_state_dict(checkpoint['state_dict'])
    
    
    
    dataset, dataloader = load_data()
    
    # inputs, targets = next(iter(dataloader["test"]))
    # inputs = inputs.cuda()

    # targets = targets.cuda()

    # # Inference
    # outputs = model(inputs)
    
    Byte = 8
    KiB = 1024 * Byte
    MiB = 1024 * KiB
    GiB = 1024 * MiB
    
    fp32_model_accuracy = evaluate(model, dataloader['test'])
    fp32_model_size = get_model_size(model)
    print(f"fp32 model has accuracy={fp32_model_accuracy:.2f}%")
    print(f"fp32 model has size={fp32_model_size/MiB:.2f} MiB")

if __name__ == '__main__':
    main()
    