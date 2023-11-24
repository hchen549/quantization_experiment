import sys
import os
import torch
from torch import nn

from model_util import download_url, get_model_size
from model import VGG, KMeansQuantizer
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
    
    Byte = 8
    KiB = 1024 * Byte
    MiB = 1024 * KiB
    GiB = 1024 * MiB
    
    
    
    # fp32_model_accuracy = evaluate(model, dataloader['test'])
    # fp32_model_size = get_model_size(model)
    # print(f"fp32 model has accuracy={fp32_model_accuracy:.2f}%")
    # print(f"fp32 model has size={fp32_model_size/MiB:.2f} MiB")
    
    bitwidth = 2
    print(f'k-means quantizing model into {bitwidth} bits')
    # quantizer = KMeansQuantizer(model, bitwidth)
    # quantized_model_size = get_model_size(model, bitwidth)
    # print(f"    {bitwidth}-bit k-means quantized model has size={quantized_model_size/MiB:.2f} MiB")
    # quantized_model_accuracy = evaluate(model, dataloader['test'])
    # print(f"    {bitwidth}-bit k-means quantized model has accuracy={quantized_model_accuracy:.2f}%")
    
    quantizer = KMeansQuantizer(model, bitwidth)
    # quantizer.apply(model, update_centroids=False)
    quantized_model_size = get_model_size(model, bitwidth)
    print(f" {bitwidth}-bit k-means quantized model has size={quantized_model_size/MiB:.2f} MiB")
    quantized_model_accuracy = evaluate(model, dataloader['test'])
    print(f" {bitwidth}-bit k-means quantized model has accuracy={quantized_model_accuracy:.2f}% before quantization-aware training ")
    
    num_finetune_epochs = 3
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0
   
    for i in range(num_finetune_epochs):
        train(model, dataloader['train'], criterion, optimizer, scheduler, 
                callbacks=[lambda: quantizer.apply(model, update_centroids=True)])
        model_accuracy = evaluate(model, dataloader['test'])
        is_best = model_accuracy > best_accuracy
        best_accuracy = max(model_accuracy, best_accuracy)
        print(f' Epoch {i} Accuracy {model_accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%')
        

if __name__ == '__main__':
    main()
    