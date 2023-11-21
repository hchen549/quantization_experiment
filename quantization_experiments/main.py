import sys
import os
import torch
from utils.model import download_url


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_url = "https://hanlab18.mit.edu/files/course/labs/vgg.cifar.pretrained.pth"
    checkpoint = torch.load(download_url(checkpoint_url), map_location="cpu")
    print("Using device:", device)
    

if __name__ == '__main__':
   main()
    