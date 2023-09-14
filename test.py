import torch
import torchvision.transforms as transforms
import numpy as np
from utils import mnist_dataloader
import argparse
import mlflow
import lightning.pytorch as L

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", type=str, dest="model_name", default="ResNet")
	parser.add_argument("-w","--weight_path", type=str, dest="weight_path", default=r'weight\best_weight.pt')
	parser.add_argument("-img","--image_path", type=str, dest="img", default=r'test_img7.npy')

	return parser.parse_args()

def main():

    test_dataloader = mnist_dataloader(type='test', batch_size=1)

    # Inference after loading the logged model
    model_uri = r"mlruns\283747705774240454\91338da70e0b4f99b13e9f192b90b36b\artifacts\model"
    model = mlflow.pytorch.load_model(model_uri)

    trainer = L.Trainer()
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == '__main__':
    main()