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

	model_uri = r"mlruns\283747705774240454\91338da70e0b4f99b13e9f192b90b36b\artifacts\model"
	model = mlflow.pytorch.load_model(model_uri)
	
	ckpt_path = r"lightning_logs\version_0\checkpoints\epoch=1-step=1500.ckpt"
	model = model.load_from_checkpoint(ckpt_path)
	model.eval()

	img = np.load(get_args().img)
	transform=transforms.ToTensor()
	img = torch.unsqueeze(transform(img), 0)
	img = img.to(DEVICE)

	with torch.no_grad():
		output = model(img)
	_, pred = torch.max(output, 1)

	print(pred.item())


if __name__ == '__main__':
    main()