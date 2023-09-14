import warnings
from lightning.pytorch.utilities.types import STEP_OUTPUT
warnings.filterwarnings('ignore')
import torch
from utils import MNISTModel, mnist_dataloader
from torch import nn, cuda
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torchvision.models import resnet50, ResNet50_Weights, vgg16
import argparse
import mlflow
import lightning.pytorch as L

NUM_CLASSES = 10
DEVICE = "cuda:0" if cuda.is_available() else "cpu"

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", type=str, dest="model_name", default="ResNet")
    parser.add_argument("-e", "--epoch", type=int, dest="NUM_EPOCHS", default=50)
    parser.add_argument("-op", "--optim", type=str, dest="OPTIM", default="Adam")
    parser.add_argument("-lr", "--learning_rate", type=float, dest="LEARNING_RATE", default=0.001)
    parser.add_argument("-lrs" "--lr_scheduler", type=str, dest="lr_scheduler", default='stepLR')
    parser.add_argument("-loss", "--loss_function", type=str, dest="loss_function", default='CrossEntropyLoss')
    parser.add_argument("-bs", "--batch_size", type=int, dest="BATCH_SIZE", default=64)
    parser.add_argument("-wd", "--weight_decay", type=float, dest="WEIGHT_DECAY", default=1e-5)
    parser.add_argument("-mm", "--momentum", type=float, dest="MOMENTUM", default=0.9)
    parser.add_argument("-wdir", "--weight_dir", type=str, dest="dir", default="weight")
    
    return parser.parse_args()

def main():

    args = get_args()
    
    train_dataloader, val_dataloader = mnist_dataloader(type='train', batch_size=64)

    # if not exist, creste a new experiment
    mlflow.set_experiment(experiment_name="mnist")

    model = MNISTModel(args, num_classes=NUM_CLASSES)
    # trainer config
    trainer = L.Trainer(max_epochs=5, devices=1, accelerator="gpu")
    
    # Enables (or disables) and configures autologging from PyTorch Lightning to MLflow.
    mlflow.pytorch.autolog()

    # Train the model
    with mlflow.start_run() as run:
        trainer.fit(model, train_dataloader, val_dataloader)

    # trainer.save_checkpoint("best_model.ckpt")

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in mlflow.MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


if __name__ == '__main__':
    main()