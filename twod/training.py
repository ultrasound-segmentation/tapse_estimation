import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import random
import os
import wandb  # Import wandb
from tqdm import tqdm  # Import tqdm for progress bar
from torchinfo import summary

from dataloader.dataset_class import KeypointDataset
from losses.distances import OrderedDistanceLoss, GaussianKeypointLoss
from models.weights_initialization import initialize_weights
from models.models import EncoderDecoder_3d, Unet as monai_UNet, ResNet50Regression, ResNet34Regression, ResNet18Regression, SwinUNETR, ResNeXt50Regression
from models.improved_unet import ImprovedUnet
from postprocessing.coordinates_calculation_from_masks import center_of_mass
from dataloader.preprocessing import preprocess_images
from utils.plot import save_image, visualize_image
from utils.save import get_experiment_path
from callbacks.early_stopping import EarlyStopping
from callbacks.lr_schedule import ReduceLROnPlateau



# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train U-Net model for keypoint detection.')
    parser.add_argument('--epochs', type=int, default= 300, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--num_keypoints', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='Path to save model checkpoints')
    parser.add_argument('--model', type=str, default='U-Net', help='name of the model: supported "U-Net"')
    parser.add_argument('--save_images', action='store_true', help='If to save test images with predictions')
    parser.add_argument('--train_data', type=str, default= r'd:\mmissana\data\RV_PATIENTS\dataset_after_review\train.npz', help='Path to the training dataset')
    parser.add_argument('--val_data', type=str, default=r'd:\mmissana\data\RV_PATIENTS\dataset_after_review\val.npz', help='Path to the validation dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--initial_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--model_path', type=str, default='dl_mapse/Data/best_loss_weights_unet_light.pth', help='Path to the pre-trained model weights')
    parser.add_argument('--wandb_project', type=str, default='rv_focused_training', help='tapse')
    parser.add_argument('--augm_version', type=str, default='0', help='augmentation version you want to use')
    parser.add_argument('--loss', type=str, default='ordered_distance', help='select the type of loss: "MSE" or "distance"')
    parser.add_argument('--wandb_entity', type=str, default=None, help='master_thesis_NTNU_mmissana')
    parser.add_argument('--save_model_path', type=str, default=None, help='Path to save trained model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--thresh', type=float, default=0.9, help='Treshold for center of mass calculation')
    parser.add_argument('--from_scratch', action='store_true', help='Train model from scratch')
    parser.add_argument('--start_filts', type=int, default=8, help='Number of filters in the first layer of the model when using monai unet')
    parser.add_argument('--depth', type=int, default=6, help='Number of layers in the model when using monai unet')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the model, supported only for monai unet')
    parser.add_argument('--n_residuals', type=int, default=0, help='Number of residual blocks in the model when using monai unet')

    return parser.parse_args()

args = parse_args()
# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

g = torch.Generator()
g.manual_seed(args.seed)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, checkpoint_path, save_model_path=None, thresh=0.9):
    model.train()
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=patience, path=checkpoint_path)
    scheduler = ReduceLROnPlateau(optimizer, monitor='val_loss', mode='min', patience=3, factor=0.3, min_lr=0, initial_lr=args.initial_lr)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        
        # Use tqdm to create a progress bar for the training loop
        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                # print(masks.shape)  # Debugging line to check shapes
                # print(images.max(), images.min())  # Debugging line to check image values
                # visualize_image(images[0, 0, 0].cpu().numpy(), points=tuple(masks[0].cpu().numpy()))  # Visualize first image and its keypoints
                optimizer.zero_grad()

                outputs = model(images)  # Forward pass	

                if 'ResNet' not in args.model and 'resnext' not in args.model: # If not using ResNet, compute center of mass
                    # Compute center of mass for output masks
                    com_tensor = torch.stack([
                        torch.stack([center_of_mass(mask, device=device, normalize=False, thresh=thresh) for mask in output])
                        for output in outputs
                    ]).to(device)

                    loss = criterion(com_tensor, masks)

                else: # resnet does a regression
                    loss = criterion(outputs, masks)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())  # Update progress bar with loss
                pbar.update(1)  # Move progress bar forward

        avg_loss = running_loss / len(train_loader)
        val_loss = validate(model, val_loader, criterion, thresh=thresh)
        scheduler(val_loss)

        # Log losses to wandb
        wandb.log({"train_loss": avg_loss, "val_loss": val_loss, "learning_rate": scheduler.currentlr})

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")

            # Save last model
            model_path = os.path.join(save_model_path, "last_model.pth")
            torch.save({'model_state_dict': model.state_dict()}, model_path)

            # load best model
            model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'best_model.pth'), map_location=device)['model_state_dict'])
            # save best model
            model_path = os.path.join(save_model_path, "best_model.pth")
            torch.save({'model_state_dict': model.state_dict()}, model_path)

            break

    print("Training complete!")
    print(f"results saved in {save_model_path}")

def validate(model, val_loader, criterion, thresh=0.9):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            if 'ResNet' not in args.model and 'resnext' not in args.model:  # If not using ResNet, compute center of mass
                # Compute center of mass for output masks
                com_tensor = torch.stack([
                    torch.stack([center_of_mass(mask, device=device, normalize=False, thresh=thresh) for mask in output])
                    for output in outputs
                ]).to(device)

                loss = criterion(com_tensor, masks)

            else: # resnet does a regression
                loss = criterion(outputs, masks)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

class Tester:
    """
    Evaluates the model using two different criteria and returns two metrics.

    Args:
        criterion1 (callable): First loss function.
        criterion2 (callable): Second loss function.
    """
    
    def _init_(self, criterion1, criterion2, thresh=0.9):
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.thresh = thresh
    
    def _call_(self, model, test_loader, device):
        """
        Runs evaluation on the validation set and computes two metrics.

        Args:
            model (torch.nn.Module): The model to evaluate.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            device (torch.device): Device to run the evaluation on.

        Returns:
            tuple: (metric1, metric2) computed using criterion1 and criterion2.
        """
        model.eval()
        val_loss1 = 0.0
        val_loss2 = 0.0

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                if 'ResNet' not in args.model and 'resnext' not in args.model:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                    com_tensor = torch.stack([
                        torch.stack([center_of_mass(mask, device=device, normalize=False, thresh= self.thresh) for mask in output])
                        for output in outputs
                    ]).to(device)

                    loss1 = self.criterion1(com_tensor, masks)
                    loss2 = self.criterion2(com_tensor, masks)
                else:  # ResNet does a regression                                   
                    loss1 = self.criterion1(outputs, masks)                                       
                    loss2 = self.criterion2(outputs, masks)
                
                val_loss1 += loss1.item()
                val_loss2 += loss2.item()

        avg_val_loss1 = val_loss1 / len(test_loader)
        avg_val_loss2 = val_loss2 / len(test_loader)

        return avg_val_loss1, avg_val_loss2

def main():
    args = parse_args()

    # Initialize Weights & Biases
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.model}{args.loss}_augm{args.augm_version}",  # Set a meaningful run name
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.initial_lr,
            "patience": args.patience,
            "model": "U-Net",
            "augmentation_version": args.augm_version,
            "loss_type": args.loss,
            "seed": args.seed
        }
    )

    # Load dataset
    train_dataset = KeypointDataset(args.train_data, filter=True, model_type=args.model, transform=args.augm_version)
    val_dataset = KeypointDataset(args.val_data, model_type=args.model, filter=True)
    

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, generator=g)

    # Define model
    if args.model == "monai_U-Net":
        model = monai_UNet(start_filts = args.start_filts, depth = args.depth, dropout= args.dropout, num_residuals=args.n_residuals).to(device)
        if args.from_scratch:
            pass
        else:
            model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
    elif args.model == "ResNet50":
        model = ResNet50Regression().to(device)
        if args.from_scratch:
            initialize_weights(model)
        else:
            model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
    elif args.model == "ResNet34":
        model = ResNet34Regression().to(device)
        if args.from_scratch:
            initialize_weights(model)
        else:
            model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
    elif args.model == "ResNet18":
        model = ResNet18Regression().to(device)
        if args.from_scratch:
            initialize_weights(model)
        else:
            model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
    elif args.model == "swinunetr":
        model = SwinUNETR(start_filts=args.start_filts).to(device)
        if args.from_scratch:
            initialize_weights(model)
        else:
            # TODO: load state dict
            pass
    elif args.model == "resnext50":
        model = ResNeXt50Regression().to(device)
        if args.from_scratch:
            initialize_weights(model)
        else:
            model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
    # elif args.model == "improved_unet":
    #     model = ImprovedUnet(in_channels=1, out_channels=args.num_keypoints, max_channels=32).to(device)
    #     if args.from_scratch:
    #         initialize_weights(model)
    #     else:
    #         model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])


    summary(model, input_size=(1, 1, 256, 256))

    # Define loss function and optimizer
    if args.loss == 'ordered_distance':
        criterion = OrderedDistanceLoss()
    if args.loss == 'gaussian':
        criterion = GaussianKeypointLoss(sigma = 20)

    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=1e-5)

    # Train the model
    save_model_path = args.save_model_path if args.save_model_path else get_experiment_path()
    
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs, args.patience, args.checkpoint_path, save_model_path, thresh=args.thresh)

    # Run inference on test set
    model.eval()
    test_path = r'd:\mmissana\data\RV_PATIENTS\dataset_after_review\test.npz'
    test_dataset = KeypointDataset(test_path, filter=True, model_type=args.model)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, generator=g)

    metric2 = OrderedDistanceLoss()
    tester = Tester(metric2, metric2, thresh=args.thresh)
    test_distance, test_MSE = tester(model, test_loader, device=device)

        # Log final test results to wandb
    wandb.summary["test_distance"] = test_distance
    wandb.summary["test_MSE"] = test_MSE

    if args.save_images:
        data = np.load(test_path)
        images = data['images']

        os.makedirs(save_model_path, exist_ok=True)

        for im in images:
            im = np.expand_dims(im, axis=0)
            im = preprocess_images(im, model_type=args.model, device=device)
            im = im.unsqueeze(0)
            output = model(im)
            coordinates_1 = center_of_mass(output[0, 0].detach(), thresh=args.thresh)
            coordinates_2 = center_of_mass(output[0, 1].detach(), thresh=args.thresh)
            coordinates_3 = center_of_mass(output[0, 2].detach(), thresh=args.thresh)

            # Save results
            save_image(im[0, 0, 0].cpu().numpy(), points=[tuple(coordinates_1.tolist()), tuple(coordinates_2.tolist()), tuple(coordinates_3.tolist())], save_folder=save_model_path)

    #Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()