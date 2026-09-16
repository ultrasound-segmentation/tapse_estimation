import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb

from temporal_pipeline.postprocessing.coordinates_from_heatmaps import (
    center_of_mass_3d,
    argmax_3d,
)
from temporal_pipeline.utils.plot import save_image, visualize_image

"""class with all the helpers for training"""


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loss_fn: nn.Module,
        val_loss_fn: nn.Module,
        device: str | torch.device = "cpu",
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        checkpoint_dir: str = "checkpoints",
        model_type: str = "3D_UNet",  # just to understand how outputs should be processed
        wandb: bool = False,
        heatmap_training: bool = False,
        train_dataset=None,
        val_dataset=None,
        heatmap_initial_radius: int | None = None,
        heatmap_radius_step: int = 0,
        heatmap_radius_step_epochs: int = 0,
        heatmap_radius_min: int = 1,
    ):
        self.model = model.to(device)
        self.model_type = model_type
        self.optimizer = optimizer
        self.train_loss_fn = train_loss_fn
        self.val_loss_fn = val_loss_fn
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.wandb = wandb
        self.heatmap_training = heatmap_training
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.heatmap_initial_radius = heatmap_initial_radius
        self.heatmap_radius_step = heatmap_radius_step
        self.heatmap_radius_step_epochs = heatmap_radius_step_epochs
        self.heatmap_radius_min = heatmap_radius_min
        self.current_heatmap_radius = heatmap_initial_radius

        self.best_val_loss = float("inf")

        # define the parameters to keep track of during training
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "train_distance_loss": [],
            "train_motion_loss": [],
            "val_motion_loss": [],
        }

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _adjust_heatmap_radius(self, epoch: int) -> None:
        if not self.heatmap_training:
            return

        if self.train_dataset is None or self.heatmap_initial_radius is None:
            return

        if self.heatmap_radius_step_epochs <= 0:
            return

        steps = (epoch - 1) // self.heatmap_radius_step_epochs
        new_radius = max(
            self.heatmap_radius_min,
            self.heatmap_initial_radius - steps * self.heatmap_radius_step,
        )

        if new_radius != self.current_heatmap_radius:
            self.current_heatmap_radius = new_radius
            if hasattr(self.train_dataset, "set_activation_radius"):
                self.train_dataset.set_activation_radius(new_radius)
            elif hasattr(self.train_dataset, "activation_radius"):
                self.train_dataset.activation_radius = int(new_radius)
            print(f"Heatmap radius adjusted to {new_radius} at epoch {epoch}.")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping_patience: int | None = None,
    ) -> dict:
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # print epoch number
            print(f"Epoch {epoch}/{epochs}")

            if self.heatmap_training:
                self._adjust_heatmap_radius(epoch)

            # calculate losses with a progress bar inside the training loop
            train_losses = self._train_one_epoch(
                train_loader, epoch=epoch, total_epochs=epochs
            )

            val_loss = self._evaluate(val_loader)

            if self.scheduler:
                # ReduceLROnPlateau expects a metric; others don't
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # save losses on history
            self.history["train_loss"].append(train_losses["loss"])
            self.history["train_distance_loss"].append(train_losses["dist"])
            self.history["train_motion_loss"].append(train_losses["motion"])
            self.history["val_loss"].append(val_loss)

            # early stopping logic
            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)
                patience_counter = 0
            else:
                patience_counter += 1

            # calculate learning rete
            lr = self.optimizer.param_groups[0]["lr"]
            # save lr in history
            self.history["learning_rate"].append(lr)

            # save logs on wandb when enabled
            if self.wandb and wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_losses["loss"],
                        "val_loss": val_loss,
                        "learning_rate": lr,
                        "train_distance_loss": train_losses["dist"],
                        "train_motion_loss": train_losses["motion"],
                    }
                )

            flag = " ✓" if improved else ""
            print(
                f"Epoch {epoch:>3}/{epochs} | "
                f"train {train_losses['loss']:.4f} | val {val_loss:.4f} | "
                f"lr {lr:.2e}{flag}"
            )

            # stop if patience is exhausted
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        # load the best model after training
        self._load_best_checkpoint()
        return self.history

    # ------------------------------------------------------------------
    # Train / eval steps
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self, loader: DataLoader, epoch: int, total_epochs: int
    ) -> dict:
        self.model.train()
        # total loss
        running_loss = 0.0
        # distance only loss
        train_dist = 0.0
        # motion only loss (difference between frame-by-frame movements)
        train_motion = 0.0

        desc = f"Training Epoch {epoch}/{total_epochs}"
        with tqdm(loader, desc=desc, unit="batch", leave=False) as pbar:
            for images, masks in pbar:
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()

                # for i in range(32):
                #     print(masks.shape)
                #     img = images[0, 0, i].cpu().numpy()
                #     mask = masks[0, 1, i].cpu().numpy()
                #     visualize_image(img + mask)

                # Forward pass
                outputs = self.model(images)

                if (
                    self.model_type in ["3D_UNet", "echocoder"]
                    and not self.heatmap_training
                ):
                    # Compute center of mass for output masks
                    com_tensor = center_of_mass_3d(
                        outputs, device=self.device, normalize=False
                    ).to(self.device)
                    # Compute the loss
                    loss, loss_breakdown = self.train_loss_fn(com_tensor, masks)
                elif self.heatmap_training:
                    # For heatmap training, use mean squared error plus distance error loss
                    loss, loss_breakdown = self.train_loss_fn(outputs, masks)
                    print(loss_breakdown)
                else:
                    raise Exception(
                        "you found a bug? This should never happen, signal it to the developers please"
                    )  # this should never happen

                # Backward pass
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                # save total decomposed loss so as to log it into wandb
                if not self.heatmap_training:
                    train_dist += loss_breakdown["dist"]
                    train_motion += loss_breakdown["motion"]

                pbar.set_postfix(loss=loss.item(), refresh=True)

            # mask = masks[0, 0, 0].detach().cpu().numpy()
            # visualize_image(mask)

            # mask = outputs[0, 0, 0].detach().cpu().numpy()
            # visualize_image(mask)

        # calculate avg decomposed loss
        avg_loss = running_loss / len(loader)
        avg_train_dist = train_dist / len(loader)
        avg_train_motion = train_motion / len(loader)

        losses = {"loss": avg_loss, "dist": avg_train_dist, "motion": avg_train_motion}

        return losses

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        for images, masks in loader:

            # put on device
            images, masks = images.to(self.device), masks.to(self.device)

            # for i in range(32):
            #         print(masks.shape)
            #         img = images[0, 0, i].cpu().numpy()
            #         mask = masks[0, 1, i].cpu().numpy()
            #         visualize_image(img + mask)

            # compute outputs
            outputs = self.model(images)

            if (
                self.model_type in ["3D_UNet", "echocoder"]
                and not self.heatmap_training
            ):
                # Compute center of mass for output masks
                com_tensor = center_of_mass_3d(
                    outputs, device=self.device, normalize=False
                ).to(self.device)
                # calculate the loss: for the validation I use the distance loss, that's
                # what I want to minimize
                loss, _ = self.val_loss_fn(com_tensor, masks)
            elif self.heatmap_training:
                # For validation, I prefer using the classic euclidean distance metric
                outputs = argmax_3d(outputs, device=self.device)
                loss, _ = self.val_loss_fn(outputs, masks)
            else:
                raise Exception(
                    "you found a bug? This should never happen, signal it to the developers please"
                )  # this should never happen

            total_loss += loss.item()

        return total_loss / len(loader)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.checkpoint_dir / "best.pt"
        torch.save(state, path)

    def _load_best_checkpoint(self) -> None:
        path = self.checkpoint_dir / "best.pt"
        if path.exists():
            state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            print(
                f"Loaded best checkpoint (epoch {state['epoch']}, val loss {state['val_loss']:.4f})"
            )


# ----------------------------------------------------------------------
# Usage example
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import TensorDataset

    # --- Toy data ---
    X = torch.randn(1000, 16)
    y = (X[:, 0] > 0).long()

    split = 800
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds = TensorDataset(X[split:], y[split:])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # --- Model ---
    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )

    # --- Trainer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device="cpu",
        scheduler=scheduler,
        checkpoint_dir="checkpoints",
    )

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=20,
        early_stopping_patience=5,
    )
