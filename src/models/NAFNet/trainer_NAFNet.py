import torch
from src.models.NAFNet.loss_metrics_NAFNet import nafnet_loss, batch_metrics


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Train the NAFNet denoiser for one epoch.

    NAFNet predicts the clean image directly, so the loss compares the predicted
    denoised image with the real clean image. Returns the average loss over all
    batches.
    """

    model.train()
    total_loss = 0.0

    for noisy, clean in dataloader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        denoised = model(noisy)
        loss = nafnet_loss(denoised, clean)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    """
    Validate the NAFNet denoiser.

    The model output is clamped to [0, 1] inside batch_metrics before the image
    quality metrics are calculated, because valid image pixel values should stay
    in this range. Returns the average loss and metrics over all batches.
    """

    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_psnr = 0.0

    for noisy, clean in dataloader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        denoised = model(noisy)
        loss = nafnet_loss(denoised, clean)

        metrics = batch_metrics(denoised, clean)

        total_loss += loss.item()
        total_mse += metrics["mse"]
        total_mae += metrics["mae"]
        total_psnr += metrics["psnr"]

    num_batches = len(dataloader)

    return {
        "loss": total_loss / num_batches,
        "mse": total_mse / num_batches,
        "mae": total_mae / num_batches,
        "psnr": total_psnr / num_batches,
    }


def fit(model, train_loader, val_loader, optimizer, device, epochs, save_path=None):
    """
    Train and validate the NAFNet denoiser for multiple epochs.

    This function stores the results from each epoch, which makes it easier to
    plot learning curves and compare this NAFNet model with the CNN and DnCNN
    models later. If save_path is given, the model with the best validation PSNR
    is saved there.

    Args:
        model: NAFNet denoising model
        train_loader: dataloader for training data
        val_loader: dataloader for validation data
        optimizer: optimizer used for training
        device: cuda, mps or cpu device
        epochs: number of training epochs
        save_path: where to save the best model. None disables saving.
    Returns:
        list of dictionaries containing train and validation results
    """

    history = []
    best_psnr = 0.0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)

        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_psnr": val_metrics["psnr"],
        }
        history.append(epoch_results)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_metrics['loss']:.6f} "
            f"Val PSNR: {val_metrics['psnr']:.2f} dB"
        )

        if save_path is not None and val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved (PSNR: {best_psnr:.2f} dB)")

    return history