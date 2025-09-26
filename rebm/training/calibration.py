import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

from rebm.training.eval_utils import eval_acc
from rebm.utils import assert_no_grad


class VectorScaling(nn.Module):
    def __init__(self, model, num_classes):
        super(VectorScaling, self).__init__()
        assert_no_grad(model)
        assert not model.training
        self.model = model
        self.weights = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

        # # Freeze the original model's parameters
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward_no_base_model_grad(self, x, y=None):
        assert not self.model.training
        with torch.no_grad():  # Ensure no gradients flow to the base model
            logits = self.model(x=x, y=y)
        return logits * self.weights + self.bias

    def forward(self, x, y=None):
        assert not self.model.training
        # Allow gradients to flow to the base model
        logits = self.model(x=x, y=y)
        return logits * self.weights + self.bias


def calibrate_for_accuracy(
    model, train_loader, valid_loader, device, num_classes, lr=0.01, epochs=10
):
    """
    Calibrates model to optimize for classification accuracy.

    Args:
        model: Base model to calibrate
        train_loader: DataLoader with training data
        valid_loader: DataLoader with validation data
        device: Device to run the model on
        num_classes: Number of output classes
        lr: Learning rate for optimizer
        epochs: Number of training epochs

    Returns:
        Calibrated model with best validation accuracy
    """
    calibrated_model = VectorScaling(model, num_classes).to(device)
    optimizer = optim.Adam(
        [
            {"params": calibrated_model.weights},
            {"params": calibrated_model.bias},
        ],
        lr=lr,
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Training phase
        train_loss = 0.0
        train_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = calibrated_model.forward_no_base_model_grad(
                x=inputs, y=None
            )
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            assert_no_grad(model)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)

        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0

        calibrated_model.zero_grad()

        # Validation phase
        total = 0
        correct = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = calibrated_model(x=inputs, y=None)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = correct / total if total > 0 else 0
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Accuracy: {accuracy:.4f}"
        )

        if accuracy > best_acc:
            best_acc = accuracy
            best_state = {
                "weights": calibrated_model.weights.clone(),
                "bias": calibrated_model.bias.clone(),
            }

    # Load the best parameters
    if best_state is not None:
        calibrated_model.weights.data = best_state["weights"]
        calibrated_model.bias.data = best_state["bias"]

    print(f"Best validation accuracy: {best_acc:.4f}")

    return calibrated_model


def eval_calibration(
    model, train_set, test_set, device, batch_size=256, lr=0.01, epochs=2
):
    """
    Calibrate a model and evaluate its post-calibration accuracy on training and test data.

    Args:
        model: The model to calibrate and evaluate
        train_set: Training dataset
        test_set: Test dataset
        device: Device to run the model on
        batch_size: Maximum batch size for dataloaders (will be reduced if necessary)
        lr: Learning rate for calibration
        epochs: Number of epochs for calibration

    Returns:
        tuple: (calibrated_model, train_acc, test_acc) - The calibrated model and post-calibration
               accuracies for training and test sets
    """
    # Infer num_classes from the dataset
    if hasattr(train_set, "classes"):
        num_classes = len(train_set.classes)
    else:
        raise ValueError(
            "Could not infer num_classes from dataset. The dataset must have a 'classes' attribute."
        )

    # Create a train/validation split from the training set
    train_size = int(0.8 * len(train_set))
    valid_size = len(train_set) - train_size

    # Use random_split to create the split with a fixed seed
    train_subset, valid_subset = data_utils.random_split(
        train_set,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(42),  # Use a fixed seed
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_subset, batch_size=batch_size, shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True
    )

    assert_no_grad(model)
    assert not model.training

    # Calibrate the model
    print("Calibrating model...")
    calibrated_model = calibrate_for_accuracy(
        model=model,
        train_loader=train_loader,  # Training subset for training
        valid_loader=valid_loader,  # Validation subset for evaluation
        device=device,
        num_classes=num_classes,
        lr=lr,
        epochs=epochs,
    )
    calibrated_model.eval()

    # Calculate accuracy after calibration using eval_acc from eval_utils
    print("Calculating post-calibration accuracy...")
    train_acc = eval_acc(calibrated_model, train_loader, device)
    test_acc = eval_acc(calibrated_model, test_loader, device)
    print(
        f"Post-calibration - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
    )

    return calibrated_model, train_acc, test_acc
