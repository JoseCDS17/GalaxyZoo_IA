from collections.abc import Callable
from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def perform_train_loop(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: Optimizer,
    train_loader: DataLoader,
    task: Literal["classification", "binary_class", "regression"]
) -> tuple[float, list[int], list[int]]:
    """Utility function to perform a training loop.

    Args:
        model: The model to train.
        loss_fn: The loss function to use for training.
        optimizer: The optimizer to use for training.
        train_loader: The DataLoader to use for training.

    Returns:
        A tuple containing the total loss, predictions, and targets for the training loop.

    """
    predictions = []
    targets = []
    total_loss = 0.0
    for X, y in train_loader:
        # Compute prediction and loss
        if task == "binary_class":
            pred = model(X).flatten()  
        else:  
            pred = model(X)
        loss = loss_fn(pred, y)

        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save the predictions and targets
        if task == "classification":
            predictions += pred.argmax(dim=1).tolist()
        elif task == "binary_class":
            predictions += (pred > 0).int().tolist()
        else:
            predictions += pred.tolist()
        targets += y.tolist()

    return total_loss, predictions, targets


def perform_validation_loop(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    validation_loader: DataLoader,
    task: Literal["classification", "binary_class", "regression"]
) -> tuple[float, list[int], list[int]]:
    """Utility function to perform a validation loop.

    Args:
        model: The model to validate.
        loss_fn: The loss function to use for validation.
        validation_loader: The DataLoader to use for validation.

    Returns:
        A tuple containing the total loss, predictions, and targets for the validation loop.
    """
    predictions = []
    targets = []
    total_loss = 0.0
    with (
        torch.no_grad()
    ):  # No need to store the computation graph, since we're not using it for autograd.
        for X, y in validation_loader:
            if task == "binary_class":
                pred = model(X).flatten()
            else:
                pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()

            if task == "classification":
                predictions += pred.argmax(dim=1).tolist()
            elif task == "binary_class":
                predictions += (pred > 0).int().tolist()
            else:
                predictions += pred.tolist()
            targets += y.tolist()

    return total_loss, predictions, targets


def print_box(*strings: str):
    max_length = max(map(len, strings))
    print("-" * (max_length + 4))
    for string in strings:
        print(f"| {string}{' ' * (max_length - len(string))} |")
    print("-" * (max_length + 4))


def print_log(
    train_loss: float,
    val_loss: float,
    train_accuracy: float,
    val_accuracy: float,
    lr: float,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epoch: int,
):
    train_loss_avg = train_loss / len(train_loader)
    val_loss_avg = val_loss / len(validation_loader)
    train_loss_str = f"Train loss: {train_loss_avg:.4f}"
    val_loss_str = f"Validation loss: {val_loss_avg:.4f}"
    train_accuracy_str = f"Train accuracy: {train_accuracy * 100:.2f}%"
    val_accuracy_str = f"Validation accuracy: {val_accuracy * 100:.2f}%"
    lr_str = f"Learning rate: {lr:.2e}"
    epoch_str = f"Epoch {epoch + 1}"

    print_box(
        epoch_str, train_loss_str, val_loss_str, train_accuracy_str, val_accuracy_str, lr_str
    )


def fit(
    epochs: int,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    include_accuracy: bool = True,
    scheduler = None,
    task: Literal["classification", "binary_class", "regression"] = "classification"
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Utility function to fit a model to the training data and evaluate it on the validation data.

    Args:
        epochs: The number of epochs to train for.
        loss_function: The loss function to use for training and validation.
        model: The model to train and validate.
        optimizer: The optimizer to use for training.
        train_loader: The DataLoader to use for training.
        validation_loader: The DataLoader to use for validation.
        include_accuracy: Whether to compute and include accuracy in the logs and return values. If False, the function will return empty lists for accuracies.
    """
    train_losses = []
    val_losses = []
    if include_accuracy:
        train_accuracies = []
        val_accuracies = []

    for epoch in range(epochs):
        train_loss, train_predictions, train_targets = perform_train_loop(
            model, loss_function, optimizer, train_loader, task=task
        )
        val_loss, val_predictions, val_targets = perform_validation_loop(
            model, loss_function, validation_loader, task=task
        )

        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(validation_loader)

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)

        if scheduler is not None:
            scheduler.step(val_loss_avg)

        if include_accuracy:
            train_accuracy = sum(
                [1 for i, j in zip(train_predictions, train_targets) if i == j]
            ) / len(train_predictions)
            val_accuracy = sum(
                [1 for i, j in zip(val_predictions, val_targets) if i == j]
            ) / len(val_predictions)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

        if include_accuracy:
            print_log(
                train_loss,
                val_loss,
                train_accuracy,
                val_accuracy,
                optimizer.param_groups[0]["lr"],
                train_loader,
                validation_loader,
                epoch,
            )
        else:
            print_box(
                f"Epoch {epoch + 1}",
                f"Train loss: {train_loss / len(train_loader)}",
                f"Validation loss: {val_loss / len(validation_loader)}",
                f"Learning rate: {optimizer.param_groups[0]["lr"]:.2e}"
                ,
            )
    if include_accuracy:
        return train_losses, val_losses, train_accuracies, val_accuracies
    else:
        return train_losses, val_losses, [], []
