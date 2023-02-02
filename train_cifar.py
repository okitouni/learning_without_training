import copy
import torch
import torch.nn as nn
import numpy as np
import cifar_model as model
from data import load_cifar


def train_cifar(args):
    # Configurable parameters
    device = torch.device(args.DEV) if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16 if device.type != "cpu" else torch.float32

    # First, the learning rate rises from 0 to 0.002 for the first 194 batches.
    # Next, the learning rate shrinks down to 0.0002 over the next 582 batches.
    lr_schedule = torch.cat(
        [torch.linspace(0e0, 2e-3, 194), torch.linspace(2e-3, 2e-4, 582),]
    )

    lr_schedule_bias = args.LR_BIAS_MULTIPLIER * lr_schedule
    lr_schedule *= args.LR_MULTIPLIER

    # Set random seed to increase chance of reproducability
    torch.manual_seed(args.SEED)

    # Load dataset
    train_data, train_targets, valid_data, valid_targets = load_cifar(device, dtype)

    # Compute special weights for first layer
    weights = model.patch_whitening(train_data[:10000, :, 4:-4, 4:-4])

    # Construct the neural network
    train_model = model.Model(
        weights, c_in=3, c_out=10, scale_out=args.SCALE_OUT, masking=args.MASKING
    )

    # Convert model weights to half precision
    train_model.to(dtype)

    # Convert BatchNorm back to single precision for better accuracy
    for module in train_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.float()

    # Upload model to GPU
    train_model.to(device)

    # Collect weights and biases and create nesterov velocity values
    weights = [
        (w, torch.zeros_like(w))
        for w in train_model.parameters()
        if w.requires_grad and len(w.shape) > 1
    ]
    n_weights = np.sum([w.numel() for w, _ in weights])
    biases = [
        (w, torch.zeros_like(w))
        for w in train_model.parameters()
        if w.requires_grad and len(w.shape) <= 1
    ]

    # Copy the model for validation
    valid_model = copy.deepcopy(train_model)

    # Train and validate
    print("\nepoch  batch  train time [sec]  validation accuracy  train accuracy  weight norm  max weight  min weight  weight var  frac_active")
    batch_count = 0
    for epoch in range(1, args.EPOCHS + 1):

        # Randomly shuffle training data
        indices = torch.randperm(len(train_data), device=device)
        data = train_data[indices]
        targets = train_targets[indices]

        # Crop random 32x32 patches from 40x40 training data
        data = [
            random_crop(data[i : i + args.BATCH_SIZE], crop_size=(32, 32))
            for i in range(0, len(data), args.BATCH_SIZE)
        ]
        data = torch.cat(data)

        # Randomly flip half the training data
        data[: len(data) // 2] = torch.flip(data[: len(data) // 2], [-1])
        train_correct = []
        for i in range(0, len(data), args.BATCH_SIZE):
            # discard partial batches
            if i + args.BATCH_SIZE > len(data):
                break

            # Slice batch from data
            inputs = data[i : i + args.BATCH_SIZE]
            target = targets[i : i + args.BATCH_SIZE]
            batch_count += 1

            # Compute new gradients
            train_model.zero_grad()
            train_model.train(True)

            logits = train_model(inputs)

            loss = model.label_smoothing_loss(logits, target, alpha=args.ALPHA_SMOOTHING)

            # accuracy
            correct = torch.eq(torch.argmax(logits, dim=1), target).detach().float()
            train_correct.append(correct)

            loss.sum().backward()

            lr_index = min(batch_count, len(lr_schedule) - 1)
            lr = lr_schedule[lr_index]
            lr_bias = lr_schedule_bias[lr_index]

            # Update weights and biases of training model
            update_nesterov(weights, lr, args.WEIGHT_DECAY, args.MOMENTUM)
            update_nesterov(biases, lr_bias, args.WEIGHT_DECAY_BIAS, args.MOMENTUM)

            # Update validation model with exponential moving averages
            if (i // args.BATCH_SIZE % args.EMA_UPDATE_FREQ) == 0:
                # valid_model.load_state_dict(train_model.state_dict())
                update_ema(train_model, valid_model, args.EMA_RHO ** args.EMA_UPDATE_FREQ)

        # Add training time

        valid_correct = []
        for i in range(0, len(valid_data), args.BATCH_SIZE):
            valid_model.train(False)

            # Test time agumentation: Test model on regular and flipped data
            regular_inputs = valid_data[i : i + args.BATCH_SIZE]
            flipped_inputs = torch.flip(regular_inputs, [-1])

            logits1 = valid_model(regular_inputs).detach()
            logits2 = valid_model(flipped_inputs).detach()

            # Final logits are average of augmented logits
            logits = torch.mean(torch.stack([logits1, logits2], dim=0), dim=0)

            # Compute correct predictions
            correct = logits.max(dim=1)[1] == valid_targets[i : i + args.BATCH_SIZE]

            valid_correct.append(correct.detach().type(torch.float64))

        # Accuracy is average number of correct predictions
        train_acc = torch.mean(torch.cat(train_correct)).item()
        valid_acc = torch.mean(torch.cat(valid_correct)).item()

        # monitor average weight norm
        weights_norm = np.sum([w.float().abs().sum().item() for w,_ in weights]) / n_weights
        weights_max = np.max([w.abs().max().item() for w,_ in weights])
        weights_min = np.min([w.abs().min().item() for w,_ in weights])
        weights_var = np.mean([w.var().item() for w,_ in weights])
        active_fraction = np.sum([(w > .5).sum().item() for w,_ in weights]) / n_weights

        print(f"{epoch:5} {batch_count:6d} {valid_acc:20.4f} {train_acc:15.4f} {weights_norm:12.4f} {weights_max:11.4f} {weights_min:11.4f} {weights_var:11.4f} {active_fraction:12.4f}")

    return valid_acc


def update_ema(train_model, valid_model, rho):
    # The trained model is not used for validation directly. Instead, the
    # validation model weights are updated with exponential moving averages.
    train_weights = train_model.state_dict().values()
    valid_weights = valid_model.state_dict().values()
    for train_weight, valid_weight in zip(train_weights, valid_weights):
        if valid_weight.dtype in [torch.float16, torch.float32]:
            valid_weight *= rho
            valid_weight += (1 - rho) * train_weight


def update_nesterov(weights, lr, weight_decay, momentum):
    for weight, velocity in weights:
        if weight.requires_grad:
            gradient = weight.grad.data
            weight = weight.data

            gradient.add_(weight, alpha=weight_decay).mul_(-lr)
            velocity.mul_(momentum).add_(gradient)
            weight.add_(gradient.add_(velocity, alpha=momentum))


def random_crop(data, crop_size):
    crop_h, crop_w = crop_size
    h = data.size(2)
    w = data.size(3)
    x = torch.randint(w - crop_w, size=(1,))[0]
    y = torch.randint(h - crop_h, size=(1,))[0]
    return data[:, :, y : y + crop_h, x : x + crop_w]
