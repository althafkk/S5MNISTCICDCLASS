import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import os

def train(num_epochs=1):
    print(f"Training for {num_epochs} epochs...")
    print("\nStarting training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Even more minimal augmentation
    transform_train = transforms.Compose([
        transforms.RandomRotation(2),  # Further reduced rotation
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),  # Minimal translation
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=16,  # Further reduced batch size
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)  # Further reduced label smoothing
    
    # Modified optimizer settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.008,  # Increased learning rate
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5  # Reduced weight decay
    )
    
    # Modified scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.008,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.03,  # Even faster warmup
        div_factor=10,
        final_div_factor=50,
        anneal_strategy='cos'
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {total_params:,}")
    
    print("\nTraining progress:")
    print("="*50)
    
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        total_batches = len(train_loader)
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision training
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Calculate batch statistics
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            running_correct += pred.eq(target).sum().item()
            running_total += target.size(0)
            
            if batch_idx % 20 == 0:
                progress = (batch_idx / total_batches) * 100
                current_accuracy = 100. * running_correct / running_total
                avg_loss = running_loss / (batch_idx + 1)
                current_lr = scheduler.get_last_lr()[0]
                print(f"Progress: {progress:.1f}% [{batch_idx}/{total_batches}]")
                print(f"Loss: {avg_loss:.4f}, Training Accuracy: {current_accuracy:.2f}%")
                print(f"Learning Rate: {current_lr:.6f}")
        
        # Epoch statistics
        epoch_accuracy = 100. * running_correct / running_total
        epoch_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Training Accuracy: {epoch_accuracy:.2f}%")
        
        # Save best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f'model_mnist_{timestamp}_epoch{epoch+1}_acc{epoch_accuracy:.1f}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved as: {save_path}")
    
    print("\n" + "="*50)
    print(f"Training completed!")
    print(f"Best Training Accuracy: {best_accuracy:.2f}%")
    print("="*50)
    
if __name__ == '__main__':
    train(num_epochs=1)
    