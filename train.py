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
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {total_params:,}")
    
    # Training loop for multiple epochs
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
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate batch statistics
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            running_correct += pred.eq(target).sum().item()
            running_total += target.size(0)
            
            if batch_idx % 100 == 0:
                progress = (batch_idx / total_batches) * 100
                current_accuracy = 100. * running_correct / running_total
                avg_loss = running_loss / (batch_idx + 1)
                print(f"Progress: {progress:.1f}% [{batch_idx}/{total_batches}]")
                print(f"Loss: {avg_loss:.4f}, Training Accuracy: {current_accuracy:.2f}%")
        
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
    