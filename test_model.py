import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import MNISTModel
import glob
import pytest
import sys
import numpy as np

# Add this to prevent pytest from capturing output
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def pytest_configure(config):
    config.option.capture = 'no'

def print_summary_box(title, content_dict):
    """Helper function to print formatted summary box"""
    print("\n" + "="*50, flush=True)
    print(f" {title} ".center(50, "="), flush=True)
    print("="*50, flush=True)
    for key, value in content_dict.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}", flush=True)
        else:
            print(f"{key}: {value}", flush=True)
    print("="*50, flush=True)

def get_latest_model():
    """Get the most recently trained model file"""
    model_files = glob.glob('model_mnist_*.pth')
    if not model_files:
        raise FileNotFoundError("No model file found")
    latest_model = max(model_files)
    return latest_model

def evaluate_model():
    """Evaluate model performance on test dataset"""
    print("\nStarting model evaluation...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    model = MNISTModel().to(device)
    
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print_summary_box("MODEL ARCHITECTURE", {
        "Model Path": model_path,
        "Total Parameters": f"{total_params:,}",
        "Parameter Limit": "25,000",
        "Status": "✓ PASSED" if total_params < 25000 else "✗ FAILED"
    })
    
    if total_params >= 25000:
        raise ValueError(f"Model has {total_params:,} parameters (should be < 25,000)")
    
    # Evaluate accuracy
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    print("\nEvaluating model on test set...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += (pred[i] == label).item()
                class_total[label] += 1
    
    accuracy = 100. * correct / total
    class_accuracies = [100 * class_correct[i] / class_total[i] for i in range(10)]
    
    print_summary_box("MODEL PERFORMANCE", {
        "Test Accuracy": accuracy,
        "Required Accuracy": 95.00,
        "Total Images": total,
        "Correct Predictions": correct,
        "Status": "✓ PASSED" if accuracy >= 95 else "✗ FAILED"
    })
    
    print("\nPer-class Accuracy:")
    print("-" * 30)
    for i, acc in enumerate(class_accuracies):
        print(f"Digit {i}: {acc:.2f}%")
    print("-" * 30)
    
    if accuracy < 95:
        raise ValueError(f"Model accuracy is {accuracy:.2f}% (should be > 95%)")
    
    return {
        'model_path': model_path,
        'total_params': total_params,
        'total_images': total,
        'correct_predictions': correct,
        'overall_accuracy': accuracy,
        'class_accuracies': class_accuracies
    }

def test_model_parameter_count():
    """Test 1: Check if model has less than 25,000 parameters"""
    print("\nTest 1: Checking model parameter count...", flush=True)
    
    # Run multiple evaluations
    num_runs = 5
    param_counts = []
    
    for i in range(num_runs):
        print(f"\nRun {i+1}/{num_runs}", flush=True)
        model = MNISTModel()
        total_params = sum(p.numel() for p in model.parameters())
        param_counts.append(total_params)
        
    avg_params = np.mean(param_counts)
    std_params = np.std(param_counts)
    
    print_summary_box("PARAMETER COUNT STATISTICS", {
        "Average Parameters": f"{avg_params:,.0f}",
        "Standard Deviation": f"{std_params:,.2f}",
        "Minimum": f"{min(param_counts):,}",
        "Maximum": f"{max(param_counts):,}",
        "Status": "✓ PASSED" if avg_params < 25000 else "✗ FAILED"
    })
    
    assert avg_params < 25000, f"Model has {avg_params:,.0f} parameters (should be < 25,000)"
    print("✓ Parameter count test passed!", flush=True)

def test_model_accuracy():
    """Test 2: Verify model achieves >95% accuracy and has <25,000 parameters"""
    print("\nTest 2: Checking model accuracy and parameters...", flush=True)
    
    # Run multiple evaluations
    num_runs = 3
    accuracies = []
    
    try:
        for i in range(num_runs):
            print(f"\nAccuracy Test Run {i+1}/{num_runs}", flush=True)
            results = evaluate_model()
            accuracies.append(results['overall_accuracy'])
        
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        print_summary_box("ACCURACY STATISTICS", {
            "Average Accuracy": avg_accuracy,
            "Standard Deviation": std_accuracy,
            "Minimum": min(accuracies),
            "Maximum": max(accuracies),
            "Status": "✓ PASSED" if avg_accuracy >= 95 else "✗ FAILED"
        })
        
        print("\n✓ All model tests passed!", flush=True)
    except ValueError as e:
        pytest.fail(str(e))

if __name__ == '__main__':
    print("\nRunning all model tests...", flush=True)
    print("="*50, flush=True)
    # Run pytest with -s flag to show output
    pytest.main([__file__, "-v", "-s"]) 