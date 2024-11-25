import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import MNISTModel
import glob
import pytest
import sys

def get_latest_model():
    """Get the most recently trained model file"""
    model_files = glob.glob('model_mnist_*.pth')
    if not model_files:
        raise FileNotFoundError("No model file found")
    latest_model = max(model_files)
    return latest_model

def evaluate_model():
    """Evaluate model performance on test dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
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
    
    with torch.no_grad():
        for data, target in test_loader:
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
    
    return {
        'model_path': model_path,
        'total_images': total,
        'correct_predictions': correct,
        'overall_accuracy': accuracy,
        'class_accuracies': class_accuracies
    }

def test_model_parameter_count():
    """Test 1: Check if model has less than 100,000 parameters"""
    print("\nTest 1: Checking model parameter count...")
    model = MNISTModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    assert total_params < 100000, f"Model has {total_params:,} parameters (should be < 100,000)"
    print("✓ Parameter count test passed!")

def test_input_shape():
    """Test 2: Verify model accepts 28x28 input images"""
    print("\nTest 2: Checking input shape compatibility...")
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)  # Single image, 1 channel, 28x28
    try:
        output = model(test_input)
        print("✓ Input shape test passed!")
    except Exception as e:
        pytest.fail(f"Model failed to process 28x28 input: {str(e)}")

def test_output_dimension():
    """Test 3: Verify model outputs 10 class probabilities"""
    print("\nTest 3: Checking output dimensions...")
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, expected (1, 10)"
    print("✓ Output dimension test passed!")

def test_model_accuracy():
    """Test 4: Verify model achieves >80% accuracy on test set"""
    print("\nTest 4: Checking model accuracy...")
    results = evaluate_model()
    accuracy = results['overall_accuracy']
    print(f"Model accuracy: {accuracy:.2f}%")
    assert accuracy > 80, f"Accuracy is {accuracy:.2f}% (should be > 80%)"
    print("✓ Accuracy test passed!")
    
    # Print detailed results
    print("\nDetailed Test Results:")
    print("="*50)
    print(f"Model path: {results['model_path']}")
    print(f"Total images tested: {results['total_images']}")
    print(f"Correct predictions: {results['correct_predictions']}")
    print("\nPer-class accuracy:")
    for i, acc in enumerate(results['class_accuracies']):
        print(f"Digit {i}: {acc:.2f}%")

if __name__ == '__main__':
    print("\nRunning all model tests...")
    print("="*50)
    pytest.main([__file__, "-v"]) 