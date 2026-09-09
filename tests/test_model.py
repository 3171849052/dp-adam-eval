import torch
from dp_wiener_mnist.model import SimpleCNN


def test_architecture():
    m = SimpleCNN()
    assert m.conv1.weight.shape == (16, 1, 3, 3)
    assert m.conv2.weight.shape == (32, 16, 3, 3)
    assert m.fc1.weight.shape == (128, 1568)
    assert m.fc2.weight.shape == (10, 128)
    assert m(torch.zeros(4, 1, 28, 28)).shape == (4, 10)
    assert not any(
        isinstance(x, (torch.nn.Dropout, torch.nn.BatchNorm2d)) for x in m.modules()
    )
