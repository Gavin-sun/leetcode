import torchvision

from torchvision.datasets import  MNIST
from torchvision import transforms

mnist = MNIST(root= r"E:\git_code\Machine_learning\resource", train=True, download=False)
print(mnist[0][0].show())

ret = transforms.ToTensor()(mnist[0][0])
print(ret.size())
