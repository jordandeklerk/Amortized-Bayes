from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import SVHN
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from src.utils.parser import args

dataset_path = '~/datasets'

dataset_configs = {
    "MNIST": {"x_dim": 28 * 28, "hidden_dim": 400, "latent_dim": 200, "dataset": MNIST},
    "FashionMNIST": {"x_dim": 28 * 28, "hidden_dim": 400, "latent_dim": 200, "dataset": FashionMNIST},
    "CIFAR10": {"x_dim": 32 * 32 * 3, "hidden_dim": 128, "latent_dim": 100, "dataset": CIFAR10},
    "SVHN": {"x_dim": 32 * 32, "hidden_dim": 128, "latent_dim": 100, "dataset": SVHN},
}

transform = transforms.Compose([transforms.ToTensor()])
kwargs = {'num_workers': 4, 'pin_memory': True}

def get_dataset(data_set_name, dataset_path, transform, **kwargs):
    if data_set_name not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {data_set_name}")

    config = dataset_configs[data_set_name]
    dataset_class = config["dataset"]

    if data_set_name == "SVHN":
        train_dataset = dataset_class(root=dataset_path, split='train', transform=transform, download=True)
        test_dataset = dataset_class(root=dataset_path, split='test', transform=transform, download=True)
    else:
        train_dataset = dataset_class(root=dataset_path, train=True, transform=transform, download=True)
        test_dataset = dataset_class(root=dataset_path, train=False, transform=transform, download=True)

    return train_dataset, test_dataset

train_dataset, test_dataset = get_dataset(args.dataset, dataset_path, transform, **kwargs)

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)