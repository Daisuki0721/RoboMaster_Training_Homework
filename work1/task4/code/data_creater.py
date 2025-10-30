import os
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class ArmorDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        """
        Dogs-vs-cats dataset
        Args:
            data_dir: Image dataset directory
            transform: Image transform
        """
        self.data_dir = data_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {'1': 0, '2': 1, '3':2, '4':3, '5':4, 'base':5, 'outpost':6, 'sentry':7}

       # Iterate through each class folder
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(data_dir, class_name)

            # Check if the class folder exists
            if not os.path.isdir(class_dir):
                print(f"Warning: Class folder '{class_name}' not found in {data_dir}")
                continue

            # Get all image files in the class folder
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, filename)
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def create_datasets(data_dir, transform = None, test_size = 0.2, random_seed = 42):
    """
    创建训练集和测试集
    Args:
        data_dir: Image dataset directory
        transform: Image transform
        test_size: test ratio (0-1)
        random_seed: random_seed
    Returns:
        train_dataset, test_dataset
    """

    full_dataset = ArmorDataset(data_dir, transform = transform)

    # 划分训练集和测试集 (使用分层抽样保持类别比例)
    indices = list(range(len(full_dataset)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=full_dataset.labels,  # 保持类别分布
        random_state=random_seed
    )

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    return train_dataset, test_dataset


if __name__ == "__main__":
    DATA_DIR = "./armor_8c_new"
    BATCH_SIZE = 64
    TEST_RATIO = 0.2  # 20%作为测试集
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_set, test_set = create_datasets(DATA_DIR, transform = data_transform,  test_size = TEST_RATIO)

    train_loader = DataLoader(
        train_set,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 2
    )

    test_loader = DataLoader(
        test_set,
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = 2
    )

    print(f"train_data size: {len(train_set)}")
    print(f"test_data size: {len(test_set)}")

    images, labels = next(iter(train_loader))
    print(f"Image shape: {images.shape}")
    print(f"Label shape: {labels.shape}")
