# import os
# from torchvision import datasets, transforms
# from torch.utils.data import Dataset
# from .dataset_base import DatasetBase

# class Fruits(DatasetBase):
#     def __init__(self, root, session_id=0):
#         super(Fruits, self).__init__(root=root, name='fruits')

#         session_folder = f"session_{session_id}"
#         # self.data_path = os.path.join(root, session_folder)
        
#         self.data_path = "fruits_data"
        
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor()
#         ])

#         # Load bằng ImageFolder
#         self.dataset = datasets.ImageFolder(self.data_path, transform=self.transform)

#         self.classes = self.dataset.classes     
#         self.class_to_idx = self.dataset.class_to_idx  # nếu cần map ngược

#         self.gpt_prompt_path = f'description/fruits_prompts_full.json'

#     def get_class_name(self):
#         return self.classes

#     def get_train_data(self):
#         imgs, labels = zip(*self.dataset)
#         return imgs, labels

#     def get_test_data(self):
#         imgs, labels = zip(*self.dataset)
#         return imgs, labels
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split
from .dataset_base import DatasetBase

class Fruits(DatasetBase):
    def __init__(self, root, session_id=0, train_split=0.8):
        super(Fruits, self).__init__(root=root, name='fruits')
        
        self.data_path = "fruits_data"
        
        # Simplified transforms without augmentation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # CLIP requires 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])

        # Load dataset
        full_dataset = datasets.ImageFolder(
            self.data_path,
            transform=self.transform
        )
        
        # Calculate splits
        total_size = len(full_dataset)
        train_size = int(train_split * total_size)
        test_size = total_size - train_size
        
        # Split dataset
        self.train_dataset, self.test_dataset = random_split(
            full_dataset, 
            [train_size, test_size]
        )

        # Store class information
        self.classes = full_dataset.classes
        self.class_to_idx = full_dataset.class_to_idx
        
        self.gpt_prompt_path = 'description/fruits_prompts_full.json'
        
        print(f"Dataset loaded: {len(self.train_dataset)} training, {len(self.test_dataset)} testing")
        print(f"Number of classes: {len(self.classes)}")

    def get_class_name(self):
        return self.classes

    def get_train_data(self):
        imgs = []
        labels = []
        for img, label in self.train_dataset:
            imgs.append(img)
            labels.append(label)
        return imgs, labels

    def get_test_data(self):
        imgs = []
        labels = []
        for img, label in self.test_dataset:
            imgs.append(img)
            labels.append(label)
        return imgs, labels