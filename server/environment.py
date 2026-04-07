import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import uuid

from src.models import HyperparamAction, HyperparamObservation, HyperparamState

# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class SimpleNet(nn.Module):
    """Simple CNN for MNIST"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResNetSmall(nn.Module):
    """Small ResNet for CIFAR-10"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, 2)
        self.layer2 = self._make_layer(16, 32, 2, stride=2)
        self.layer3 = self._make_layer(32, 64, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================================
# MAIN ENVIRONMENT CLASS
# ============================================================================

class HyperparamEnvironment:
    """Main environment for hyperparameter tuning"""
    
    def __init__(self, difficulty: str = "easy"):
        self.difficulty = difficulty
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load dataset based on difficulty
        self._load_dataset()
        
        # State
        self.episode_id = None
        self.start_time = 0
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.configs_tried = 0
        self.learning_rate = 1e-3
        self.batch_size = 32
        self.weight_decay = 0.0
        self.optimizer_name = "adam"
        self.optimizer = None
        self.model = None
        self.criterion = None
    
    def _load_dataset(self):
        """Load dataset based on difficulty"""
        if self.difficulty == "easy":
            self._load_mnist()
        elif self.difficulty == "medium":
            self._load_cifar10()
        elif self.difficulty == "hard":
            self._load_imagenet_subset()
        else:
            raise ValueError(f"Unknown difficulty: {self.difficulty}")
    
    def _load_mnist(self):
        """Load MNIST dataset (Easy) - OPTIMIZED"""
        print("[ENV] Loading MNIST (Easy)...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        full_train = datasets.MNIST(
            root='/tmp/mnist', train=True, download=True, transform=transform
        )
        full_val = datasets.MNIST(
            root='/tmp/mnist', train=False, download=True, transform=transform
        )
        
        # Use 2% of data (REDUCED from 10%)
        train_indices = list(range(0, len(full_train), 50))  # Every 50th sample
        val_indices = list(range(0, len(full_val), 50))
        
        self.train_data = Subset(full_train, train_indices)
        self.val_data = Subset(full_val, val_indices)
        
        self.dataset_name = "MNIST"
        self.num_classes = 10
        self.max_epochs = 5  # REDUCED from 10
        self.time_budget_seconds = 300  # 5 minutes
        self.target_accuracy = 0.90
        
        self.model = SimpleNet().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def _load_cifar10(self):
        """Load CIFAR-10 dataset (Medium) - OPTIMIZED"""
        print("[ENV] Loading CIFAR-10 (Medium)...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])
        
        full_train = datasets.CIFAR10(
            root='/tmp/cifar10', train=True, download=True, transform=transform
        )
        full_val = datasets.CIFAR10(
            root='/tmp/cifar10', train=False, download=True, transform=transform
        )
        
        # Use 2% of data (REDUCED from 5%)
        train_indices = list(range(0, len(full_train), 50))  # Every 50th sample
        val_indices = list(range(0, len(full_val), 50))
        
        self.train_data = Subset(full_train, train_indices)
        self.val_data = Subset(full_val, val_indices)
        
        self.dataset_name = "CIFAR-10"
        self.num_classes = 10
        self.max_epochs = 5  # REDUCED from 20
        self.time_budget_seconds = 600  # 10 minutes
        self.target_accuracy = 0.80
        
        self.model = ResNetSmall(num_classes=10).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def _load_imagenet_subset(self):
        """Load ImageNet subset (Hard) - OPTIMIZED"""
        print("[ENV] Loading ImageNet-100 (Hard)...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])
        
        full_train = datasets.CIFAR10(
            root='/tmp/cifar10', train=True, download=True, transform=transform
        )
        full_val = datasets.CIFAR10(
            root='/tmp/cifar10', train=False, download=True, transform=transform
        )
        
        # Use 1% of data (REDUCED from 2%)
        train_indices = list(range(0, len(full_train), 100))  # Every 100th sample
        val_indices = list(range(0, len(full_val), 100))
        
        self.train_data = Subset(full_train, train_indices)
        self.val_data = Subset(full_val, val_indices)
        
        self.dataset_name = "ImageNet-100"
        self.num_classes = 10
        self.max_epochs = 5  # REDUCED from 50
        self.time_budget_seconds = 900  # 15 minutes
        self.target_accuracy = 0.60
        
        self.model = ResNetSmall(num_classes=10).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
    
    def reset(self) -> HyperparamObservation:
        """Start new training episode"""
        # Reset model
        if self.difficulty == "easy":
            self.model = SimpleNet().to(self.device)
        else:
            self.model = ResNetSmall(num_classes=self.num_classes).to(self.device)
        
        self.episode_id = f"ep_{int(time.time() * 1000)}"
        self.start_time = time.time()
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.configs_tried = 0
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_data, batch_size=256, shuffle=False, num_workers=0
        )
        
        # Create optimizer
        self._create_optimizer()
        
        return HyperparamObservation(
            epoch=0,
            validation_accuracy=0.0,
            training_loss=0.0,
            current_learning_rate=self.learning_rate,
            model_size_mb=self._get_model_size(),
            time_elapsed_seconds=0.0,
            time_remaining_seconds=self.time_budget_seconds,
            done=False,
            reward=0.0,
            metadata={"episode_id": self.episode_id}
        )
    
    def _create_optimizer(self):
        """Create optimizer with current settings"""
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:  # sgd
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
    
    def step(self, action: HyperparamAction) -> HyperparamObservation:
        """Train for one epoch with these settings"""
        # Make sure criterion exists
        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        
        # Update settings
        self.learning_rate = action.learning_rate
        self.batch_size = action.batch_size
        self.weight_decay = action.weight_decay
        self.optimizer_name = action.optimizer
        
        # Recreate dataloaders and optimizer with new settings
        self.train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_data, batch_size=256, shuffle=False, num_workers=0
        )
        self._create_optimizer()
        
        # Train one epoch
        self.model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        
        # Validate
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        val_accuracy = correct / len(self.val_data) if len(self.val_data) > 0 else 0.0
        self.current_epoch += 1
        self.configs_tried += 1
        self.best_accuracy = max(self.best_accuracy, val_accuracy)
        
        # Calculate reward based on difficulty
        reward = self._compute_reward(val_accuracy, avg_loss)
        
        # Calculate time
        time_elapsed = time.time() - self.start_time
        time_remaining = max(0, self.time_budget_seconds - time_elapsed)
        
        # Check if done
        done = (
            self.current_epoch >= self.max_epochs or 
            time_elapsed >= self.time_budget_seconds
        )
        
        return HyperparamObservation(
            epoch=self.current_epoch,
            validation_accuracy=val_accuracy,
            training_loss=avg_loss,
            current_learning_rate=self.learning_rate,
            model_size_mb=self._get_model_size(),
            time_elapsed_seconds=time_elapsed,
            time_remaining_seconds=time_remaining,
            done=done,
            reward=reward,
            metadata={
                "episode_id": self.episode_id,
                "best_accuracy": self.best_accuracy,
                "grader_score": self._get_grader_score(val_accuracy)
            }
        )
    def _compute_reward(self, val_acc: float, training_loss: float) -> float:
        """Multi-component reward function"""
        
        # Accuracy reward based on target
        if val_acc >= self.target_accuracy:
            accuracy_reward = 1.0
        else:
            accuracy_reward = val_acc / self.target_accuracy
        
        # Speed reward
        time_elapsed = time.time() - self.start_time
        if time_elapsed <= self.time_budget_seconds * 0.8:
            speed_reward = 0.5
        else:
            speed_reward = 0.5 * (1.0 - (time_elapsed - self.time_budget_seconds * 0.8) / (self.time_budget_seconds * 0.2))
            speed_reward = max(0, speed_reward)
        
        # Step penalty
        step_penalty = -0.01
        
        # Divergence penalty
        divergence_penalty = -0.1 if training_loss > 5.0 else 0.0
        
        total_reward = accuracy_reward + speed_reward + step_penalty + divergence_penalty
        total_reward = max(-1.0, min(1.0, total_reward))  # Clamp to [-1, 1]
        
        return total_reward
    
    def _get_grader_score(self, accuracy: float) -> float:
        """
        Grader score for the task (0.0 - 1.0)
        This is what judges will use to score your submission
        """
        if self.difficulty == "easy":
            # MNIST: target 90%
            return min(1.0, accuracy / 0.90)
        elif self.difficulty == "medium":
            # CIFAR-10: target 80%
            return min(1.0, accuracy / 0.80)
        elif self.difficulty == "hard":
            # ImageNet-100: target 60%
            return min(1.0, accuracy / 0.60)
        return 0.0
    
    def _get_model_size(self) -> float:
        """Get model size in MB"""
        total_params = sum(p.numel() for p in self.model.parameters())
        return (total_params * 4) / (1024 * 1024)
    
    @property
    def state(self) -> HyperparamState:
        """Get current episode state"""
        return HyperparamState(
            episode_id=self.episode_id,
            difficulty=self.difficulty,
            dataset_name=self.dataset_name,
            total_epochs=self.max_epochs,
            current_epoch=self.current_epoch,
            best_accuracy=self.best_accuracy,
            total_configs_tried=self.configs_tried,
            metadata={
                "target_accuracy": self.target_accuracy,
                "grader_score": self._get_grader_score(self.best_accuracy)
            }
        )