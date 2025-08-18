"""Single example dataset for debugging JAX vs PyTorch training."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any

from openpi.models import model as _model
from openpi.shared import image_tools
from openpi import transforms as _transforms
from openpi.training.config import DataConfig


class SingleExampleDataset(Dataset):
    """Dataset that always returns the same example for debugging."""
    
    def __init__(self, action_dim: int = 8, action_horizon: int = 10, image_size: int = 224):
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.image_size = image_size
        
        # Create a fixed example
        self._create_fixed_example()
    
    def _create_fixed_example(self):
        """Create a fixed example with deterministic values."""
        np.random.seed(42)  # Fixed seed for reproducibility
        
        batch_size = 1
        
        # Create fixed images (simple patterns)
        images = {}
        for key in _model.IMAGE_KEYS:
            # Create a simple gradient pattern
            img = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
            
            # Create a gradient from top-left to bottom-right
            for i in range(self.image_size):
                for j in range(self.image_size):
                    # Normalize to [-1, 1] range
                    val = (i + j) / (2 * self.image_size) * 2 - 1
                    img[0, i, j, :] = [val, val * 0.5, val * 0.25]
            
            images[key] = img
        
        # Create fixed state
        state = np.random.randn(batch_size, self.action_dim).astype(np.float32)
        # Normalize state to reasonable range
        state = state * 0.1
        
        # Create fixed actions
        actions = np.random.randn(batch_size, self.action_horizon, self.action_dim).astype(np.float32)
        # Normalize actions to reasonable range
        actions = actions * 0.1
        
        # Create fixed language tokens
        max_token_len = 48
        tokenized_prompt = np.random.randint(0, 1000, (batch_size, max_token_len), dtype=np.int32)
        tokenized_prompt_mask = np.ones((batch_size, max_token_len), dtype=bool)
        
        # Create image masks
        image_masks = {key: np.ones(batch_size, dtype=bool) for key in _model.IMAGE_KEYS}
        
        self.fixed_example = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
            "actions": actions,
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask,
        }
    
    def __len__(self):
        return 1000  # Return many copies of the same example
    
    def __getitem__(self, idx):
        # Always return the same example
        return self.fixed_example.copy()


class SingleExampleDataConfig(DataConfig):
    """Data config that uses the single example dataset."""
    
    def __init__(self, action_dim: int = 8, action_horizon: int = 10):
        super().__init__(
            repo_id="single_example",
            asset_id="single_example",
            norm_stats=None,  # No normalization for debugging
            repack_transforms=_transforms.Group(),
            data_transforms=_transforms.Group(),
            model_transforms=_transforms.Group(),
            use_quantile_norm=False,
            action_sequence_keys=("actions",),
            prompt_from_task=False,
        )
        self.action_dim = action_dim
        self.action_horizon = action_horizon


class SingleExampleDataConfigFactory:
    """Factory for creating single example data configs."""
    
    def __init__(self, action_dim: int = 8, action_horizon: int = 10):
        self.action_dim = action_dim
        self.action_horizon = action_horizon
    
    def create(self, assets_dirs, model_config):
        return SingleExampleDataConfig(self.action_dim, self.action_horizon)


def create_single_example_dataset(data_config, action_horizon, model_config):
    """Create the single example dataset."""
    return SingleExampleDataset(
        action_dim=data_config.action_dim,
        action_horizon=action_horizon,
        image_size=224
    )


def transform_single_example_dataset(dataset, data_config):
    """Transform the single example dataset (no transforms for debugging)."""
    return dataset 