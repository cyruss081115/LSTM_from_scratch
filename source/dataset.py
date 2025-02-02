from datetime import datetime
from typing import Callable

import pandas as pd
import yfinance as yf

import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """Stock price dataset"""
    def __init__(
        self,
        stock_abbv: str,
        start: datetime,
        end: datetime,
        augmentation: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
        sequence_length: int = 30,  # days
        sequence_step: int = 5,  # days
        train_test_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.stock_abbv = stock_abbv
        self.start = start
        self.end = end
        self.sequence_length = sequence_length
        self.train_test_ratio = train_test_ratio
        self.seed = seed

        torch.manual_seed(self.seed)

        stock = yf.download(stock_abbv, start=start, end=end)
        stock = augmentation(stock)
        stock = torch.from_numpy(stock.to_numpy())
        
        X = []
        y = []
        for i in range(0, len(stock), sequence_length // sequence_step):
            if i + sequence_length >= len(stock):
                break
            current_x = stock[i:i+sequence_length, :]
            current_y = stock[i+sequence_length, 0] # select the next time frame, assume first one is closing price

            x_max, _ = torch.max(current_x, dim=0)
            x_min, _ = torch.min(current_x, dim=0)
            def min_max_scale(x: torch.Tensor, x_max: torch.Tensor, x_min: torch.Tensor):
                return (x - x_min) / (x_max - x_min)
            
            current_x = min_max_scale(current_x, x_max, x_min)
            current_y = min_max_scale(current_y, x_max[0], x_min[0])

            X.append(current_x)
            y.append(current_y)

        training_size = int(len(y) * train_test_ratio)
        indices = torch.randperm(len(y))

        self.X = X
        self.y = y
        self.train = True  # sets the init state of dataset to train
        self.training_indices = indices[:training_size]
        self.testing_indices = indices[training_size:]
    
    def set_train(self):
        self.train = True
    
    def set_eval(self):
        self.train = False
    
    def __len__(self):
        if self.train:
            return len(self.training_indices)
        else:
            return len(self.testing_indices)
    
    def __getitem__(self, index: int):
        assert index < self.__len__()
        if self.train:
            true_index = self.training_indices[index]
            return self.X[true_index], self.y[true_index]
        else:
            true_index = self.testing_indices[index]
            return self.X[true_index], self.y[true_index]