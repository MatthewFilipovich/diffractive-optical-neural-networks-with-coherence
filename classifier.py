import torch
from torch.nn import Module


class Classifier(Module):
    def __init__(self, shape, region_size):
        super().__init__()
        if shape < 4 * region_size:
            raise ValueError("shape must be at least 4*region_size")

        weight = torch.zeros(10, shape, shape, dtype=torch.double)
        row_offset = (shape - 4 * region_size) // 2
        col_offset = (shape - 3 * region_size) // 2

        # Function to set a region to 1
        def set_region(digit, row, col):
            start_row = row * (region_size) + row_offset
            start_col = col * (region_size) + col_offset
            weight[
                digit,
                start_row : start_row + region_size,
                start_col : start_col + region_size,
            ] = 1

        # Add the bottom row representing "zero" (special case)
        set_region(0, 3, 1)

        # Add the top three rows from left to right
        for digit in range(1, 10):
            row, col = (digit - 1) // 3, (digit - 1) % 3
            set_region(digit, row, col)

        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x):
        return torch.einsum("nxy,bxy->bn", self.weight, x)
