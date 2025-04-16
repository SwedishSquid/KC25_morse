from torch import nn


class ResBlock(nn.Module):
    def __init__(self, size, p_dropout):
        super().__init__()
        self.cell = nn.Sequential(
            nn.Conv1d(size, size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(size),
            nn.Dropout(p=p_dropout),
            nn.Conv1d(size, size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(size),
            nn.Dropout(p=p_dropout),
        )
        self.activation = nn.ReLU()
        pass

    def forward(self, x):
        return self.activation(x + self.cell(x))
    pass


class MySomething(nn.Module):
    def __init__(self, n_pooled_blocks = 3, n_head_blocks = 2, pooled_blocks_thickness=1, 
                 input_size = 64, inner_size = 64, output_size = 5, p_dropout = 0.1):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Conv1d(input_size, inner_size, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(inner_size),
            nn.Dropout(),
            *[
                 nn.Sequential(
                    *[ResBlock(inner_size, p_dropout) for i_ in range(pooled_blocks_thickness)],
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    ) for _ in range(n_pooled_blocks)
            ],
            *[ResBlock(inner_size, p_dropout) for _ in range(n_head_blocks)],
            nn.Conv1d(inner_size, output_size, kernel_size=3),
        )
        pass

    def forward(self, x):
        return self.estimator(x)
    