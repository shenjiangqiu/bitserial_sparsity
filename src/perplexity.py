# %%

import torch
from torcheval.metrics.text import Perplexity


def get_perplexity(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    metric = Perplexity(ignore_index=0)

    metric.update(input, target)
    out = metric.compute()
    return out


# %%
if __name__ == "__main__":
    input = torch.tensor(
        [[[100.0, 0, 0]], [[100, 0, 0]], [[0.5, 0.9, 0]]], dtype=torch.float64
    )

    # 目标 token 索引
    target = torch.tensor([[0], [0], [1]])

    out = get_perplexity(input, target)
    print(out)

    pass

# %%
