import torch
import torch.nn as nn
from utils_inference.bench import cuda_bench
import time


class FFNModel(nn.Module):
    def __init__(self, num_blocks, dim, ffn_dim):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(dim, ffn_dim),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(ffn_dim, dim),
                )
            )

    @cuda_bench.test_time_cuda(enable=True, contain_cpu=True, contain_cuda=True)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __ckp_map__(self) -> dict:
        """
        map from real state_dict key to parameter key

        :return: map dict
        :rtype: dict
        """
        map = {}
        for name, _ in self.named_parameters():
            map[name.replace("layers.", "")] = name
        return map

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[self.__ckp_map__().get(k, k)] = v

        return super().load_state_dict(new_state_dict, strict=strict)


def generate():
    torch.set_grad_enabled(False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load checkpoint
    checkpoint = "weight/wan_high_noise_ffn_modulelist.pt"
    state_dict = torch.load(checkpoint, weights_only=True)

    # init baseline model
    baseline = FFNModel(num_blocks=40, dim=5120, ffn_dim=13824)
    baseline.load_state_dict(state_dict)
    baseline.eval().to(device)

    x = torch.randn(1, 128, 5120, device=device)
    output = baseline(x)

    print("output: ", output.shape)


if __name__ == "__main__":
    generate()


