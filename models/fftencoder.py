import ptwt
import pywt
import torch
from einops import rearrange
from models import BrainEncoder
from torch import Tensor, cat, div, nn, no_grad
from torch.fft import fftn, ifftn, irfftn, rfftn

if torch.cuda.get_device_capability()[0] >= 8:
    from flash_attn.ops.triton.layer_norm import RMSNorm
else:
    from torch.nn import RMSNorm

from models import BrainEncoder


def normalize(vec: Tensor, dim=-1) -> Tensor:
    result: Tensor

    fft_vec = rfftn(vec, s=vec.shape[dim], dim=dim)
    norm_fft = fft_vec.abs().clamp(min=1e-8)
    normed_vec = div(fft_vec, norm_fft)
    result = irfftn(normed_vec, s=vec.shape[dim], dim=dim)

    return result


def fft_convolution(
    signal: Tensor, kernel_1: Tensor, kernel_2: Tensor, dim: int = 0, invert=False
) -> Tensor:
    signal_fft = fftn(signal, dim=dim)
    kernel_1_fft = fftn(kernel_1, dim=dim)
    kernel_2_fft = fftn(kernel_2, dim=dim)

    if invert:
        kernel_1_fft = kernel_1_fft.conj()
        kernel_2_fft = kernel_2_fft.conj()

    conv_fft = signal_fft * kernel_1_fft
    conv_fft = conv_fft.transpose(-2, -3)
    conv_fft = conv_fft * kernel_2_fft
    conv_fft = conv_fft.transpose(-2, -3)

    conv: Tensor = ifftn(conv_fft, dim=dim)

    return conv.real


class WaveEncoder(BrainEncoder):
    def __init__(
        self,
        size_input: int,
        size_output: int,
        wavelet: str = "db4",
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.size_input = size_input
        self.size_output = size_output
        self.gradient_checkpointing = gradient_checkpointing

        self.wavelet = pywt.Wavelet(wavelet)

        self.sizes = []
        self._compute_dwt_size()

        self.ln = RMSNorm(self.dwt_size, eps=1e-5)
        self.proj = nn.Linear(self.dwt_size, size_output, bias=False)

    def _compute_dwt_size(self):
        max_levels = pywt.dwt_max_level(self.size_input, self.wavelet)
        input_size = self.size_input
        for _ in range(max_levels):
            input_size = pywt.dwt_coeff_len(input_size, self.wavelet, "periodic")
            self.sizes.append(input_size)

        self.sizes.append(input_size)
        self.dwt_size = sum(self.sizes)

    @no_grad()
    def block(self, x: Tensor) -> Tensor:
        outputs = cat(
            ptwt.wavedec(x.unsqueeze(1).float(), self.wavelet, mode="periodic"),
            dim=-1,
        )
        outputs = outputs.squeeze(1)

        return outputs

    def forward(self, x: Tensor):
        outputs = self.block(x)

        outputs = self.ln(outputs)

        outputs = self.proj(outputs.type(x.dtype))

        return outputs

    def encode(self, x: Tensor):
        outputs = self.block(x)

        outputs = self.ln(outputs)

        outputs = self.proj(outputs.type(x.dtype))

        return outputs

    def decode(self, x: Tensor):
        bsz, frames, ch, _ = x.shape
        inputs = rearrange(x, "bsz frames ch len -> (bsz frames ch) 1 len")
        inputs *= self.proj_const
        outputs = self.proj_inv(inputs)

        outputs = self.block_inv(outputs)
        outputs = rearrange(
            outputs,
            "(bsz frames ch) 1 len -> bsz frames ch len",
            bsz=bsz,
            frames=frames,
            ch=ch,
        )

        return outputs
