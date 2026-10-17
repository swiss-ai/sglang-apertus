import torch
import triton
import triton.language as tl


@triton.jit
def _sssglu_and_mul_kernel(
    input_ptr,
    output_ptr,
    width: tl.constexpr,
    input_stride: tl.constexpr,
    output_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < width
    gate = tl.load(input_ptr + row * input_stride + offsets, mask=mask).to(tl.float32)
    up = tl.load(input_ptr + row * input_stride + width + offsets, mask=mask).to(
        tl.float32
    )
    shifted = gate - 1.0
    activated = tl.div_rn(shifted, 1.0 + tl.abs(shifted)) + 0.5
    tl.store(output_ptr + row * output_stride + offsets, activated * up, mask=mask)


def sssglu_and_mul(
    input: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute ``(softsign(gate - 1) + 0.5) * up`` in FP32, then cast.

    The input contains concatenated gate and up projections in its last axis.
    Two-dimensional inputs may have a strided row layout.
    """
    if input.ndim == 0 or input.shape[-1] % 2:
        raise ValueError("SSSGLU requires an even last dimension")
    if not input.is_cuda or input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("SSSGLU requires a CUDA/HIP floating-point tensor")
    width = input.shape[-1] // 2
    output_shape = input.shape[:-1] + (width,)
    if out is None:
        out = input.new_empty(output_shape)
    elif (
        out.shape != output_shape
        or out.dtype != input.dtype
        or out.device != input.device
        or not (
            out.is_contiguous()
            or (out.ndim == 2 and out.stride(-1) == 1 and out.stride(0) >= width)
        )
    ):
        raise ValueError(
            "SSSGLU output requires the expected shape, dtype, device, "
            "and non-overlapping rows"
        )
    if out.numel() == 0:
        return out
    if input.ndim == 2 and input.stride(-1) == 1:
        input_2d = input
    else:
        input_2d = input.contiguous().view(-1, width * 2)
    output_2d = out if out.ndim == 2 else out.view(-1, width)
    _sssglu_and_mul_kernel[(input_2d.shape[0], triton.cdiv(width, 256))](
        input_2d,
        output_2d,
        width,
        input_2d.stride(0),
        output_2d.stride(0),
        BLOCK_SIZE=256,
        enable_fp_fusion=False,
    )
    return out
