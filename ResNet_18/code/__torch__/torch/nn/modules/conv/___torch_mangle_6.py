class Conv2d(Module):
  __parameters__ = ["weight", "bias", ]
  weight : Tensor
  bias : Optional[Tensor]
  training : bool
  transposed : bool
  _padding_repeated_twice : Tuple[int, int, int, int]
  kernel_size : Final[Tuple[int, int]] = (3, 3)
  padding_mode : Final[str] = "zeros"
  padding : Final[Tuple[int, int]] = (1, 1)
  stride : Final[Tuple[int, int]] = (1, 1)
  out_channels : Final[int] = 64
  in_channels : Final[int] = 64
  dilation : Final[Tuple[int, int]] = (1, 1)
  groups : Final[int] = 1
  output_padding : Final[Tuple[int, int]] = (0, 0)
  def forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_6.Conv2d,
    input: Tensor) -> Tensor:
    _0 = (self)._conv_forward(input, self.weight, )
    return _0
  def _conv_forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_6.Conv2d,
    input: Tensor,
    weight: Tensor) -> Tensor:
    _1 = torch.conv2d(input, weight, self.bias, [1, 1], [1, 1], [1, 1], 1)
    return _1
