class Linear(Module):
  __parameters__ = ["weight", "bias", ]
  weight : Tensor
  bias : Tensor
  training : bool
  out_features : Final[int] = 2
  in_features : Final[int] = 512
  def forward(self: __torch__.torch.nn.modules.linear.Linear,
    input: Tensor) -> Tensor:
    _0 = __torch__.torch.nn.functional.linear
    return _0(input, self.weight, self.bias, )
