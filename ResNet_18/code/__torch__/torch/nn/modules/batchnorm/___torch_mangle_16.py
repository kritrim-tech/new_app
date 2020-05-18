class BatchNorm2d(Module):
  __parameters__ = ["weight", "bias", ]
  weight : Tensor
  bias : Tensor
  running_mean : Tensor
  running_var : Tensor
  num_batches_tracked : Tensor
  training : bool
  affine : Final[bool] = True
  eps : Final[float] = 1.0000000000000001e-05
  track_running_stats : Final[bool] = True
  num_features : Final[int] = 256
  momentum : Final[float] = 0.10000000000000001
  def forward(self: __torch__.torch.nn.modules.batchnorm.___torch_mangle_16.BatchNorm2d,
    input: Tensor) -> Tensor:
    _0 = __torch__.torch.nn.functional.batch_norm
    _1 = (self)._check_input_dim(input, )
    if self.training:
      _2 = True
    else:
      _2 = False
    if _2:
      _3 = torch.add(self.num_batches_tracked, 1, 1)
      self.num_batches_tracked = _3
    else:
      pass
    _4 = self.running_mean
    _5 = self.running_var
    _6 = self.weight
    _7 = self.bias
    if self.training:
      _8 = True
    else:
      _8 = False
    _9 = _0(input, _4, _5, _6, _7, _8, 0.10000000000000001, 1.0000000000000001e-05, )
    return _9
  def _check_input_dim(self: __torch__.torch.nn.modules.batchnorm.___torch_mangle_16.BatchNorm2d,
    input: Tensor) -> None:
    if torch.ne(torch.dim(input), 4):
      ops.prim.RaiseException("Exception")
    else:
      pass
    return None
