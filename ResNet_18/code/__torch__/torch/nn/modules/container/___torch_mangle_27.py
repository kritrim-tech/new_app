class Sequential(Module):
  __parameters__ = []
  training : bool
  __annotations__["0"] = __torch__.torch.nn.modules.conv.___torch_mangle_26.Conv2d
  __annotations__["1"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_24.BatchNorm2d
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_27.Sequential,
    input: Tensor) -> Tensor:
    _0 = getattr(self, "0")
    _1 = getattr(self, "1")
    input0 = (_0).forward(input, )
    return (_1).forward(input0, )
