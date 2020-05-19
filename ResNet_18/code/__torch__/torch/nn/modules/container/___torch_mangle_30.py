class Sequential(Module):
  __parameters__ = []
  training : bool
  __annotations__["0"] = __torch__.torchvision.models.resnet.___torch_mangle_28.BasicBlock
  __annotations__["1"] = __torch__.torchvision.models.resnet.___torch_mangle_29.BasicBlock
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_30.Sequential,
    input: Tensor) -> Tensor:
    _0 = getattr(self, "0")
    _1 = getattr(self, "1")
    input0 = (_0).forward(input, )
    return (_1).forward(input0, )
