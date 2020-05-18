class BasicBlock(Module):
  __parameters__ = []
  training : bool
  downsample : None
  stride : int
  conv1 : __torch__.torch.nn.modules.conv.___torch_mangle_17.Conv2d
  bn1 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_16.BatchNorm2d
  relu : __torch__.torch.nn.modules.activation.ReLU
  conv2 : __torch__.torch.nn.modules.conv.___torch_mangle_17.Conv2d
  bn2 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_16.BatchNorm2d
  def forward(self: __torch__.torchvision.models.resnet.___torch_mangle_21.BasicBlock,
    x: Tensor) -> Tensor:
    out = (self.conv1).forward(x, )
    out0 = (self.bn1).forward(out, )
    out1 = (self.relu).forward(out0, )
    out2 = (self.conv2).forward(out1, )
    out3 = (self.bn2).forward(out2, )
    out4 = torch.add_(out3, x, alpha=1)
    return (self.relu).forward(out4, )
