class BasicBlock(Module):
  __parameters__ = []
  training : bool
  stride : int
  conv1 : __torch__.torch.nn.modules.conv.___torch_mangle_23.Conv2d
  bn1 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_24.BatchNorm2d
  relu : __torch__.torch.nn.modules.activation.ReLU
  conv2 : __torch__.torch.nn.modules.conv.___torch_mangle_25.Conv2d
  bn2 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_24.BatchNorm2d
  downsample : __torch__.torch.nn.modules.container.___torch_mangle_27.Sequential
  def forward(self: __torch__.torchvision.models.resnet.___torch_mangle_28.BasicBlock,
    x: Tensor) -> Tensor:
    out = (self.conv1).forward(x, )
    out0 = (self.bn1).forward(out, )
    out1 = (self.relu).forward(out0, )
    out2 = (self.conv2).forward(out1, )
    out3 = (self.bn2).forward(out2, )
    identity = (self.downsample).forward(x, )
    out4 = torch.add_(out3, identity, alpha=1)
    return (self.relu).forward(out4, )
