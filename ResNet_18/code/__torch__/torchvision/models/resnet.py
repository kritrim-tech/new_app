class ResNet(Module):
  __parameters__ = []
  training : bool
  inplanes : int
  dilation : int
  groups : int
  base_width : int
  conv1 : __torch__.torch.nn.modules.conv.Conv2d
  bn1 : __torch__.torch.nn.modules.batchnorm.BatchNorm2d
  relu : __torch__.torch.nn.modules.activation.ReLU
  maxpool : __torch__.torch.nn.modules.pooling.MaxPool2d
  layer1 : __torch__.torch.nn.modules.container.Sequential
  layer2 : __torch__.torch.nn.modules.container.___torch_mangle_14.Sequential
  layer3 : __torch__.torch.nn.modules.container.___torch_mangle_22.Sequential
  layer4 : __torch__.torch.nn.modules.container.___torch_mangle_30.Sequential
  avgpool : __torch__.torch.nn.modules.pooling.AdaptiveAvgPool2d
  fc : __torch__.torch.nn.modules.linear.Linear
  def forward(self: __torch__.torchvision.models.resnet.ResNet,
    x: Tensor) -> Tensor:
    return (self)._forward_impl(x, )
  def _forward_impl(self: __torch__.torchvision.models.resnet.ResNet,
    x: Tensor) -> Tensor:
    x0 = (self.conv1).forward(x, )
    x1 = (self.bn1).forward(x0, )
    x2 = (self.relu).forward(x1, )
    x3 = (self.maxpool).forward(x2, )
    x4 = (self.layer1).forward(x3, )
    x5 = (self.layer2).forward(x4, )
    x6 = (self.layer3).forward(x5, )
    x7 = (self.layer4).forward(x6, )
    x8 = (self.avgpool).forward(x7, )
    x9 = torch.flatten(x8, 1, -1)
    return (self.fc).forward(x9, )
class BasicBlock(Module):
  __parameters__ = []
  training : bool
  downsample : None
  stride : int
  conv1 : __torch__.torch.nn.modules.conv.___torch_mangle_6.Conv2d
  bn1 : __torch__.torch.nn.modules.batchnorm.BatchNorm2d
  relu : __torch__.torch.nn.modules.activation.ReLU
  conv2 : __torch__.torch.nn.modules.conv.___torch_mangle_6.Conv2d
  bn2 : __torch__.torch.nn.modules.batchnorm.BatchNorm2d
  def forward(self: __torch__.torchvision.models.resnet.BasicBlock,
    x: Tensor) -> Tensor:
    out = (self.conv1).forward(x, )
    out0 = (self.bn1).forward(out, )
    out1 = (self.relu).forward(out0, )
    out2 = (self.conv2).forward(out1, )
    out3 = (self.bn2).forward(out2, )
    out4 = torch.add_(out3, x, alpha=1)
    return (self.relu).forward(out4, )
