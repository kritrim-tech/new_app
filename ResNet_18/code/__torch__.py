class MyResnet_18(Module):
  __parameters__ = []
  training : bool
  resnet : __torch__.torchvision.models.resnet.ResNet
  def forward(self: __torch__.MyResnet_18,
    x: Tensor) -> Tensor:
    return (self.resnet).forward(x, )
