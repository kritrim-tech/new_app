def batch_norm(input: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    weight: Optional[Tensor]=None,
    bias: Optional[Tensor]=None,
    training: bool=False,
    momentum: float=0.10000000000000001,
    eps: float=1.0000000000000001e-05) -> Tensor:
  _0 = __torch__.torch.nn.functional._verify_batch_size
  if training:
    _1 = _0(torch.size(input), )
  else:
    pass
  _2 = torch.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, True)
  return _2
def relu(input: Tensor,
    inplace: bool=False) -> Tensor:
  if inplace:
    result = torch.relu_(input)
  else:
    result = torch.relu(input)
  return result
def _max_pool2d(input: Tensor,
    kernel_size: List[int],
    stride: Optional[List[int]]=None,
    padding: List[int]=[0, 0],
    dilation: List[int]=[1, 1],
    ceil_mode: bool=False,
    return_indices: bool=False) -> Tensor:
  if torch.__is__(stride, None):
    stride0 = annotate(List[int], [])
  else:
    stride0 = unchecked_cast(List[int], stride)
  _3 = torch.max_pool2d(input, kernel_size, stride0, padding, dilation, ceil_mode)
  return _3
def adaptive_avg_pool2d(input: Tensor,
    output_size: List[int]) -> Tensor:
  _4 = __torch__.torch.nn.modules.utils._list_with_default
  _output_size = _4(output_size, torch.size(input), )
  _5 = torch.adaptive_avg_pool2d(input, _output_size)
  return _5
def linear(input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor]=None) -> Tensor:
  if torch.eq(torch.dim(input), 2):
    _6 = torch.__isnot__(bias, None)
  else:
    _6 = False
  if _6:
    bias0 = unchecked_cast(Tensor, bias)
    ret = torch.addmm(bias0, input, torch.t(weight), beta=1, alpha=1)
  else:
    output = torch.matmul(input, torch.t(weight))
    if torch.__isnot__(bias, None):
      bias1 = unchecked_cast(Tensor, bias)
      output0 = torch.add_(output, bias1, alpha=1)
    else:
      output0 = output
    ret = output0
  return ret
def _verify_batch_size(size: List[int]) -> None:
  size_prods = size[0]
  size_prods0 = size_prods
  for i in range(torch.sub(torch.len(size), 2)):
    size_prods0 = torch.mul(size_prods0, size[torch.add(i, 2)])
  if torch.eq(size_prods0, 1):
    ops.prim.RaiseException("Exception")
  else:
    pass
  return None
