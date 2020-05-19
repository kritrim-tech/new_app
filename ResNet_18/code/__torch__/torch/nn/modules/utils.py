def _list_with_default(out_size: List[int],
    defaults: List[int]) -> List[int]:
  _0 = torch.le(torch.len(defaults), torch.len(out_size))
  if _0:
    ops.prim.RaiseException("Exception")
  else:
    pass
  _1 = annotate(List[int], [])
  _2 = torch.slice(defaults, torch.neg(torch.len(out_size)), 9223372036854775807, 1)
  _3 = ops.prim.min([torch.len(out_size), torch.len(_2)])
  for _4 in range(_3):
    v = out_size[_4]
    _5 = torch.append(_1, v)
  return _1
