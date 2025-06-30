import functools
from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from jax_cfd.base import equations
from jax_cfd.base import filter_utils
from jax_cfd.base import grids
from jax_cfd.base import validation_problems

Array = grids.Array
GridArrayVector = grids.GridArrayVector
GridVariableVector = grids.GridVariableVector
ForcingFn = Callable[[GridVariableVector], GridArrayVector]

from flax import linen as nn
from jax import random, numpy as jnp

def exponential_force(
    grid: grids.Grid,
    center_x: float,
    center_y: float,
    offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
) -> ForcingFn:
  
  if offsets is None:
    offsets = ((0.5,0.5),(0.5,0.5))#grid.cell_faces

  x = grid.mesh(offsets[1])[0]
  y = grid.mesh(offsets[0])[1]

  exponent_internal = ((x-center_x)**2.0+(y-center_y)**2.0)
  factor = 2.0 / (2.0 * jnp.pi * 0.05 **2)
  array = factor * jnp.exp(-exponent_internal / (2.0 * 0.05 **2))

  G1 = grids.GridArray(array, offsets[1], grid)
  G2 = grids.GridArray(array, offsets[0], grid)#这里的offset设计是没有道理的

  def forcing():
    return tuple((G1,G2))
  
  return forcing

def exponential_force_adjustable(
    grid: grids.Grid,
    center_x: float,
    center_y: float,
    theta_s: float,
    offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
) -> ForcingFn:
  
  if offsets is None:
    offsets = ((0.5,0.5),(0.5,0.5))#grid.cell_faces

  x = grid.mesh(offsets[1])[0]
  y = grid.mesh(offsets[0])[1]

  exponent_internal = ((x-center_x)**2.0+(y-center_y)**2.0) / (2.0 * 0.05 **2)
  factor = theta_s / (2.0 * jnp.pi * 0.05 **2)
  array = factor * jnp.exp(-exponent_internal)

  G1 = grids.GridArray(array, offsets[1], grid)
  G2 = grids.GridArray(array, offsets[0], grid)#这里的offset设计是没有道理的

  def forcing():
    return tuple((G1,G2))
  
  return forcing



class FullyConnectedNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=8)(x)  # 输入层到隐藏层1，8个节点
        x = nn.tanh(x)  # 激活函数
        x = nn.Dense(features=4)(x)  # 隐藏层1到隐藏层2，4个节点
        x = nn.tanh(x)  # 激活函数
        x = nn.Dense(features=1)(x)  # 隐藏层2到输出层，1个节点
        x = nn.tanh(x)  # 输出层激活函数
        return x
    


def reciprocal_force_with_nn(
    grid: grids.Grid,
    center_x: float,
    center_y: float,
    params: dict,
    model: FullyConnectedNN,
    offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
) -> ForcingFn:
  
  if offsets is None:
    offsets = ((0.5,0.5),(0.5,0.5))#grid.cell_faces

  x = grid.mesh(offsets[1])[0]
  y = grid.mesh(offsets[0])[1]

  exponent_internal = ((x-center_x)**2.0+(y-center_y)**2.0)
  factor = 2.0 / (2.0 * jnp.pi * 0.05 **2.0)*1.5
  array_reciprocal = factor / (exponent_internal**2.0/(2.0 * 0.05 **2.0)**2.0+ 1.0)

  x_reshaped = x.reshape(-1, 1) - center_x
  y_reshaped = y.reshape(-1, 1) - center_y
  xy_combined = jnp.concatenate((x_reshaped, y_reshaped), axis=1)
  
  array_nn = model.apply(params,xy_combined).reshape(grid.shape) #* 100.0

  G1 = grids.GridArray(array_reciprocal+array_nn, offsets[1], grid)
  G2 = grids.GridArray(array_reciprocal+array_nn, offsets[0], grid)

  def forcing():
    return tuple((G1,G2))
  
  return forcing