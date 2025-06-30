# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Examples of defining equations."""
import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from jax_cfd.base import advection
from jax_cfd.base import diffusion
from jax_cfd.base import grids
from jax_cfd.base import pressure
from jax_cfd.base import time_stepping
from jax_cfd.base import initial_conditions
import tree_math

from jax_cfd.base import boundaries

# Specifying the full signatures of Callable would get somewhat onerous
# pylint: disable=g-bare-generic

GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
ConvectFn = Callable[[GridVariableVector], GridArrayVector]
DiffuseFn = Callable[[GridVariable, float], GridArray]
ForcingFn = Callable[[GridVariableVector], GridArrayVector]


def sum_fields(*args):
  return jax.tree_map(lambda *a: sum(a), *args)


def stable_time_step(
    max_velocity: float,
    max_courant_number: float,
    viscosity: float,
    grid: grids.Grid,
    implicit_diffusion: bool = False,
) -> float:
  """Calculate a stable time step for Navier-Stokes."""
  dt = advection.stable_time_step(max_velocity, max_courant_number, grid)
  if not implicit_diffusion:
    diffusion_dt = diffusion.stable_time_step(viscosity, grid)
    if diffusion_dt < dt:
      raise ValueError(f'stable time step for diffusion is smaller than '
                       f'the chosen timestep: {diffusion_dt} vs {dt}')
  return dt


def dynamic_time_step(v: GridVariableVector,
                      max_courant_number: float,
                      viscosity: float,
                      grid: grids.Grid,
                      implicit_diffusion: bool = False) -> float:
  """Pick a dynamic time-step for Navier-Stokes based on stable advection."""
  v_max = jnp.sqrt(jnp.max(sum(u.data ** 2 for u in v)))
  return stable_time_step(  # pytype: disable=wrong-arg-types  # jax-types
      v_max, max_courant_number, viscosity, grid, implicit_diffusion)


def _wrap_term_as_vector(fun, *, name):
  return tree_math.unwrap(jax.named_call(fun, name=name), vector_argnums=0)


size = 251
# only change adcection func
def implicit_convection_diffusion3(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffusion_solve: Callable = diffusion.solve_fast_diag,
    forcing: Optional[ForcingFn] = None,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""
  """this method is only for Dirichlet boundary condition"""
  del grid  # unused

  if convect is None:
    def convect(v,G):  # pylint: disable=function-redefined
      return tuple(
          advection.advect_van_leer(u, v, dt) for u in G)

  convect = jax.named_call(convect, name='convection')
  diffusion_solve = jax.named_call(diffusion_solve, name='diffusion')

  # TODO(shoyer): refactor to support optional higher-order time integators
  @jax.named_call
  def navier_stokes_step(G: GridVariableVector, t:float):
    """Computes state at time `t + dt` using first order time integration."""

    x_velocity_fn = lambda x, y: jnp.ones_like(x)*20.0*t
    y_velocity_fn = lambda x, y: jnp.ones_like(x)*20.0*t
    #grid1 = grids.Grid((256, 256), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    grid1 = grids.Grid((size , size), domain=((-2., 3.), (-2., 3.)))
    #grid1 = grids.Grid((size , size), domain=((-1., 2.), (-1., 2.)))
    v = initial_conditions.initial_velocity_field((x_velocity_fn, y_velocity_fn), grid1)

    #x_v = jnp.ones((256, 256))*50.0*t
    ##grid1 = grids.Grid((256, 256), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    #grid1 = grids.Grid((256, 256), domain=((0, 1), (0, 1)))
    #bc2 = boundaries.neumann_boundary_conditions(2)
    #array = grids.GridArray(x_v, offset=grid1.cell_center, grid=grid1)
    #v0 = grids.GridVariable(array, bc2)
    #v=(v0,v0)

    convection = convect(v,G)
    accelerations = [convection]

    if forcing is not None:
      # TODO(shoyer): include time in state?
      f = forcing()
      #accelerations.append(tuple(f for i in range(2)))
      accelerations.append(tuple(f / density for f in f))

    dGdt = sum_fields(*accelerations)
    # Update v by taking a time step
    G = tuple(
        grids.GridVariable(g.array + dgdt * dt, g.bc)
        for g, dgdt in zip(G, dGdt))
    # Solve for implicit diffusion
    G = diffusion_solve(G, viscosity, dt)
    current_time = t + dt
    return G,current_time
  return navier_stokes_step

def implicit_convection_diffusion5(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffusion_solve: Callable = diffusion.solve_fast_diag,
    forcing: Optional[ForcingFn] = None,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""
  """this method is only for Dirichlet boundary condition"""
  del grid  # unused

  if convect is None:
    def convect(v,G):  # pylint: disable=function-redefined
      return tuple(
          advection.advect_van_leer(u, v, dt) for u in G)

  convect = jax.named_call(convect, name='convection')
  diffusion_solve = jax.named_call(diffusion_solve, name='diffusion')

  # TODO(shoyer): refactor to support optional higher-order time integators
  @jax.named_call
  def navier_stokes_step(G: GridVariableVector, t:float):
    """Computes state at time `t + dt` using first order time integration."""

    x_velocity_fn = lambda x, y: jnp.ones_like(x)*50.0*t
    y_velocity_fn = lambda x, y: jnp.ones_like(x)*50.0*t
    grid1 = grids.Grid((size , size), domain=((-2., 3.), (-2., 3.)))
    v = initial_conditions.initial_velocity_field((x_velocity_fn, y_velocity_fn), grid1)

    convection = convect(v,G)
    accelerations = [convection]

    if forcing is not None:
      # TODO(shoyer): include time in state?
      f = forcing()
      #accelerations.append(tuple(f for i in range(2)))
      accelerations.append(tuple(f / density for f in f))

    dGdt = sum_fields(*accelerations)
    # Update v by taking a time step
    G = tuple(
        grids.GridVariable(g.array + dgdt * dt, g.bc)
        for g, dgdt in zip(G, dGdt))
    # Solve for implicit diffusion
    G = diffusion_solve(G, viscosity, dt)
    current_time = t + dt
    return G,current_time
  return navier_stokes_step


from typing import Tuple

def implicit_convection_diffusion_rhs(
    density: float,
    viscosity: float,
    dt: float,
    custom_convection_velocity1: float,
    custom_convection_velocity2: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,  # 类型上等同于 ConvectFn
    forcing: Optional[ForcingFn] = None,  # 类型上等同于 ForcingFn
) -> Callable[[GridVariableVector], Tuple[GridVariableVector, GridVariableVector]]:
    """
    Returns a function that computes the right-hand side (dG/dt) of the Navier-Stokes equations.
    
    This function computes the explicit contributions from convection and forcing at a given time t,
    without performing the state update or the implicit diffusion solve.
    
    Parameters:
      density: Fluid density.
      viscosity: Viscosity (not used in this explicit RHS computation).
      dt: Time step size (used for the convection scheme).
      custom_convection_velocity1: Custom coefficient for the x-direction velocity.
      custom_convection_velocity2: Custom coefficient for the y-direction velocity.
      grid: Grid information (unused in this implementation).
      convect: Convection function; if None, the default van Leer advection is used.
      forcing: Forcing function; if provided, its contribution is included.
      
    Returns:
      A function that takes the current state (GridVariableVector) and time (float) and returns
      the computed right-hand side dG/dt (with the same structure as GridVariableVector).
    """
    del grid  # 当前未使用传入的 grid

    if convect is None:
        def convect(v, G):
            return tuple(
                advection.advect_van_leer(u, v, dt) for u in G
            )

    convect = jax.named_call(convect, name='convection')

    @jax.named_call
    def compute_rhs(G: GridVariableVector, t: float):
        """
        Computes dG/dt (the right-hand side) at time t for the given state G,
        based on convection and (if provided) forcing terms.
        """
        # 构造局部速度场
        x_velocity_fn = lambda x, y: jnp.ones_like(x) * custom_convection_velocity1 * t
        y_velocity_fn = lambda x, y: jnp.ones_like(x) * custom_convection_velocity2 * t
        # 注意：这里使用的 grid1 的尺寸和域需要与你的实际设置一致
        grid1 = grids.Grid((size, size), domain=((-2., 3.), (-2., 3.)))
        v = initial_conditions.initial_velocity_field((x_velocity_fn, y_velocity_fn), grid1)

        # 计算对流项
        convection = convect(v, G)
        accelerations = [convection]

        if forcing is not None:
            f = forcing()
            accelerations.append(tuple(f / density for f in f))

        # 求和得到右侧项 dG/dt（仅包含对流和源项）
        dGdt = sum_fields(*accelerations)
        return dGdt

    return compute_rhs

def implicit_convection_diffusion_rhs2(
    density: float,
    viscosity: float,
    dt: float,
    custom_convection_velocity1: float,
    custom_convection_velocity2: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,  # 类型上等同于 ConvectFn
    forcing: Optional[ForcingFn] = None,  # 类型上等同于 ForcingFn
) -> Callable[[GridVariableVector], Tuple[GridVariableVector, ...]]:
    """
    Returns a function that computes the right-hand side (dG/dt) of the Navier-Stokes equations.
    
    This function computes the explicit contributions from convection and forcing at a given time t,
    without performing the state update or the implicit diffusion solve.
    
    Parameters:
      density: Fluid density.
      viscosity: Viscosity (not used in this explicit RHS computation).
      dt: Time step size (used for the convection scheme).
      custom_convection_velocity1: Custom coefficient for the x-direction velocity.
      custom_convection_velocity2: Custom coefficient for the y-direction velocity.
      grid: Grid information (unused in this implementation).
      convect: Convection function; if None, the default van Leer advection is used.
      forcing: Forcing function; if provided, its contribution is included.
      
    Returns:
      A function that takes the current state (GridVariableVector) and time (float) and returns
      the computed right-hand side dG/dt (with the same structure as GridVariableVector).
    """
    del grid  # 当前未使用传入的 grid

    if convect is None:
        def convect(v, G):
            return tuple(
                advection.advect_van_leer(u, v, dt) for u in G
            )

    convect = jax.named_call(convect, name='convection')

    @jax.named_call
    def compute_rhs(G: GridVariableVector, t: float):
        """
        Computes dG/dt (the right-hand side) at time t for the given state G,
        based on convection and (if provided) forcing terms.
        """
        # 构造局部速度场
        x_velocity_fn = lambda x, y: jnp.ones_like(x) * custom_convection_velocity1 * t
        x_velocity_0 = lambda x, y: jnp.ones_like(x) * 0.0
        y_velocity_fn = lambda x, y: jnp.ones_like(x) * custom_convection_velocity2 * t
        y_velocity_0 = lambda x, y: jnp.ones_like(x) * 0.0

        grid1 = grids.Grid((size, size), domain=((-2., 3.), (-2., 3.)))

        v1 = initial_conditions.initial_velocity_field((x_velocity_fn, y_velocity_fn), grid1)
        v2 = initial_conditions.initial_velocity_field((x_velocity_fn, y_velocity_0), grid1)
        v3 = initial_conditions.initial_velocity_field((x_velocity_0, y_velocity_fn), grid1)

        # 计算对流项
        convection1 = convect(v1, G)
        convection2 = convect(v2, G)
        convection3 = convect(v3, G)
        accelerations = [convection1, convection2, convection3]

        # 求和得到右侧项 dG/dt（仅包含对流和源项）
        return accelerations
    return compute_rhs