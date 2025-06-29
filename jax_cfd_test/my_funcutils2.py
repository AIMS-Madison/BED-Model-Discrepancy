def get_subgrid():
    """
    构造子网格，提取原始 251×251 网格中 [100:151,100:151] 的区域，
    对应的 domain 为 ((-2, 3), (-2, 3))，cell center 坐标利用 offsets=((0.5,0.5),(0.5,0.5)) 计算。
    
    返回:
      - x: 子网格上的 x 坐标数组，形状 (51, 51)
      - y: 子网格上的 y 坐标数组，形状 (51, 51)
    """
    grid = cfd.grids.Grid((251, 251), domain=((-2, 3), (-2, 3)))
    offsets = ((0.5, 0.5), (0.5, 0.5))
    # grid.mesh(offsets) 返回一个元组 (x_array, y_array)
    # 注意这里调用时顺序与 offsets 对应：第一个索引用于 x 坐标，第二个用于 y 坐标
    x_full = grid.mesh(offsets[1])[0]
    y_full = grid.mesh(offsets[0])[1]
    # 取中心区域
    x = x_full[100:151, 100:151]
    y = y_full[100:151, 100:151]
    return x, y

def compute_nn_output(params, model, center):
    """
    计算在子网格上（51×51）的神经网络输出。
    
    参数:
      - params: 神经网络参数
      - model: 神经网络模型，具有 apply(params, inputs) 方法
      - center: (center_x, center_y)，用于平移网格坐标
      
    返回:
      - array_nn: 神经网络在子网格上输出的数组，形状 (51,51)
    """
    x, y = get_subgrid()
    center_x, center_y = center
    # 平移坐标
    x_centered = x - center_x
    y_centered = y - center_y
    # 将二维网格坐标拉平成 (N,2) 形式，N = 51*51
    xy_combined = jnp.stack([x_centered.ravel(), y_centered.ravel()], axis=-1)
    # 计算神经网络输出
    array_nn = model.apply(params, xy_combined)
    array_nn = array_nn.reshape(x.shape)
    return array_nn


def project_onto_phi(nn_output, phi):
    """
    将神经网络输出在 phi 方向上做 L2 投影，计算投影系数 alpha：
         alpha = <nn_output, phi> / <phi, phi>
         
    参数:
      - nn_output: 神经网络输出数组，形状 (51,51)
      - phi: 对流方向数组，形状 (51,51)
      
    返回:
      - alpha: 标量投影系数
    """
    numerator = jnp.sum(nn_output * phi)
    denominator = jnp.sum(phi * phi)
    alpha = numerator / denominator
    return alpha

def compute_convection_direction_full(u_full, t):
    """
    利用全尺寸场（251×251）计算对流方向 phi = t * (∂u/∂y)。
    这里使用四阶中心差分（如果可能），然后提取中间区域 (51×51)。
    
    参数:
      - u_full: 完整的场数据，形状 (251,251)。
      - t: 当前时间（标量）。
      - grid: 包含 'y' 坐标的网格信息，假设 'y' 为 (251,251) 数组。
      
    返回:
      - phi: 提取中间区域（例如 [100:151, 100:151]）的对流方向分量数组，形状 (51,51)。
    """
    # 计算 y 方向步长，假设均匀网格
    _, y = get_subgrid()
    # 假设 y 方向均匀
    dy = y[1, 1] - y[0, 0]
    
    # 初始化导数数组，和 u_full 形状一致
    du_dy_full = jnp.zeros_like(u_full)
    
    # 对内部点使用四阶中心差分：i 从2 到 N-3 (251点时，i的取值为2~248)
    # 注意：这里我们假设数据足够“内部”，边界仍然需要特殊处理。
    #du_dy_full = du_dy_full.at[2:-2, :].set(
    #    (-u_full[4:, :] + 8*u_full[3:-1, :] - 8*u_full[1:-3, :] + u_full[0:-4, :]) / (12 * dy)
    #)
    du_dy_full = du_dy_full.at[:, 2:-2].set(
        (-u_full[:, 4:] + 8*u_full[:, 3:-1] - 8*u_full[:, 1:-3] + u_full[:, 0:-4]) / (12 * dy)
    )
    
    # 边界可以使用较低阶差分，这里为了简单可以采用二阶中心差分
    ## 上边界
    #du_dy_full = du_dy_full.at[1, :].set((u_full[2, :] - u_full[0, :]) / (2 * dy))
    ## 下边界
    #du_dy_full = du_dy_full.at[-2, :].set((u_full[-1, :] - u_full[-3, :]) / (2 * dy))
    ## 对于最外层的边界，可以用向前和向后差分
    #du_dy_full = du_dy_full.at[0, :].set((u_full[1, :] - u_full[0, :]) / dy)
    #du_dy_full = du_dy_full.at[-1, :].set((u_full[-1, :] - u_full[-2, :]) / dy)
    du_dy_full = du_dy_full.at[:, 1].set((u_full[:, 2] - u_full[:, 0]) / (2 * dy))
    du_dy_full = du_dy_full.at[:, -2].set((u_full[:, -1] - u_full[:, -3]) / (2 * dy))
    du_dy_full = du_dy_full.at[:, 0].set((u_full[:, 1] - u_full[:, 0]) / dy)
    du_dy_full = du_dy_full.at[:, -1].set((u_full[:, -1] - u_full[:, -2]) / dy)
    
    # 乘以 t 得到 phi
    phi_full = t * du_dy_full
    
    # 提取中间区域，假设取 [100:151, 100:151]
    phi = phi_full[100:151, 100:151]
    return phi
