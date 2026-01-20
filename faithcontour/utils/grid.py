import torch
import math
import numpy as np
from typing import Optional, Tuple, Iterator

CUBE_EDGES = np.array([
    # z=0 面环（0-2-6-4）
    [0, 2], [2, 6], [6, 4], [4, 0],
    # z=1 面环（1-3-7-5）
    [1, 3], [3, 7], [7, 5], [5, 1],
    # 垂直边（沿 z）
    (0, 1), (2, 3), (6, 7), (4, 5),
], dtype=np.int64)
CUBE_EDGES = torch.from_numpy(CUBE_EDGES)

CUBE_FACES = torch.tensor([
    [0, 2, 6, 4],  # -Z 面 (z=0, front face)
    [1, 5, 7, 3],  # +Z 面 (z=1, back face)
    [0, 1, 3, 2],  # -X 面 (x=0, left face)
    [4, 6, 7, 5],  # +X 面 (x=1, right face)
    [0, 4, 5, 1],  # -Y 面 (y=0, bottom face)
    [2, 3, 7, 6],  # +Y 面 (y=1, top face)
], dtype=torch.long)

CUBE_FACE_FLUX_INDICES = torch.tensor([
    # face 0: -Z [0,2,6,4]
    # 你原来用的是 [3,2,1,0]，这是 [0,1,2,3] 的反向，我们保留不动
    [3, 2, 1, 0],   # -Z face (front)

    # face 1: +Z [1,5,7,3]
    # 同理，你原来写 [4,5,6,7]，是 [7,6,5,4] 的反向，也保留
    [4, 5, 6, 7],   # +Z face (back)

    # face 2: -X [0,1,3,2]，正确边是 [8,4,9,0]，反向 = [0,9,4,8]
    [0, 9, 4, 8],   # -X face (left)

    # face 3: +X [4,6,7,5]，正确边是 [2,10,6,11]，反向 = [11,6,10,2]
    [11, 6, 10, 2], # +X face (right)

    # face 4: -Y [0,4,5,1]，正确边是 [3,11,7,8]，反向 = [8,7,11,3]
    [8, 7, 11, 3],  # -Y face (bottom)

    # face 5: +Y [2,3,7,6]，正确边是 [9,5,10,1]，反向 = [1,10,5,9]
    [1, 10, 5, 9],  # +Y face (top)
], dtype=torch.long)

# (The GridExtent class remains the same as before)
class GridExtent:
    """
    原/对偶网格工具（内存友好）：
      - 原网格范围 [-1, 1]，单元边长 h=2/res
      - 仅持有 1D 坐标；其余按需计算
      - 当不传索引时，默认返回“全部”
    """
    def __init__(self, res: int,
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 index_dtype: Optional[torch.dtype] = torch.long):
        if not isinstance(res, int) or res < 1:
            raise ValueError("res must be a positive int")
        self.res = int(res)
        self.p_num = res + 1              # 原顶点每轴数
        self.d_ext_num = res + 2          # 外扩对偶顶点每轴数
        self.h = 2.0 / res

        self.device = torch.device(device)
        self.dtype = dtype

        self.dual_num = self.p_num**3   # 对偶立方体数
        self.primal_num = res**3        # 原立方体数

        self.index_dtype = index_dtype

        # 1D 坐标
        self.p_coords_1d = torch.linspace(-1.0, 1.0, self.p_num, device=self.device, dtype=self.dtype)
        self.d_ext_coords_1d = torch.linspace(-1.0 - self.h/2.0, 1.0 + self.h/2.0,
                                              self.d_ext_num, device=self.device, dtype=self.dtype)

        # 角点偏移（8×3）
        self._corner_offsets = torch.tensor([
            [0,0,0],[0,0,1],[0,1,0],[0,1,1],
            [1,0,0],[1,0,1],[1,1,0],[1,1,1]
        ], device=self.device, dtype=self.index_dtype)

        # 对偶单元中心相对偏移（与上面对应；{0,-1}）
        self._cell_offsets_dual = torch.tensor([
            [ 0,  0,  0],[ 0,  0, -1],[ 0, -1,  0],[ 0, -1, -1],
            [-1,  0,  0],[-1,  0, -1],[-1, -1,  0],[-1, -1, -1]
        ], device=self.device, dtype=self.index_dtype)

        # 线性化乘数
        self._p_mul    = torch.tensor([self.p_num*self.p_num, self.p_num, 1], device=self.device, dtype=self.index_dtype)
        self._d_ext_mul= torch.tensor([self.d_ext_num*self.d_ext_num, self.d_ext_num, 1], device=self.device, dtype=self.index_dtype)

        # 立方体12条边的定义 (基于_corner_offsets的局部索引)
        self.CUBE_EDGES = CUBE_EDGES.to(self.device)

        # 6个面中心的坐标
        self.CUBE_FACES = CUBE_FACES.to(self.device)

        self._face_neighbor_offsets = torch.tensor([
            [0, 0, -1],  # Corresponds to CUBE_FACES[0] (-Z)
            [0, 0, 1],   # Corresponds to CUBE_FACES[1] (+Z)
            [-1, 0, 0],  # Corresponds to CUBE_FACES[2] (-X)
            [1, 0, 0],   # Corresponds to CUBE_FACES[3] (+X)
            [0, -1, 0],  # Corresponds to CUBE_FACES[4] (-Y)
            [0, 1, 0],   # Corresponds to CUBE_FACES[5] (+Y)
        ], device=self.device, dtype=self.index_dtype)

        self.CUBE_FACE_FLUX_INDICES = CUBE_FACE_FLUX_INDICES.to(self.device)

        # 为 x, y, z 三个方向的 primal edge 定义其 dual face 的顶点偏移
        # key: 方向向量 (元组形式)
        # value: 4个外扩对偶顶点的 ijk 偏移量 (相对于边的起始原始顶点)
        self._dual_face_v_offsets = {
            (1, 0, 0): torch.tensor([[1,0,0], [1,1,0], [1,1,1], [1,0,1]], device=self.device, dtype=self.index_dtype), # +X face
            (0, 1, 0): torch.tensor([[0,1,0], [0,1,1], [1,1,1], [1,1,0]], device=self.device, dtype=self.index_dtype), # +Y face
            (0, 0, 1): torch.tensor([[0,0,1], [1,0,1], [1,1,1], [0,1,1]], device=self.device, dtype=self.index_dtype), # +Z face
        }
        # 为负方向也添加映射
        self._dual_face_v_offsets[(-1, 0, 0)] = self._dual_face_v_offsets[(1, 0, 0)] - torch.tensor([1,0,0], device=self.device, dtype=self.index_dtype)
        self._dual_face_v_offsets[(0, -1, 0)] = self._dual_face_v_offsets[(0, 1, 0)] - torch.tensor([0,1,0], device=self.device, dtype=self.index_dtype)
        self._dual_face_v_offsets[(0, 0, -1)] = self._dual_face_v_offsets[(0, 0, 1)] - torch.tensor([0,0,1], device=self.device, dtype=self.index_dtype)

        # --- 为向量化操作预计算辅助张量 ---
        # 1. 每条边的起始角点偏移 (12, 3)
        self._edge_start_corner_offsets = self._corner_offsets[self.CUBE_EDGES[:, 0]]

        # 2. 每条边对应的对偶面顶点偏移 (12, 4, 3)
        directions = self._corner_offsets[self.CUBE_EDGES[:, 1]] - self._edge_start_corner_offsets
        offsets_list = [self._dual_face_v_offsets[tuple(d.tolist())] for d in directions]
        self._d_ext_v_offsets_all_edges = torch.stack(offsets_list, dim=0)


        # --- edge 方向 & anchor 局部坐标（只占 12 条边的常量内存） ---
        corner = self._corner_offsets          # (8,3)
        e0 = corner[self.CUBE_EDGES[:, 0]]     # (12,3)
        e1 = corner[self.CUBE_EDGES[:, 1]]     # (12,3)
        self._edge_dirs_local = e1 - e0        # (12,3)，每条边的方向向量 ∈ {-1,0,1}
        # axis: 0=x, 1=y, 2=z
        self._edge_axis = torch.argmax(self._edge_dirs_local.abs(), dim=1).to(self.index_dtype)   # (12,)

        # 每条边的 anchor 顶点（在立方体局部 0/1 坐标里取分量最小的那个端点）
        self._edge_anchor_local = torch.minimum(e0, e1).to(self.index_dtype)  # (12,3)

        # 各方向 edge 数量 + 偏移
        self._num_edges_x = self.res       * self.p_num * self.p_num
        self._num_edges_y = self.p_num     * self.res   * self.p_num
        self._num_edges_z = self.p_num     * self.p_num * self.res

        self._edge_offset_x = torch.tensor(0, device=self.device, dtype=self.index_dtype)
        self._edge_offset_y = torch.tensor(self._num_edges_x, device=self.device, dtype=self.index_dtype)
        self._edge_offset_z = torch.tensor(self._num_edges_x + self._num_edges_y,
                                        device=self.device, dtype=self.index_dtype)

        # 各方向 face 数量 + 偏移：
        # faces_x: 法向 ±x, (i in [0,p_num-1], j,k in [0,res-1])
        self._num_faces_x = self.p_num * self.res * self.res
        # faces_y: 法向 ±y, (i in [0,res-1], j in [0,p_num-1], k in [0,res-1])
        self._num_faces_y = self.res * self.p_num * self.res
        # faces_z: 法向 ±z, (i,j in [0,res-1], k in [0,p_num-1])
        self._num_faces_z = self.res * self.res * self.p_num

        self._face_offset_x = torch.tensor(0, device=self.device, dtype=self.index_dtype)
        self._face_offset_y = torch.tensor(self._num_faces_x, device=self.device, dtype=self.index_dtype)
        self._face_offset_z = torch.tensor(self._num_faces_x + self._num_faces_y,
                                        device=self.device, dtype=self.index_dtype)

        # 每个局部面对应的轴向 (0:x,1:y,2:z) 和正负号
        self._face_axis = torch.tensor([2, 2, 0, 0, 1, 1], device=self.device, dtype=self.index_dtype)
        self._face_is_pos = torch.tensor([0, 1, 0, 1, 0, 1], device=self.device, dtype=self.index_dtype)
        # 这里 face 的顺序对应你的 CUBE_FACES:
        # 0:-Z, 1:+Z, 2:-X, 3:+X, 4:-Y, 5:+Y


    def primal_cube_face_indices(self, cube_idx: torch.Tensor) -> torch.Tensor:
        """
        给一批 primal cube，全局 index (B,)，返回每个 cube 的 6 个面的全局 face index：
          形状 (B, 6)

        全程公式计算，face 全局编号规则：
          - faces_x: dims=(p_num, res, res), offset=_face_offset_x
          - faces_y: dims=(res, p_num, res), offset=_face_offset_y
          - faces_z: dims=(res, res, p_num), offset=_face_offset_z
        """
        if cube_idx.dim() != 1:
            raise ValueError("cube_idx must be 1D")

        device = self.device
        itype = self.index_dtype
        B = cube_idx.numel()
        if B == 0:
            return torch.empty(0, 6, dtype=itype, device=device)

        ijk = self.primal_cube_indices_ijk(cube_idx).to(itype)  # (B,3)
        cx, cy, cz = ijk[:, 0], ijk[:, 1], ijk[:, 2]

        # z-faces: dims = (res, res, p_num)
        f_minus_z = self.ravel_ijk(torch.stack([cx, cy, cz], dim=-1), (self.res, self.res, self.p_num)) + self._face_offset_z
        f_plus_z  = self.ravel_ijk(torch.stack([cx, cy, cz + 1], dim=-1), (self.res, self.res, self.p_num)) + self._face_offset_z

        # x-faces: dims = (p_num, res, res)
        f_minus_x = self.ravel_ijk(torch.stack([cx,     cy, cz], dim=-1), (self.p_num, self.res, self.res)) + self._face_offset_x
        f_plus_x  = self.ravel_ijk(torch.stack([cx + 1, cy, cz], dim=-1), (self.p_num, self.res, self.res)) + self._face_offset_x

        # y-faces: dims = (res, p_num, res)
        f_minus_y = self.ravel_ijk(torch.stack([cx, cy,     cz], dim=-1), (self.res, self.p_num, self.res)) + self._face_offset_y
        f_plus_y  = self.ravel_ijk(torch.stack([cx, cy + 1, cz], dim=-1), (self.res, self.p_num, self.res)) + self._face_offset_y

        # 对应 CUBE_FACES 的顺序: 0:-Z,1:+Z,2:-X,3:+X,4:-Y,5:+Y
        faces = torch.stack(
            [f_minus_z, f_plus_z, f_minus_x, f_plus_x, f_minus_y, f_plus_y],
            dim=1
        )  # (B,6)

        return faces
    
    def primal_cube_edge_indices(self, cube_idx: torch.Tensor) -> torch.Tensor:
        """
        给定一批 primal voxel 的全局索引 [B]，返回它们 12 条局部边的全局 edge index，[B,12]。

        全局 edge 编号约定：
          - 先 x 方向的 edge，一共 self._num_edges_x 条，偏移量 self._edge_offset_x
          - 再 y 方向的 edge，一共 self._num_edges_y 条，偏移量 self._edge_offset_y
          - 最后 z 方向的 edge，一共 self._num_edges_z 条，偏移量 self._edge_offset_z

        依赖：
          - self._edge_anchor_local: (12,3)，局部 anchor 顶点偏移（0/1）
          - self._edge_axis:         (12,) ∈ {0,1,2}
        """
        if cube_idx is None:
            cube_idx = self.all_primal_cube_indices()
        # 统一到 device + dtype
        cube_idx = cube_idx.to(self.device)
        cube_ijk = self.primal_cube_indices_ijk(cube_idx)                 # (B,3), index_dtype, self.device
        cube_ijk = cube_ijk.to(dtype=self.index_dtype, device=self.device)

        B = cube_ijk.shape[0]

        # (12,3) on correct device/dtype
        anchor_local = self._edge_anchor_local.to(device=self.device,
                                                  dtype=self.index_dtype) # (12,3)

        # (B,12,3) = (B,1,3) + (1,12,3)
        anchor_ijk = cube_ijk[:, None, :] + anchor_local[None, :, :]      # (B,12,3)

        # edge 对应的方向轴 0:x,1:y,2:z
        edge_axis = self._edge_axis.to(device=self.device,
                                       dtype=self.index_dtype)            # (12,)
        edge_axis_b = edge_axis.view(1, 12).expand(B, 12)                 # (B,12)

        # 准备输出
        E = torch.empty((B, 12),
                        device=self.device,
                        dtype=self.index_dtype)                           # (B,12)

        # ----- x 方向 edge -----
        mask_x = (edge_axis_b == 0)                                       # (B,12)
        if mask_x.any():
            idx_x = mask_x
            ijk_x = anchor_ijk[idx_x]                                     # (Nx,3) anchor_ijk: (ix, jv, kv)
            ix = ijk_x[:, 0]
            jv = ijk_x[:, 1]
            kv = ijk_x[:, 2]
            # dims: (res, p_num, p_num)
            ex_local = ix * (self.p_num * self.p_num) + jv * self.p_num + kv
            ex_local = ex_local.to(self.index_dtype)
            E[idx_x] = ex_local + self._edge_offset_x

        # ----- y 方向 edge -----
        mask_y = (edge_axis_b == 1)
        if mask_y.any():
            idx_y = mask_y
            ijk_y = anchor_ijk[idx_y]                                     # (Ny,3) (iv, jy, kv)
            iv = ijk_y[:, 0]
            jy = ijk_y[:, 1]
            kv = ijk_y[:, 2]
            # dims: (p_num, res, p_num)
            ey_local = iv * (self.res * self.p_num) + jy * self.p_num + kv
            ey_local = ey_local.to(self.index_dtype)
            E[idx_y] = ey_local + self._edge_offset_y

        # ----- z 方向 edge -----
        mask_z = (edge_axis_b == 2)
        if mask_z.any():
            idx_z = mask_z
            ijk_z = anchor_ijk[idx_z]                                     # (Nz,3) (iv, jv, kz)
            iv = ijk_z[:, 0]
            jv = ijk_z[:, 1]
            kz = ijk_z[:, 2]
            # dims: (p_num, p_num, res)
            ez_local = iv * (self.p_num * self.res) + jv * self.res + kz
            ez_local = ez_local.to(self.index_dtype)
            E[idx_z] = ez_local + self._edge_offset_z

        return E  # [B,12], index_dtype, self.device
    

    def primal_edge_endpoints_coords(self, edge_idx: torch.Tensor) -> torch.Tensor:
        """
        给任意形状的 primal edge 全局 index，返回每条 edge 两个端点的坐标。

        输入:
            edge_idx: [...,]  任意形状的整型张量，全局 edge 线性索引

        输出:
            coords:  [..., 2, 3]
                coords[..., 0, :] = 端点 v0 的 (x,y,z)
                coords[..., 1, :] = 端点 v1 的 (x,y,z)
        """
        dev   = self.device
        itype = self.index_dtype

        # 保留原始形状，统一 flatten 成 (E,)
        orig_shape = edge_idx.shape
        edge_flat  = edge_idx.to(itype).to(dev).reshape(-1)
        E = edge_flat.numel()

        if E == 0:
            return torch.empty(*orig_shape, 2, 3, dtype=self.dtype, device=dev)

        coords_flat = torch.empty(E, 2, 3, dtype=self.dtype, device=dev)

        res   = self.res
        p_num = self.p_num

        # 三段 edge 区间
        off_x = self._edge_offset_x
        off_y = self._edge_offset_y
        off_z = self._edge_offset_z

        # ---------- X 方向的 edge ----------
        mask_x = (edge_flat >= off_x) & (edge_flat < off_y)
        if mask_x.any():
            idx_x   = mask_x.nonzero(as_tuple=False).squeeze(1)       # (Nx,)
            local_x = edge_flat[idx_x] - off_x                         # (Nx,)
            # dims = (res, p_num, p_num)，ijk = (ix, jy, kz)
            ijk_x   = self.unravel_idx(local_x, (res, p_num, p_num))  # (Nx,3)
            ix, jy, kz = ijk_x[:, 0], ijk_x[:, 1], ijk_x[:, 2]

            # v0 = (ix    , jy, kz)
            # v1 = (ix + 1, jy, kz)
            v0_ijk = torch.stack([ix,     jy, kz], dim=-1)  # (Nx,3)
            v1_ijk = torch.stack([ix + 1, jy, kz], dim=-1)  # (Nx,3)

            v0_idx = self.ravel_ijk(v0_ijk, (p_num, p_num, p_num))
            v1_idx = self.ravel_ijk(v1_ijk, (p_num, p_num, p_num))

            v0_xyz = self.primal_vertex_coords_from_indices(v0_idx)   # (Nx,3)
            v1_xyz = self.primal_vertex_coords_from_indices(v1_idx)   # (Nx,3)

            coords_flat[idx_x, 0, :] = v0_xyz
            coords_flat[idx_x, 1, :] = v1_xyz

        # ---------- Y 方向的 edge ----------
        mask_y = (edge_flat >= off_y) & (edge_flat < off_z)
        if mask_y.any():
            idx_y   = mask_y.nonzero(as_tuple=False).squeeze(1)       # (Ny,)
            local_y = edge_flat[idx_y] - off_y                         # (Ny,)
            # dims = (p_num, res, p_num)，ijk = (ix, jy, kz)
            ijk_y   = self.unravel_idx(local_y, (p_num, res, p_num))  # (Ny,3)
            ix, jy, kz = ijk_y[:, 0], ijk_y[:, 1], ijk_y[:, 2]

            # v0 = (ix, jy    , kz)
            # v1 = (ix, jy + 1, kz)
            v0_ijk = torch.stack([ix, jy,     kz], dim=-1)
            v1_ijk = torch.stack([ix, jy + 1, kz], dim=-1)

            v0_idx = self.ravel_ijk(v0_ijk, (p_num, p_num, p_num))
            v1_idx = self.ravel_ijk(v1_ijk, (p_num, p_num, p_num))

            v0_xyz = self.primal_vertex_coords_from_indices(v0_idx)
            v1_xyz = self.primal_vertex_coords_from_indices(v1_idx)

            coords_flat[idx_y, 0, :] = v0_xyz
            coords_flat[idx_y, 1, :] = v1_xyz

        # ---------- Z 方向的 edge ----------
        mask_z = edge_flat >= off_z
        if mask_z.any():
            idx_z   = mask_z.nonzero(as_tuple=False).squeeze(1)       # (Nz,)
            local_z = edge_flat[idx_z] - off_z                         # (Nz,)
            # dims = (p_num, p_num, res)，ijk = (ix, jy, kz)
            ijk_z   = self.unravel_idx(local_z, (p_num, p_num, res))  # (Nz,3)
            ix, jy, kz = ijk_z[:, 0], ijk_z[:, 1], ijk_z[:, 2]

            # v0 = (ix, jy, kz    )
            # v1 = (ix, jy, kz + 1)
            v0_ijk = torch.stack([ix, jy, kz    ], dim=-1)
            v1_ijk = torch.stack([ix, jy, kz + 1], dim=-1)

            v0_idx = self.ravel_ijk(v0_ijk, (p_num, p_num, p_num))
            v1_idx = self.ravel_ijk(v1_ijk, (p_num, p_num, p_num))

            v0_xyz = self.primal_vertex_coords_from_indices(v0_idx)
            v1_xyz = self.primal_vertex_coords_from_indices(v1_idx)

            coords_flat[idx_z, 0, :] = v0_xyz
            coords_flat[idx_z, 1, :] = v1_xyz

        # reshape 回原有 batch 形状
        coords = coords_flat.view(*orig_shape, 2, 3)
        return coords

    
    def primal_edge_incident_cubes(self, edge_idx: torch.Tensor) -> torch.Tensor:
        """
        给一批全局 edge index，返回每条 edge 邻接的 primal 体元（voxel）index。
        - 输入: edge_idx 形状 (E,)
        - 输出: cubes 形状 (E, 4)，最多 4 个 cube，若不存在则为 -1

        全程按公式计算，不预存全局表。
        """
        if edge_idx.dim() != 1:
            raise ValueError("edge_idx must be 1D (E,) tensor")

        device = self.device
        itype = self.index_dtype
        E = edge_idx.numel()
        if E == 0:
            return torch.empty(0, 4, dtype=itype, device=device)

        edge_idx = edge_idx.to(itype).to(device)

        # ----- 计算每个方向的 edge 数量 & 偏移 -----
        res    = self.res
        p_num  = self.p_num

        num_edges_x = res * p_num * p_num
        num_edges_y = p_num * res * p_num
        num_edges_z = p_num * p_num * res

        offset_x = 0
        offset_y = num_edges_x
        offset_z = num_edges_x + num_edges_y

        # 输出初始化为 -1（表示不存在）
        out = torch.full((E, 4), -1, dtype=itype, device=device)

        # ----- helper: 做一个 block 内的 4 邻近 cube 计算 -----
        def _assign_cubes(mask: torch.Tensor, local_to_ijk, candidate_builder):
            """
            mask: (E,) bool, 选出属于某个方向 block 的 edge
            local_to_ijk: 函数 local_idx -> (N,3) ijk
            candidate_builder: 函数(ijk: (N,3)) -> candidate cube ijk (N,4,3)
            """
            idx = mask.nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                return
            local = local_to_ijk(idx)  # 返回 (N,3) 的 ijk（edge anchor 坐标）
            cubes_ijk = candidate_builder(local)  # (N,4,3)
            N = cubes_ijk.shape[0]

            cubes_flat = cubes_ijk.view(-1, 3)    # (N*4,3)
            # 合法范围: 0 <= i,j,k < res
            valid = (cubes_flat[:, 0] >= 0) & (cubes_flat[:, 0] < res) & \
                    (cubes_flat[:, 1] >= 0) & (cubes_flat[:, 1] < res) & \
                    (cubes_flat[:, 2] >= 0) & (cubes_flat[:, 2] < res)

            cube_idx_flat = self.ravel_ijk(cubes_flat, (res, res, res))
            # 初始化 block 内输出为 -1
            block_out = torch.full((N, 4), -1, dtype=itype, device=device)

            if valid.any():
                vid = valid.nonzero(as_tuple=False).squeeze(1)      # 哪些 candidate 有效
                cubes_valid = cube_idx_flat[vid]                    # (Nv,)
                n_id = torch.div(vid, 4, rounding_mode='floor')     # 属于第几个 edge（0..N-1）
                slot = vid % 4                                      # 属于该 edge 的第几个 candidate（0..3）
                block_out[n_id, slot] = cubes_valid.to(self.index_dtype)

            # 写回全局 out
            out[idx] = block_out

        # ----- x-block -----
        mask_x = (edge_idx >= offset_x) & (edge_idx < offset_y)
        def _local_to_ijk_x(idx_tensor: torch.Tensor) -> torch.Tensor:
            local = edge_idx[idx_tensor] - offset_x
            return self.unravel_idx(local, (res, p_num, p_num))  # (N,3) -> (ix,jy,kz)

        def _cubes_from_x_edge(edge_ijk: torch.Tensor) -> torch.Tensor:
            # edge_ijk: (N,3) = (ix, jy, kz)
            ix = edge_ijk[:, 0]
            jy = edge_ijk[:, 1]
            kz = edge_ijk[:, 2]
            N  = ix.numel()

            cy_opts = torch.stack([jy - 1, jy], dim=1)  # (N,2)
            cz_opts = torch.stack([kz - 1, kz], dim=1)  # (N,2)

            cx = ix[:, None, None].expand(N, 2, 2)
            cy = cy_opts[:, :, None].expand(N, 2, 2)
            cz = cz_opts[:, None, :].expand(N, 2, 2)

            cubes_ijk = torch.stack([cx, cy, cz], dim=-1)  # (N,2,2,3)
            return cubes_ijk.view(N, 4, 3)

        _assign_cubes(mask_x, _local_to_ijk_x, _cubes_from_x_edge)

        # ----- y-block -----
        mask_y = (edge_idx >= offset_y) & (edge_idx < offset_z)
        def _local_to_ijk_y(idx_tensor: torch.Tensor) -> torch.Tensor:
            local = edge_idx[idx_tensor] - offset_y
            return self.unravel_idx(local, (p_num, res, p_num))  # (ix,jy,kz)

        def _cubes_from_y_edge(edge_ijk: torch.Tensor) -> torch.Tensor:
            # edge_ijk: (N,3) = (ix, jy, kz)
            ix = edge_ijk[:, 0]
            jy = edge_ijk[:, 1]
            kz = edge_ijk[:, 2]
            N  = ix.numel()

            cx_opts = torch.stack([ix - 1, ix], dim=1)  # (N,2)
            cz_opts = torch.stack([kz - 1, kz], dim=1)  # (N,2)

            cx = cx_opts[:, :, None].expand(N, 2, 2)
            cy = jy[:, None, None].expand(N, 2, 2)
            cz = cz_opts[:, None, :].expand(N, 2, 2)

            cubes_ijk = torch.stack([cx, cy, cz], dim=-1)  # (N,2,2,3)
            return cubes_ijk.view(N, 4, 3)

        _assign_cubes(mask_y, _local_to_ijk_y, _cubes_from_y_edge)

        # ----- z-block -----
        mask_z = edge_idx >= offset_z
        def _local_to_ijk_z(idx_tensor: torch.Tensor) -> torch.Tensor:
            local = edge_idx[idx_tensor] - offset_z
            return self.unravel_idx(local, (p_num, p_num, res))  # (ix,jy,kz)

        def _cubes_from_z_edge(edge_ijk: torch.Tensor) -> torch.Tensor:
            ix = edge_ijk[:, 0]
            jy = edge_ijk[:, 1]
            kz = edge_ijk[:, 2]
            N  = ix.numel()

            cx_opts = torch.stack([ix - 1, ix], dim=1)
            cy_opts = torch.stack([jy - 1, jy], dim=1)

            cx = cx_opts[:, :, None].expand(N, 2, 2)
            cy = cy_opts[:, None, :].expand(N, 2, 2)
            cz = kz[:, None, None].expand(N, 2, 2)

            cubes_ijk = torch.stack([cx, cy, cz], dim=-1)  # (N,2,2,3)
            return cubes_ijk.view(N, 4, 3)

        _assign_cubes(mask_z, _local_to_ijk_z, _cubes_from_z_edge)

        return out
    
    def edge_flux_to_face_flux(self,
                            edge_flux: torch.Tensor,
                            mode: str = "any_nonzero",
                            thresh: int = 1) -> torch.Tensor:
        """
        edge_flux: [K, 12]  每个 voxel 12 条 edge 的 flux（-1/0/1 或实数）
        mode:
            - "sum_sign" : 对 4 条边求和再 sign
            - "majority" : 多数表决（正/负票差超过阈值才给方向）
            - "any_nonzero": 只要有非零就给方向（更激进，洞会少）
            - "all_consistent": 4 条边必须同号才给方向（最保守，洞多）
        thresh:
            - 对 "majority" 生效，表示 |pos - neg| >= thresh 才给非零方向

        return:
            face_flux: [K, 6] 每个面的净 flux
        """
        if edge_flux.dim() != 2 or edge_flux.size(1) != 12:
            raise ValueError(f"edge_flux must be [K,12], got {edge_flux.shape}")

        K = edge_flux.size(0)
        device = edge_flux.device
        dtype = edge_flux.dtype

        face2edges = self.CUBE_FACE_FLUX_INDICES.to(device)  # [6,4]

        # [K, 6, 12]
        edge_flux_exp = edge_flux.unsqueeze(1).expand(K, 6, 12)
        # [K, 6, 4]  每个面对应的 4 条边的 flux
        face_edges = torch.gather(
            edge_flux_exp,
            dim=2,
            index=face2edges.unsqueeze(0).expand(K, -1, -1)
        )

        if mode == "sum_sign":
            # 以前那种：sum 再 sign
            face_flux = face_edges.sum(dim=-1).sign()

        elif mode == "majority":
            # 多数投票：统计正/负票
            pos = (face_edges > 0).int().sum(dim=-1)   # [K,6]
            neg = (face_edges < 0).int().sum(dim=-1)
            diff = pos - neg                           # >0 说明整体偏正，<0 偏负
            face_flux = torch.zeros_like(diff, dtype=dtype)

            face_flux = torch.where(diff >= thresh,  torch.ones_like(face_flux), face_flux)
            face_flux = torch.where(diff <= -thresh, -torch.ones_like(face_flux), face_flux)
            # 剩下 |diff| < thresh 的保持 0

        elif mode == "any_nonzero":
            # 只要有非零 edge，就用「平均方向」
            nonzero = (face_edges != 0).float()          # [K,6,4]
            nz_count = nonzero.sum(dim=-1, keepdim=True) # [K,6,1]
            masked_sum = (face_edges * nonzero).sum(dim=-1, keepdim=True)
            avg = masked_sum / (nz_count + 1e-8)
            face_flux = torch.where(nz_count > 0, avg.sign(), torch.zeros_like(avg))
            face_flux = face_flux.squeeze(-1)

        elif mode == "all_consistent":
            # 4 条边必须全部同号，否则这个面给 0
            # （这个是拓扑保守版，洞最多但最干净）
            sign_edges = face_edges.sign()              # [-1,0,1]
            # 如果存在 0，或同时有正有负，就视为不一致
            same_sign = (
                (sign_edges.abs().sum(dim=-1) == 4) &       # 都非零
                ((sign_edges.sum(dim=-1).abs()) == 4)       # 且全同号
            )
            # 提取任一条边的符号（都一样）
            face_flux = torch.where(
                same_sign,
                sign_edges[..., 0],   # 随便拿一个
                torch.zeros_like(sign_edges[..., 0])
            ).to(dtype)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return face_flux.to(dtype)

    def face_flux_to_edge_flux(self,
                           face_flux: torch.Tensor,
                            mode: str = "any") -> torch.Tensor:
        """
        face_flux: [K, 6]
        mode:
            - "sum_sign": 默认，6 个面对每条 edge 投票，累加再 sign
            - "any": 任一 incident face 非零，就给该方向
            - "consensus": 只有 incident faces 同号时才非零
        """
        if face_flux.dim() != 2 or face_flux.size(1) != 6:
            raise ValueError(f"face_flux must be [K,6], got {face_flux.shape}")

        K = face_flux.size(0)
        device = face_flux.device
        dtype = face_flux.dtype

        edge_flux = torch.zeros(K, 12, device=device, dtype=dtype)
        face2edges = self.CUBE_FACE_FLUX_INDICES.to(device)  # [6,4]

        if mode == "sum_sign":
            for f in range(6):
                val = face_flux[:, f:f+1]            # [K,1]
                e_idx = face2edges[f]                # [4]
                edge_flux[:, e_idx] += val
            edge_flux = edge_flux.sign()

        elif mode == "any":
            for f in range(6):
                val = face_flux[:, f:f+1]            # [K,1]
                e_idx = face2edges[f]
                # 只要有任一面非零，就直接覆盖成那一面的方向
                mask = (val != 0)
                edge_flux[:, e_idx] = torch.where(
                    mask,
                    val.expand_as(edge_flux[:, e_idx]),
                    edge_flux[:, e_idx],
                )

        elif mode == "consensus":
            # edges 的 incident faces 只有 2 个（最多），我们可以先建立一个 [12,2] 的 "edge->faces" 映射，
            # 然后对两个面做一致性检查。
            # 为了不预存大表，这里直接手工建一个小的 edge->faces_local 映射（常量 12x2）。
            edge2faces_local = torch.zeros(12, 2, device=device, dtype=torch.long) - 1
            counts = torch.zeros(12, device=device, dtype=torch.long)

            for f in range(6):
                e_idx = face2edges[f]  # [4]
                for ei in e_idx:
                    c = counts[ei]
                    if c < 2:
                        edge2faces_local[ei, c] = f
                        counts[ei] = c + 1

            edge_flux = torch.zeros(K, 12, device=device, dtype=dtype)
            for e in range(12):
                f0, f1 = edge2faces_local[e]  # 可能有 -1（边界）
                vals = []
                if f0 >= 0:
                    vals.append(face_flux[:, f0])
                if f1 >= 0:
                    vals.append(face_flux[:, f1])
                if len(vals) == 0:
                    continue
                vals = torch.stack(vals, dim=-1)      # [K, n_faces<=2]
                s = vals.sign()
                # 一致性：所有非零且同号
                same_sign = (
                    (s.abs().sum(dim=-1) == (s != 0).sum(dim=-1)) &  # 全非零
                    (s.sum(dim=-1).abs() == (s != 0).sum(dim=-1))    # 全同号
                )
                edge_flux[:, e] = torch.where(
                    same_sign,
                    vals.mean(dim=-1).sign(),
                    torch.zeros_like(vals[..., 0])
                )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return edge_flux

    def primal_edge_incident_cubes_righthand(self, edge_idx: torch.Tensor) -> torch.Tensor:
        """
        给定若干 global edge index [E]，返回每条边最多 4 个相邻 primal cubes 的全局索引 [E,4]，
        顺序满足：沿着 edge 的 +axis 方向看过去时，四个 voxel 按右手定则逆时针排列。

        无相邻 voxel 的位置填 -1。
        """
        # 统一 dtype / device
        edge_idx = edge_idx.to(device=self.device, dtype=self.index_dtype)
        E = edge_idx.numel()

        # 默认填 -1，dtype 使用 index_dtype
        cubes_out = torch.full(
            (E, 4), -1,
            device=self.device,
            dtype=self.index_dtype,
        )

        num_ex = self._num_edges_x
        num_ey = self._num_edges_y
        num_ez = self._num_edges_z

        # ---------- X 方向边 ----------
        mask_x = (edge_idx < num_ex)
        if mask_x.any():
            idx_x = torch.nonzero(mask_x, as_tuple=False).squeeze(-1)          # [Ex]
            e_local_x = edge_idx[idx_x] - self._edge_offset_x                  # [Ex]

            # dims_x = (res, p_num, p_num): (ix, jv, kv)
            ijk_x = self.unravel_idx(e_local_x, (self.res, self.p_num, self.p_num))  # [Ex,3]
            ijk_x = ijk_x.to(self.index_dtype)

            ix = ijk_x[:, 0:1]  # [Ex,1]
            jv = ijk_x[:, 1:2]
            kv = ijk_x[:, 2:3]

            # 看向 +x，平面 yz：
            # (0) (y,   z)
            # (1) (y,   z-1)
            # (2) (y-1, z-1)
            # (3) (y-1, z)
            dy = torch.tensor([0, 0, 1, 1], device=self.device, dtype=self.index_dtype)
            dz = torch.tensor([0, 1, 1, 0], device=self.device, dtype=self.index_dtype)

            cy = jv - dy        # [Ex,4]
            cz = kv - dz        # [Ex,4]
            cx = ix.expand_as(cy)

            valid_x = (cx >= 0) & (cx < self.res) & \
                      (cy >= 0) & (cy < self.res) & \
                      (cz >= 0) & (cz < self.res)

            cubes_x = torch.stack([cx, cy, cz], dim=-1)     # [Ex,4,3]
            cubes_x_flat = cubes_x.view(-1, 3)
            cubes_x_idx = self.ravel_ijk(cubes_x_flat, (self.res, self.res, self.res))
            cubes_x_idx = cubes_x_idx.view(-1, 4)

            cubes_x_idx = torch.where(
                valid_x,
                cubes_x_idx,
                torch.full_like(cubes_x_idx, -1),
            )

            # ★ 关键：写回之前强制 cast 到 index_dtype
            cubes_out[idx_x] = cubes_x_idx.to(self.index_dtype)

        # ---------- Y 方向边 ----------
        mask_y = (edge_idx >= num_ex) & (edge_idx < num_ex + num_ey)
        if mask_y.any():
            idx_y = torch.nonzero(mask_y, as_tuple=False).squeeze(-1)          # [Ey]
            e_local_y = edge_idx[idx_y] - self._edge_offset_y                  # [Ey]

            # dims_y = (p_num, res, p_num): (iv, jy, kv)
            ijk_y = self.unravel_idx(e_local_y, (self.p_num, self.res, self.p_num))
            ijk_y = ijk_y.to(self.index_dtype)

            iv = ijk_y[:, 0:1]
            jy = ijk_y[:, 1:2]
            kv = ijk_y[:, 2:3]

            # 看向 +y，平面 xz：
            # (0) (x-1, z)
            # (1) (x-1, z-1)
            # (2) (x,   z-1)
            # (3) (x,   z)
            dx_y = torch.tensor([-1, -1, 0, 0], device=self.device, dtype=self.index_dtype)
            dz_y = torch.tensor([0, -1, -1, 0], device=self.device, dtype=self.index_dtype)

            cx = iv + dx_y
            cy = jy.expand_as(cx)
            cz = kv + dz_y

            valid_y = (cx >= 0) & (cx < self.res) & \
                      (cy >= 0) & (cy < self.res) & \
                      (cz >= 0) & (cz < self.res)

            cubes_y = torch.stack([cx, cy, cz], dim=-1)     # [Ey,4,3]
            cubes_y_flat = cubes_y.view(-1, 3)
            cubes_y_idx = self.ravel_ijk(cubes_y_flat, (self.res, self.res, self.res))
            cubes_y_idx = cubes_y_idx.view(-1, 4)

            cubes_y_idx = torch.where(
                valid_y,
                cubes_y_idx,
                torch.full_like(cubes_y_idx, -1),
            )

            cubes_out[idx_y] = cubes_y_idx.to(self.index_dtype)

        # ---------- Z 方向边 ----------
        mask_z = ~(mask_x | mask_y)
        if mask_z.any():
            idx_z = torch.nonzero(mask_z, as_tuple=False).squeeze(-1)          # [Ez]
            e_local_z = edge_idx[idx_z] - self._edge_offset_z                  # [Ez]

            # dims_z = (p_num, p_num, res): (iv, jv, kz)
            ijk_z = self.unravel_idx(e_local_z, (self.p_num, self.p_num, self.res))
            ijk_z = ijk_z.to(self.index_dtype)

            iv = ijk_z[:, 0:1]
            jv = ijk_z[:, 1:2]
            kz = ijk_z[:, 2:3]

            # 看向 +z，平面 xy：
            # (0) (x-1, y)
            # (1) (x,   y)
            # (2) (x,   y-1)
            # (3) (x-1, y-1)
            dx_z = torch.tensor([-1, 0, 0, -1], device=self.device, dtype=self.index_dtype)
            dy_z = torch.tensor([0, 0, -1, -1], device=self.device, dtype=self.index_dtype)

            cx = iv + dx_z
            cy = jv + dy_z
            cz = kz.expand_as(cx)

            valid_z = (cx >= 0) & (cx < self.res) & \
                      (cy >= 0) & (cy < self.res) & \
                      (cz >= 0) & (cz < self.res)

            cubes_z = torch.stack([cx, cy, cz], dim=-1)     # [Ez,4,3]
            cubes_z_flat = cubes_z.view(-1, 3)
            cubes_z_idx = self.ravel_ijk(cubes_z_flat, (self.res, self.res, self.res))
            cubes_z_idx = cubes_z_idx.view(-1, 4)

            cubes_z_idx = torch.where(
                valid_z,
                cubes_z_idx,
                torch.full_like(cubes_z_idx, -1),
            )

            cubes_out[idx_z] = cubes_z_idx.to(self.index_dtype)

        return cubes_out

    def primal_edge_incident_local_faces(self, edge_idx: torch.Tensor) -> torch.Tensor:
        """
        输入:  edge_idx (E,)
        输出: (E, 4, 3):
            [voxel_idx, local_face0, local_face1]
        """
        dev = self.device
        itype = self.index_dtype

        E = edge_idx.numel()
        out = torch.full((E, 4, 3), -1, dtype=itype, device=dev)

        # (E,4) 每条 edge 的 voxel
        cubes = self.primal_edge_incident_cubes(edge_idx)  # 已经有的函数

        valid = cubes >= 0
        if valid.sum() == 0:
            return out

        # 展开有效项
        edge_ids = torch.arange(E, device=dev, dtype=itype).unsqueeze(1).expand(E,4)
        flat_eid  = edge_ids[valid]        # (M,)
        flat_cid  = cubes[valid]           # (M,)
        M = flat_eid.numel()

        # voxel 内 12 条 edge 全局 idx
        cube_edges = self.primal_cube_edge_indices(flat_cid)   # (M,12)

        # 找局部边 id
        mask = (cube_edges == edge_idx[flat_eid].unsqueeze(1))  # (M,12)
        local_e = mask.argmax(dim=1)  # (M,)

        # 查 2 个局部 face
        lf0, lf1 = self._edge_to_local_faces[local_e].unbind(dim=1)

        # 写回 out
        pos = 0
        for eid, cid, a, b in zip(flat_eid, flat_cid, lf0, lf1):
            # 找到 eid 这一行的第一个空位
            row = out[eid]
            free = (row[:,0] == -1).nonzero(as_tuple=False)
            if free.numel() == 0:
                continue
            slot = free[0,0].item()
            out[eid, slot, 0] = cid     # voxel index
            out[eid, slot, 1] = a       # local face0
            out[eid, slot, 2] = b       # local face1

        return out


    # ---------- 全量索引 ----------
    def all_primal_cube_indices(self) -> torch.Tensor:
        """[0 .. res^3-1]"""
        return torch.arange(self.res**3, device=self.device, dtype=self.index_dtype)

    def all_dual_cube_indices(self) -> torch.Tensor:
        """[0 .. p_num^3-1]（对偶立方体以原顶点网格索引表示）"""
        return torch.arange(self.p_num**3, device=self.device, dtype=self.index_dtype)

    # ---------- 基础 ravel/unravel ----------
    def ravel_ijk(self, ijk: torch.Tensor, dims: Tuple[int, int, int]) -> torch.Tensor:
        nx, ny, nz = dims
        # 统一 device + dtype
        ijk = ijk.to(device=self.device, dtype=self.index_dtype)  # (...,3)
        mul = torch.tensor([ny * nz, nz, 1],
                           device=self.device,
                           dtype=self.index_dtype)                # (3,)
        return (ijk * mul).sum(dim=-1)    


    def unravel_idx(self, idx: torch.Tensor, dims: Tuple[int, int, int]) -> torch.Tensor:
        nx, ny, nz = dims
        # 统一 device + dtype
        idx = idx.to(device=self.device, dtype=self.index_dtype)

        i = torch.div(idx, ny * nz, rounding_mode='floor')
        rem = idx - i * (ny * nz)
        j = torch.div(rem, nz,      rounding_mode='floor')
        k = rem - j * nz
        return torch.stack([i, j, k], dim=-1)  

    # ---------- 原始立方体 ----------
    def primal_cube_indices_ijk(self, cube_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """线性立方体索引 -> (i,j,k)。不传则返回全部."""
        if cube_idx is None:
            cube_idx = self.all_primal_cube_indices()
        # 保证在 self.device 上
        cube_idx = cube_idx.to(self.device)
        dims = (self.res, self.res, self.res)
        return self.unravel_idx(cube_idx, dims) 

    def primal_cube_corner_vertex_indices(self, cube_idx: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        原立方体 -> 8 原顶点线性索引，形状 (B,8)；不传则 B=res^3。
        注意：全量时内存很大（~ 8 * res^3 * sizeof(int)）
        """
        if cube_idx is None:
            cube_idx = self.all_primal_cube_indices()
        ijk = self.primal_cube_indices_ijk(cube_idx.to(self.device)).to(self.index_dtype)   # (B,3)
        corners = ijk[:, None, :] + self._corner_offsets[None, :, :]           # (B,8,3)
        vidx = (corners * self._p_mul[None, None, :]).sum(dim=-1)              # (B,8)
        return vidx

    def primal_vertex_coords_from_indices(self, v_idx: torch.Tensor) -> torch.Tensor:
        """原顶点线性索引 -> 坐标 (B,3)"""
        ijk = self.unravel_idx(v_idx, (self.p_num, self.p_num, self.p_num)).to(torch.long)
        x = self.p_coords_1d.index_select(0, ijk[..., 0].flatten()).view(ijk.shape[:-1])
        y = self.p_coords_1d.index_select(0, ijk[..., 1].flatten()).view(ijk.shape[:-1])
        z = self.p_coords_1d.index_select(0, ijk[..., 2].flatten()).view(ijk.shape[:-1])
        return torch.stack([x, y, z], dim=-1)

    def primal_cube_aabbs_minmax(self, cube_idx: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        直接公式算 AABB（min + [0,0,0], max = min + h）。
        不传则对全部立方体返回 (res^3,3)。
        """
        if cube_idx is None:
            cube_idx = self.all_primal_cube_indices()
        ijk = self.primal_cube_indices_ijk(cube_idx).to(torch.long)
        xmin = self.p_coords_1d.index_select(0, ijk[..., 0].flatten()).view(ijk.shape[:-1])
        ymin = self.p_coords_1d.index_select(0, ijk[..., 1].flatten()).view(ijk.shape[:-1])
        zmin = self.p_coords_1d.index_select(0, ijk[..., 2].flatten()).view(ijk.shape[:-1])
        mins = torch.stack([xmin, ymin, zmin], dim=-1)
        maxs = mins + self.h
        return mins.to(self.dtype), maxs.to(self.dtype)

    def primal_cube_aabbs_centers(self, cube_idx: Optional[torch.Tensor]=None) -> torch.Tensor:
        """返回原始立方体中心点坐标。"""
        mins, maxs = self.primal_cube_aabbs_minmax(cube_idx)
        return (mins + maxs) * 0.5

    def primal_cubes_to_dual_cubes_indices(self, cube_idx: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        原立方体 -> 8 对偶立方体（以原顶点网格线性索引表示）。不传则返回全部 (res^3,8)
        """
        return self.primal_cube_corner_vertex_indices(cube_idx)

    # ---------- 对偶立方体 & 外扩对偶顶点 ----------
    def dual_cubes_to_dual_vertices_ext_indices(self, dual_cube_idx: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        对偶立方体（[0..p_num^3)）-> 8 外扩对偶顶点线性索引（[0..d_ext_num^3)）。不传则全量 (p_num^3,8)
        """
        if dual_cube_idx is None:
            dual_cube_idx = self.all_dual_cube_indices()
        ijk = self.unravel_idx(dual_cube_idx, (self.p_num, self.p_num, self.p_num)).to(self.index_dtype)
        base = ijk[:, None, :] + self._cell_offsets_dual[None, :, :] + 1       # (B,8,3)
        vidx = (base * self._d_ext_mul[None, None, :]).sum(dim=-1)             # (B,8)
        return vidx

    def dual_vertices_ext_coords_from_indices(self, v_idx: torch.Tensor) -> torch.Tensor:
        """外扩对偶顶点线性索引 -> 坐标 (B,3)"""
        ijk = self.unravel_idx(v_idx, (self.d_ext_num, self.d_ext_num, self.d_ext_num)).to(torch.long)
        x = self.d_ext_coords_1d.index_select(0, ijk[..., 0].flatten()).view(ijk.shape[:-1])
        y = self.d_ext_coords_1d.index_select(0, ijk[..., 1].flatten()).view(ijk.shape[:-1])
        z = self.d_ext_coords_1d.index_select(0, ijk[..., 2].flatten()).view(ijk.shape[:-1])
        return torch.stack([x, y, z], dim=-1)


    def get_primal_face_dual_vertex_pairs(self, primal_cube_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        For each primal cube, finds the pair of extended dual vertex indices that
        represent the centers of the cube and its neighbor across each of the 6 faces.

        The order of the 6 faces/neighbors corresponds to the order in CUBE_FACES:
        -Z, +Z, -X, +X, -Y, +Y.
        
        Thanks to the extended dual grid design (d_ext_num = res + 2), every
        neighboring dual vertex has a valid index.

        Args:
            primal_cube_idx (Optional[torch.Tensor]): A 1D tensor of primal cube
                linear indices. If None, computes for all primal cubes.

        Returns:
            torch.Tensor: A tensor of shape (B, 6, 2) where B is the number
                of input cubes. Each pair contains [center_dual_vertex_idx,
                neighbor_center_dual_vertex_idx].
        """
        if primal_cube_idx is None:
            primal_cube_idx = self.all_primal_cube_indices()

        if not isinstance(primal_cube_idx, torch.Tensor) or primal_cube_idx.dim() != 1:
            raise TypeError("Input must be a 1D torch.Tensor of primal cube indices.")

        B = primal_cube_idx.numel()
        if B == 0:
            return torch.empty(0, 6, 2, dtype=self.index_dtype, device=self.device)

        # 1. Get the ijk of the input primal cubes.
        base_primal_ijks = self.primal_cube_indices_ijk(primal_cube_idx)

        # 2. Find the ijk of the extended dual vertex at the center of each primal cube.
        #    The +1 offset maps from the primal cube grid to the extended dual vertex grid.
        center_dual_v_ijks = base_primal_ijks + 1

        # 3. Find the ijk of the 6 neighboring primal cubes.
        neighbor_primal_ijks = base_primal_ijks.unsqueeze(1) + self._face_neighbor_offsets.unsqueeze(0)
        
        # 4. Find the ijk of the extended dual vertices at the center of the neighbors.
        neighbor_dual_v_ijks = neighbor_primal_ijks + 1

        # 5. Convert all dual vertex ijk coordinates to linear indices.
        #    No boundary check is needed due to the extended grid's halo layer.
        dims = (self.d_ext_num, self.d_ext_num, self.d_ext_num)
        
        center_dual_v_indices = self.ravel_ijk(center_dual_v_ijks, dims)
        
        flat_neighbor_dual_v_ijks = neighbor_dual_v_ijks.view(-1, 3)
        flat_neighbor_indices = self.ravel_ijk(flat_neighbor_dual_v_ijks, dims)
        neighbor_dual_v_indices = flat_neighbor_indices.view(B, 6)
        
        # 6. Create the final pairs.
        center_indices_expanded = center_dual_v_indices.unsqueeze(1).expand(-1, 6)
        pairs = torch.stack([center_indices_expanded, neighbor_dual_v_indices], dim=-1)

        return pairs

    def dual_cube_aabbs_minmax(self, dual_cube_idx: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对偶立方体（以原顶点网格线性索引表示） → 其 AABB 的 (min, max)。
        """
        if dual_cube_idx is None:
            dual_cube_idx = self.all_dual_cube_indices()
        centers = self.primal_vertex_coords_from_indices(dual_cube_idx)
        half = self.h * 0.5
        return centers - half, centers + half

    def dual_cube_aabbs_centers(self, dual_cube_idx: Optional[torch.Tensor]=None) -> torch.Tensor:
        """返回对偶立方体中心点坐标 (等同于其对应的原顶点坐标)。"""
        if dual_cube_idx is None:
            dual_cube_idx = self.all_dual_cube_indices()
        return self.primal_vertex_coords_from_indices(dual_cube_idx)

    # ---------- 分块迭代 ----------
    def iter_primal_all(self, chunk: int) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        迭代全量原立方体，按块产出：
          yield (cube_idx_chunk, Fp_chunk(·,8), aabb_min_chunk(·,3), aabb_max_chunk(·,3))
        """
        all_idx = self.all_primal_cube_indices()
        for s in range(0, all_idx.numel(), chunk):
            idx = all_idx[s:s+chunk]
            Fp = self.primal_cube_corner_vertex_indices(idx)
            a_min, a_max = self.primal_cube_aabbs_minmax(idx)
            yield idx, Fp, a_min, a_max

    def iter_dual_all(self, chunk: int) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        迭代全量对偶立方体，按块产出：
        yield (dual_idx_chunk, Fd_chunk(·,8), aabb_min_chunk(·,3), aabb_max_chunk(·,3))
        """
        all_idx = self.all_dual_cube_indices()
        for s in range(0, all_idx.numel(), chunk):
            idx = all_idx[s:s+chunk]
            Fd  = self.dual_cubes_to_dual_vertices_ext_indices(idx)
            a_min, a_max = self.dual_cube_aabbs_minmax(idx)
            yield idx, Fd, a_min, a_max

    # ---------- 新增方法：获取 Primal Edge 对应的 Dual Face ----------
    
    def get_flux_dualvertsidx_from_primal(self, primal_cube_idx: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (向量化版) 为一批原始立方体，计算其12条边对应的12个对偶“挡板面”。

        参数:
          primal_cube_idx (Tensor): 形状为 (B,) 的一维张量，包含 B 个原始立方体的线性索引。

        返回:
          - dual_face_vertices (Tensor[B, 12, 4]): B个立方体，每个12个面，每个面4个顶点的线性索引。
          - dual_cube_pairs (Tensor[B, 12, 2]): B个立方体，每个12个面所分隔的对偶立方体的局部索引对(0-7)。
        """
        if not isinstance(primal_cube_idx, torch.Tensor) or primal_cube_idx.dim() != 1:
            raise TypeError("Input must be a 1D torch.Tensor of primal cube indices.")

        B = primal_cube_idx.numel()
        if B == 0:
            return torch.empty(0, 12, 4, dtype=self.index_dtype, device=self.device), \
                   torch.empty(0, 12, 2, dtype=torch.long, device=self.device)
        

        # 1. 获取所有 B 个立方体的 "基准" ijk 坐标 (B, 3)
        base_primal_ijks = self.primal_cube_indices_ijk(primal_cube_idx)

        # 2. 计算所有边的起始点 ijk 坐标 (广播: (B, 1, 3) + (1, 12, 3) -> (B, 12, 3))
        start_p_ijks = base_primal_ijks.unsqueeze(1) + self._edge_start_corner_offsets.unsqueeze(0)

        # 3. 计算所有对偶面顶点的 ijk 坐标 (广播: (B, 12, 1, 3) + (1, 12, 4, 3) -> (B, 12, 4, 3))
        dual_v_ijks = start_p_ijks.unsqueeze(2) + self._d_ext_v_offsets_all_edges.unsqueeze(0)

        # 4. 将所有 ijk 坐标大批量转换为线性索引
        dims = (self.d_ext_num, self.d_ext_num, self.d_ext_num)
        flat_dual_v_ijks = dual_v_ijks.view(-1, 3)
        flat_linear_indices = self.ravel_ijk(flat_dual_v_ijks, dims)
        dual_face_vertices = flat_linear_indices.view(B, 12, 4)

        # 5. 获取每个 primal cube 周围8个 dual cubes 的【绝对索引】
        #    形状: (B, 8)
        all_surrounding_dual_indices = self.primal_cubes_to_dual_cubes_indices(primal_cube_idx)
        
        # 6. 使用 CUBE_EDGES 的局部索引 (0-7) 来从上面的绝对索引中挑选出正确的配对
        #    CUBE_EDGES[:, 0] 是12条边的起始点局部索引
        #    CUBE_EDGES[:, 1] 是12条边的结束点局部索引
        
        # 获取所有边的起始点的绝对索引，形状: (B, 12)
        v0_global = all_surrounding_dual_indices[:, self.CUBE_EDGES[:, 0]]
        
        # 获取所有边的结束点的绝对索引，形状: (B, 12)
        v1_global = all_surrounding_dual_indices[:, self.CUBE_EDGES[:, 1]]
        
        # 将它们堆叠起来，形成最终的配对，形状: (B, 12, 2)
        absolute_dual_pairs = torch.stack([v0_global, v1_global], dim=2)
        # ==================== 修改部分结束 ====================

        return dual_face_vertices, absolute_dual_pairs
    

    def primal_level_size(self, level: int) -> int:
        """某层的每轴单元数 = 2^level"""
        return 1 << level

    def primal_level_cells(self, level: int) -> int:
        """某层总单元数 = (2^level)^3"""
        s = 1 << level
        return s * s * s
    
    def primal_cube_aabbs_centers(self, cube_idx: Optional[torch.Tensor]=None) -> torch.Tensor:
        if cube_idx is None:
            cube_idx = torch.arange(self.primal_num, device=self.device, dtype=self.index_dtype)
        mins, maxs = self.primal_cube_aabbs_minmax(cube_idx)
        return (mins + maxs) * 0.5
    
    def dual_cube_aabbs_centers(self, dual_cube_idx: Optional[torch.Tensor]=None) -> torch.Tensor:
        if dual_cube_idx is None:
            dual_cube_idx = torch.arange(self.dual_num, device=self.device, dtype=self.index_dtype)
        mins, maxs = self.dual_cube_aabbs_minmax(dual_cube_idx)
        return (mins + maxs) * 0.5

    def get_semiaxis_dualverts_pairs_from_primal(self, primal_cube_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        For each primal cube, finds the pair of extended dual vertex indices that
        represent the centers of the cube and its neighbor across each of the 6 faces.

        The order of the 6 faces/neighbors corresponds to the order in CUBE_FACES:
        -Z, +Z, -X, +X, -Y, +Y.
        
        Thanks to the extended dual grid design (d_ext_num = res + 2), every
        neighboring dual vertex has a valid index.

        Args:
            primal_cube_idx (Optional[torch.Tensor]): A 1D tensor of primal cube
                linear indices. If None, computes for all primal cubes.

        Returns:
            torch.Tensor: A tensor of shape (B, 6, 2) where B is the number
                of input cubes. Each pair contains [center_dual_vertex_idx,
                neighbor_center_dual_vertex_idx].
        """
        if primal_cube_idx is None:
            primal_cube_idx = self.all_primal_cube_indices()

        if not isinstance(primal_cube_idx, torch.Tensor) or primal_cube_idx.dim() != 1:
            raise TypeError("Input must be a 1D torch.Tensor of primal cube indices.")

        B = primal_cube_idx.numel()
        if B == 0:
            return torch.empty(0, 6, 2, dtype=self.index_dtype, device=self.device)

        # 1. Get the ijk of the input primal cubes.
        base_primal_ijks = self.primal_cube_indices_ijk(primal_cube_idx)

        # 2. Find the ijk of the extended dual vertex at the center of each primal cube.
        #    The +1 offset maps from the primal cube grid to the extended dual vertex grid.
        center_dual_v_ijks = base_primal_ijks + 1

        # 3. Find the ijk of the 6 neighboring primal cubes.
        neighbor_primal_ijks = base_primal_ijks.unsqueeze(1) + self._face_neighbor_offsets.unsqueeze(0)
        
        # 4. Find the ijk of the extended dual vertices at the center of the neighbors.
        neighbor_dual_v_ijks = neighbor_primal_ijks + 1

        # 5. Convert all dual vertex ijk coordinates to linear indices.
        #    No boundary check is needed due to the extended grid's halo layer.
        dims = (self.d_ext_num, self.d_ext_num, self.d_ext_num)
        
        center_dual_v_indices = self.ravel_ijk(center_dual_v_ijks, dims)
        
        flat_neighbor_dual_v_ijks = neighbor_dual_v_ijks.view(-1, 3)
        flat_neighbor_indices = self.ravel_ijk(flat_neighbor_dual_v_ijks, dims)
        neighbor_dual_v_indices = flat_neighbor_indices.view(B, 6)
        
        # 6. Create the final pairs.
        center_indices_expanded = center_dual_v_indices.unsqueeze(1).expand(-1, 6)
        pairs = torch.stack([center_indices_expanded, neighbor_dual_v_indices], dim=-1)

        return pairs
    
    
# ===================================================================
#  Octree Indexer with Maximum Level
# ===================================================================

class OctreeIndexer(GridExtent):
    """
    八叉树工具类 (最终修正版)
    修正了 Morton 编码/解码中的位运算bug，确保了转换的可逆性。
    """
    def __init__(self, max_level: int, device: str = "cpu"):
        if not (0 < max_level <= 21): raise ValueError(f"max_level must be between 1 and 21")
        self.max_level = max_level
        super().__init__(res=(1 << max_level), device=device)

    # ====================================================================
    # --- Morton encode/decode ----
    # ====================================================================
    @staticmethod
    def _part1by2(x: torch.Tensor) -> torch.Tensor:
        """Spreads the bits of a 21-bit integer so that there are two zero bits between each original bit."""
        x = x & 0x1fffff
        x = (x | x << 32) & 0x1f00000000ffff
        x = (x | x << 16) & 0x1f0000ff0000ff
        x = (x | x << 8)  & 0x100f00f00f00f00f
        x = (x | x << 4)  & 0x10c30c30c30c30c3
        x = (x | x << 2)  & 0x1249249249249249
        return x

    @staticmethod
    def _compact1by2(x: torch.Tensor) -> torch.Tensor:
        """The inverse of _part1by2, compacting every third bit."""
        x = x & 0x1249249249249249
        x = (x ^ x >> 2)  & 0x10c30c30c30c30c3
        x = (x ^ x >> 4)  & 0x100f00f00f00f00f
        x = (x ^ x >> 8)  & 0x1f0000ff0000ff
        x = (x ^ x >> 16) & 0x1f00000000ffff
        x = (x ^ x >> 32) & 0x1fffff
        return x

    @staticmethod
    def morton_encode(ijk: torch.Tensor) -> torch.Tensor:
        """将 [..., 3] 形式的 (i,j,k) 坐标编码为 Morton 码。"""
        i, j, k = ijk[..., 0].to(torch.int64), ijk[..., 1].to(torch.int64), ijk[..., 2].to(torch.int64)
        return (OctreeIndexer._part1by2(i) | 
                (OctreeIndexer._part1by2(j) << 1) | 
                (OctreeIndexer._part1by2(k) << 2))

    @staticmethod
    def morton_decode(code: torch.Tensor) -> torch.Tensor:
        """将 Morton 码解码为 [..., 3] 形式的 (i,j,k) 坐标。"""
        code = code.to(torch.int64)
        i = OctreeIndexer._compact1by2(code)
        j = OctreeIndexer._compact1by2(code >> 1)
        k = OctreeIndexer._compact1by2(code >> 2)
        return torch.stack([i, j, k], dim=-1)

    # --- 等级感知方法 (无变化) ---
    def primal_cube_indices_ijk(self, cube_idx: Optional[torch.Tensor] = None, level: Optional[int] = None) -> torch.Tensor:
        if level is None: level = self.max_level
        n_level = 1 << level
        if cube_idx is None: cube_idx = torch.arange(n_level**3, device=self.device, dtype=self.index_dtype)
        return self.unravel_idx(cube_idx, dims=(n_level, n_level, n_level))

    def primal_cube_aabbs_minmax(self, cube_idx: Optional[torch.Tensor] = None, level: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if level is None: level = self.max_level
        n_level = 1 << level; h_level = 2.0 / n_level
        ijk = self.primal_cube_indices_ijk(cube_idx, level=level)
        mins = -1.0 + ijk.to(self.dtype) * h_level
        return mins, mins + h_level

    # --- 其他 Octree 专用方法 (无变化) ---
    def linear_to_morton(self, linear_idx: torch.Tensor, level: int) -> torch.Tensor:
        n = 1 << level
        ijk = self.unravel_idx(linear_idx, dims=(n, n, n))
        return self.morton_encode(ijk)

    def morton_to_linear(self, morton: torch.Tensor, level: int) -> torch.Tensor:
        ijk = self.morton_decode(morton)
        n = 1 << level
        return ijk[..., 0] * n*n + ijk[..., 1] * n + ijk[..., 2]

    def children_morton(self, parent_morton: torch.Tensor) -> torch.Tensor:
        base = parent_morton.unsqueeze(-1).to(torch.int64) << 3
        return base + torch.arange(8, device=self.device, dtype=torch.int64)
    
if __name__ == '__main__':
    # 1. 通过 level 初始化，更直观
    # max_level=7 意味着网格分辨率 res = 2**7 = 128
    indexer = OctreeIndexer(max_level=7, device='cuda')

    a = indexer.primal_cube_aabbs_minmax(level=4)

    indexer = OctreeIndexer(max_level=4, device='cuda')
    
    b = indexer.primal_cube_aabbs_minmax(level=4)

    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1]), "Mismatch in AABB computation across levels"

    print(a[0], a[1])
    assert 1==2, "stopdasda"

    print(f"Octree Initialized:")
    print(f"  Max Level: {indexer.max_level}")
    print(f"  Resolution (res): {indexer.res}")
    print(f"  Device: {indexer.device}")
    
    # 2. 自动拥有 GridExtent 的所有方法
    print("\nTesting inherited GridExtent method...")
    # 获取索引为 0, 1, 2 的 primal 立方体的包围盒
    some_primal_indices = torch.tensor([0, 1, 2], device=indexer.device)
    mins, maxs = indexer.primal_cube_aabbs_minmax(some_primal_indices)
    print(f"AABBs for primals [0, 1, 2]:\n{mins.cpu().numpy()}")

    # 3. 使用 OctreeIndexer 的特定方法
    print("\nTesting OctreeIndexer specific method...")
    # 将这些 primal 索引转换为 level 5 上的 Morton 码
    morton_codes = indexer.morton_at_level(some_primal_indices, level=5)
    print(f"Morton codes at level 5:\n{morton_codes.cpu().numpy()}")
    
    # 4. 使用强大的父子映射功能
    print("\nTesting Parent-Child mapping...")
    # 获取从 level 2 到 level 3 的所有父子关系
    parent_child_map = indexer.get_parent_child_map(parent_level=2)
    print(f"Parent-child map shape (COO format): {parent_child_map.shape}")
    num_nodes_level2 = (1 << 2)**3
    print(f"Expected number of connections: {num_nodes_level2 * 8}")
    print("Example pairs (child_morton, parent_morton):")
    print(parent_child_map[:, :5].cpu().numpy())