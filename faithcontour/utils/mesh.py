import numpy as np
import trimesh

def normalize_mesh(mesh, rescalar=0.99):
    """
    Normalize the mesh to fit in the unit sphere
    Args:
        mesh: Meshes object or trimesh object
        rescalar: float, the scale factor to rescale the mesh
    """

    if isinstance(mesh, trimesh.Trimesh):
        bbox_min, bbox_max = mesh.bounds
        bbox_center = (bbox_min + bbox_max) / 2 
        bbox_size = bbox_max - bbox_min

        # Scale factor to normalize to [-1, 1]
        scale_factor = 2.0 / np.max(bbox_size) * rescalar

        # Apply translation and scaling
        mesh.apply_translation(-bbox_center)  # Move the mesh center to the origin
        mesh.apply_scale(scale_factor)
    else:
        raise ValueError("Unsupported mesh type. Please provide a trimesh.Trimesh object.")

    return mesh


import torch
import einops as eins

QUAD2TRI = np.array([[[0,1,2], [0,2,3]], [[0,1,3], [1,2,3]]])


def rectangle_mesh_angle(face_verts):
    '''
    计算四边形的所有内角
    '''
    v0 = face_verts[..., 0, :]
    v1 = face_verts[..., 1, :]
    v2 = face_verts[..., 2, :]
    v3 = face_verts[..., 3, :]
    e01 = v1 - v0
    e12 = v2 - v1
    e23 = v3 - v2
    e30 = v0 - v3   
    e01 = e01 / (e01.norm(dim=-1, keepdim=True).clamp_min(1e-9))
    e12 = e12 / (e12.norm(dim=-1, keepdim=True).clamp_min(1e-9))
    e23 = e23 / (e23.norm(dim=-1, keepdim=True).clamp_min(1e-9))
    e30 = e30 / (e30.norm(dim=-1, keepdim=True).clamp_min(1e-9))
    angle0 = torch.atan2(torch.linalg.norm(torch.cross(e01, e30, dim=-1), dim=-1), (e01 * e30).sum(dim=-1))
    angle1 = torch.atan2(torch.linalg.norm(torch.cross(e12, e01, dim=-1), dim=-1), (e12 * e01).sum(dim=-1))
    angle2 = torch.atan2(torch.linalg.norm(torch.cross(e23, e12, dim=-1), dim=-1), (e23 * e12).sum(dim=-1))
    angle3 = torch.atan2(torch.linalg.norm(torch.cross(e30, e23, dim=-1), dim=-1), (e30 * e23).sum(dim=-1))

    angles = torch.stack([angle0, angle1, angle2, angle3], dim=-1)
    return angles

def filter_duplicate_faces(faces):
    '''
        Filter out duplicate faces from a list of faces.
        Args:
            faces: (F, d) Tensor of face indices
            output: unique faces
    '''

    faces_sorted = torch.sort(faces, dim=-1).values
    faces_unique_indices = np.unique(faces_sorted.cpu().numpy(), axis=0, return_index=True)[1]
    return faces[faces_unique_indices]

def filter_duplicate_faces_index(faces):
    '''
        Filter out duplicate faces from a list of faces.
        Args:
            faces: (F, d) Tensor of face indices
            output: indices of unique faces
    '''

    faces_sorted = torch.sort(faces, dim=-1).values
    faces_unique_indices = np.unique(faces_sorted.cpu().numpy(), axis=0, return_index=True)[1]
    return faces_unique_indices

import torch.nn.functional as F
# Define the constant for the two triangulation patterns
QUAD2TRI = [
    [[0, 1, 2], [0, 2, 3]],  # Pattern 0: Splits along the (0, 2) diagonal
    [[0, 1, 3], [1, 2, 3]],  # Pattern 1: Splits along the (1, 3) diagonal
]

def rectangle_mesh_angle(faces_quad_coord: torch.Tensor) -> torch.Tensor:
    """
    Calculates the interior angle at each vertex for a batch of quadrilateral faces.
    """
    # Get vectors for the edges meeting at each vertex
    p0, p1, p2, p3 = faces_quad_coord[:, 0], faces_quad_coord[:, 1], faces_quad_coord[:, 2], faces_quad_coord[:, 3]
    v01, v12, v23, v30 = p1 - p0, p2 - p1, p3 - p2, p0 - p3
    
    # Calculate the cosine of the angle using dot products of normalized vectors
    cos_a0 = torch.einsum('bi,bi->b', F.normalize(v30, dim=-1), F.normalize(-v01, dim=-1))
    cos_a1 = torch.einsum('bi,bi->b', F.normalize(v01, dim=-1), F.normalize(-v12, dim=-1))
    cos_a2 = torch.einsum('bi,bi->b', F.normalize(v12, dim=-1), F.normalize(-v23, dim=-1))
    cos_a3 = torch.einsum('bi,bi->b', F.normalize(v23, dim=-1), F.normalize(-v30, dim=-1))

    # Return the angles in radians
    return torch.acos(torch.stack([cos_a0, cos_a1, cos_a2, cos_a3], dim=1).clamp(-1.0, 1.0))

def triangulate_quads_by_angle(all_verts: torch.Tensor, faces_quad_unique: torch.Tensor,
                               vertex_normals: torch.Tensor = None, chunk_size: int = 4096) -> np.ndarray:
    """
    Splits quadrilateral faces into triangular faces based on normal consistency, processing in chunks to prevent OOM.

    If vertex normals are provided, chooses the diagonal split that generates triangles
    with normals most consistent with the vertex normals. Otherwise falls back to
    angle-based splitting for backward compatibility.

    Args:
        all_verts (torch.Tensor): A tensor of shape (N, 3) containing all vertex coordinates.
        faces_quad_unique (torch.Tensor): A tensor of shape (F, 4) containing the vertex
                                           indices for F unique quadrilateral faces.
        vertex_normals (torch.Tensor, optional): A tensor of shape (N, 3) containing vertex normals.
        chunk_size (int): The number of quadrilateral faces to process in each chunk.

    Returns:
        torch.Tensor: A tensor of shape (F * 2, 3) containing the new triangular face indices.
    """
    if isinstance(all_verts, np.ndarray):
        all_verts = torch.from_numpy(all_verts)
    if isinstance(faces_quad_unique, np.ndarray):
        faces_quad_unique = torch.from_numpy(faces_quad_unique)

    device = all_verts.device
    faces_quad_unique = faces_quad_unique.to(device)

    if vertex_normals is not None and isinstance(vertex_normals, np.ndarray):
        vertex_normals = torch.from_numpy(vertex_normals).to(device)

    num_quads = faces_quad_unique.shape[0]
    all_tri_faces = []
    quad2tri_tensor = torch.tensor(QUAD2TRI, device=device, dtype=torch.long)

    for i in range(0, num_quads, chunk_size):
        faces_quad_chunk = faces_quad_unique[i:i + chunk_size]

        if vertex_normals is None:
            # Legacy angle-based implementation
            faces_quad_coords_chunk = all_verts[faces_quad_chunk]
            faces_quad_angle_chunk = rectangle_mesh_angle(faces_quad_coords_chunk)
            faces_quad_angle_02_chunk = faces_quad_angle_chunk[:, [0, 2]].sum(dim=-1)
            faces_quad_angle_13_chunk = faces_quad_angle_chunk[:, [1, 3]].sum(dim=-1)
            condition_chunk = faces_quad_angle_02_chunk < faces_quad_angle_13_chunk
        else:
            # Normal-based implementation
            faces_quad_coords_chunk = all_verts[faces_quad_chunk]  # (F_chunk, 4, 3)
            quad_vert_normals_chunk = vertex_normals[faces_quad_chunk]  # (F_chunk, 4, 3)

            # Generate triangles for both patterns
            tri_coords_02_chunk = faces_quad_coords_chunk[:, quad2tri_tensor[0]]  # (F_chunk, 2, 3, 3)
            tri_coords_13_chunk = faces_quad_coords_chunk[:, quad2tri_tensor[1]]  # (F_chunk, 2, 3, 3)
            tri_normals_02_chunk = quad_vert_normals_chunk[:, quad2tri_tensor[0]]  # (F_chunk, 2, 3, 3)
            tri_normals_13_chunk = quad_vert_normals_chunk[:, quad2tri_tensor[1]]  # (F_chunk, 2, 3, 3)

            # Reshape for computation
            tri_coords_02_flat = tri_coords_02_chunk.view(-1, 3, 3)  # (F_chunk*2, 3, 3)
            tri_coords_13_flat = tri_coords_13_chunk.view(-1, 3, 3)  # (F_chunk*2, 3, 3)
            tri_normals_02_flat = tri_normals_02_chunk.view(-1, 3, 3)  # (F_chunk*2, 3, 3)
            tri_normals_13_flat = tri_normals_13_chunk.view(-1, 3, 3)  # (F_chunk*2, 3, 3)

            # Compute actual triangle normals from geometry
            def compute_triangle_normals(verts):
                v0, v1, v2 = verts[:, 0], verts[:, 1], verts[:, 2]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normals = torch.cross(edge1, edge2, dim=-1)
                return normals / (normals.norm(dim=-1, keepdim=True).clamp_min(1e-9))

            tri_geom_normals_02 = compute_triangle_normals(tri_coords_02_flat)  # (F_chunk*2, 3)
            tri_geom_normals_13 = compute_triangle_normals(tri_coords_13_flat)  # (F_chunk*2, 3)

            # Compute consistency: dot product between geometric normal and vertex normals
            def compute_consistency(geom_normals, vert_normals):
                # Average dot product between triangle normal and each vertex normal
                dot_products = torch.sum(geom_normals.unsqueeze(1) * vert_normals, dim=-1)  # (F_chunk*2, 3)
                return torch.mean(dot_products, dim=-1)  # (F_chunk*2,)

            consistency_02 = compute_consistency(tri_geom_normals_02, tri_normals_02_flat)
            consistency_13 = compute_consistency(tri_geom_normals_13, tri_normals_13_flat)

            # Reshape back to (F_chunk, 2) and average the two triangles for each split
            consistency_02 = consistency_02.view(-1, 2).mean(dim=1)  # (F_chunk,)
            consistency_13 = consistency_13.view(-1, 2).mean(dim=1)  # (F_chunk,)

            # Choose split with better normal consistency (higher dot product = more consistent)
            condition_chunk = consistency_02 > consistency_13

        # Apply the chosen triangulation pattern
        faces_pattern0 = faces_quad_chunk[:, quad2tri_tensor[0]]  # Split along (0,2)
        faces_pattern1 = faces_quad_chunk[:, quad2tri_tensor[1]]  # Split along (1,3)

        faces_quad2tri_torch_chunk = torch.where(condition_chunk[:, None, None], faces_pattern0, faces_pattern1)

        # Rearrange from (F_chunk, 2, 3) to (F_chunk * 2, 3)
        faces_quad2tri_rearranged_chunk = eins.rearrange(faces_quad2tri_torch_chunk, 'b t v -> (b t) v')
        all_tri_faces.append(faces_quad2tri_rearranged_chunk)

    if not all_tri_faces:
        return torch.empty((0, 3), dtype=torch.long)

    return torch.cat(all_tri_faces, dim=0)



def normalize_mesh_max_fill(mesh, margin=0.01):
    pass
