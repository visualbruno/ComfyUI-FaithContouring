import os
import sys
import math
import time
import contextlib
import copy

import torch
import trimesh as Trimesh
import numpy as np
from pathlib import Path

import meshlib.mrmeshnumpy as mrmeshnumpy
import meshlib.mrmeshpy as mrmeshpy

import folder_paths

# FaithContour imports
from .faithcontour import FCTEncoder, FCTDecoder

# Atom3d imports
from atom3d import MeshBVH
from atom3d.grid import OctreeIndexer

def simplify_with_meshlib(vertices, faces, target=1000000):
    current_faces_num = len(faces)
    print(f'Current Faces Number: {current_faces_num}')
    
    if current_faces_num<target:
        return

    settings = mrmeshpy.DecimateSettings()
    faces_to_delete = current_faces_num - target
    settings.maxDeletedFaces = faces_to_delete                        
    settings.packMesh = True
    
    print('Generating Meshlib Mesh ...')
    mesh = mrmeshnumpy.meshFromFacesVerts(faces, vertices)
    print('Packing Optimally ...')
    mesh.packOptimally()
    print('Decimating ...')
    mrmeshpy.decimateMesh(mesh, settings)
    
    new_vertices = mrmeshnumpy.getNumpyVerts(mesh)
    new_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)               
    
    print(f"Reduced faces, resulting in {len(new_vertices)} vertices and {len(new_faces)} faces")
        
    return new_vertices, new_faces

@contextlib.contextmanager
def SuppressPrint(turn_stdout: bool = True):
    """
    A context manager to temporarily suppress print statements.

    Usage:
        with SuppressPrint():
            noisy_function()
    """
    if not turn_stdout:
        yield
        return
    original_stdout = sys.stdout
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = original_stdout

def normalize_mesh(mesh: Trimesh.Trimesh, margin: float = 0.05) -> Trimesh.Trimesh:
    """
    Normalize mesh to fit within [-1+margin, 1-margin]^3 centered at origin.
    
    Args:
        mesh: Input trimesh
        margin: Margin from grid boundary (0.0-1.0), e.g., 0.05 means 5% margin on each side
    
    Returns:
        Normalized mesh
    """
    # Get current bounding box
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    
    # Center the mesh at the bounding box center (not centroid)
    bbox_center = (bbox_min + bbox_max) / 2.0
    mesh.vertices -= bbox_center
    
    # Scale to fit in [-1+margin, 1-margin]^3
    # Target half-size: 1.0 - margin
    target_half_size = 1.0 - margin
    current_half_size = np.abs(mesh.vertices).max()
    
    if current_half_size > 1e-8:
        scale = target_half_size / current_half_size
        mesh.vertices *= scale
    
    return mesh

class FaithCLoadMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "tooltip": "The glb path with mesh to load."}), 
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    
    FUNCTION = "load"
    CATEGORY = "FaithCWrapper"
    DESCRIPTION = "Loads a glb model from the given path."

    def load(self, glb_path):
        if not os.path.exists(glb_path):
            glb_path = os.path.join(folder_paths.get_input_directory(), glb_path)
        
        trimesh = Trimesh.load(glb_path, force="mesh")
        
        return (trimesh,)
        
class FaithCNormalizeMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "margin": ("FLOAT",{"default":0.05,"min":0.00,"max":99.99,"step":0.01}),
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    
    FUNCTION = "process"
    CATEGORY = "FaithCWrapper"

    def process(self, trimesh, margin):
        mesh_copy = copy.deepcopy(trimesh)
        
        mesh_copy = normalize_mesh(mesh_copy, margin)
        
        return (mesh_copy,)  

class FaithCPreProcessMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    
    FUNCTION = "process"
    CATEGORY = "FaithCWrapper"

    def process(self, trimesh):
        mesh_copy = copy.deepcopy(trimesh)
        
        orig_faces = len(mesh_copy.faces)
        valid_mask = mesh_copy.nondegenerate_faces()
        if valid_mask.sum() < orig_faces:
            mesh_copy.update_faces(valid_mask)
            mesh_copy.remove_unreferenced_vertices()
            print(f"   ⚠️  Removed {orig_faces - len(mesh_copy.faces)} degenerate faces")
        
        return (mesh_copy,)     

class FaithCEncodeMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "resolution": ([64,128,256,512,1024,2048],{"default":512}),
                "device": (["cpu","cuda"],{"default":"cuda"}),
                "verbose": ("BOOLEAN",{"default":False}),
                "compute_flux": ("BOOLEAN",{"default":True}),
                "clamp_anchors": ("BOOLEAN",{"default":True}),
            }
        }
    RETURN_TYPES = ("INT","GRID_BOUNDS","FCT_RESULT","STRING")
    RETURN_NAMES = ("resolution","grid_bounds","fct_result","device")
    
    FUNCTION = "process"
    CATEGORY = "FaithCWrapper"

    def process(self, trimesh, resolution, device, verbose, compute_flux, clamp_anchors):
        max_level = int(math.log2(resolution))
        min_level = min(4, max(1, max_level - 1))
        
        vertices = torch.tensor(trimesh.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(trimesh.faces, dtype=torch.long, device=device)
        
        print(f"\n🔧 Building spatial structures...")
        bvh = MeshBVH(vertices, faces)
        
        grid_bounds = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=device)
        octree = OctreeIndexer(max_level=max_level, bounds=grid_bounds, device=device)
        print(f"   • OctreeIndexer initialized (res={octree.res}, bounds=[-1, 1]^3)")        
        
        # --- FCT Encoding ---
        print(f"\n🔄 FCT Encoding...")        
        encoder = FCTEncoder(bvh, octree, device=device)        
        
        time_start = time.time()
        with SuppressPrint(turn_stdout=not verbose):
            solver_weights = {
                'lambda_n': 1.0,
                'lambda_d': 1e-3,
                'weight_power': 1
            }
            fct_result = encoder.encode(
                min_level=min_level,
                solver_weights=solver_weights,
                compute_flux=compute_flux,
                clamp_anchors=clamp_anchors
            )
        time_encode = time.time() - time_start        
        
        print(f"   ✅ Encoding completed in {time_encode:.3f}s")
        print(f"   • Active voxels: {fct_result.active_voxel_indices.shape[0]}")
        print(f"   • Anchor shape: {fct_result.anchor.shape}")
        print(f"   • Normal shape: {fct_result.normal.shape}")
        print(f"   • Edge flux shape: {fct_result.edge_flux_sign.shape}")        
        
        return (resolution, grid_bounds, fct_result, device,)  

class FaithCDecodeMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resolution": ("INT",),
                "grid_bounds": ("GRID_BOUNDS",),
                "fct_result": ("FCT_RESULT",),
                "device": ("STRING",),
                "tri_mode": (['auto', 'simple_02', 'simple_13', 'length', 'angle', 'normal', 'normal_abs'],{"default":"auto"}),
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    
    FUNCTION = "process"
    CATEGORY = "FaithCWrapper"

    def process(self, resolution, grid_bounds, fct_result, device, tri_mode,):
        print(f"\n🔄 FCT Decoding (tri_mode={tri_mode})...")        
        decoder = FCTDecoder(resolution=resolution, bounds=grid_bounds, device=device)
        
        time_start = time.time()
        decoded_mesh = decoder.decode(
            active_voxel_indices=fct_result.active_voxel_indices,
            anchors=fct_result.anchor,
            edge_flux_sign=fct_result.edge_flux_sign,
            normals=fct_result.normal,
            triangulation_mode=tri_mode
        )
        time_decode = time.time() - time_start    

        print(f"   ✅ Decoding completed in {time_decode:.3f}s")
        print(f"   • Generated vertices: {decoded_mesh.vertices.shape[0]}")
        print(f"   • Generated faces: {decoded_mesh.faces.shape[0]}")        
        
        # Create trimesh for export
        recon_mesh = Trimesh.Trimesh(
            vertices=decoded_mesh.vertices.cpu().numpy(),
            faces=decoded_mesh.faces.cpu().numpy(),
            process=False
        )        
        
        return (recon_mesh,)

class FaithCExportMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "3D/Trellis2"}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            },
            "optional": {
                "save_file": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "process"
    CATEGORY = "FaithCWrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, filename_prefix, file_format, save_file=True):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.{file_format}')
        output_glb_path.parent.mkdir(exist_ok=True)
        if save_file:
            trimesh.export(output_glb_path, file_type=file_format)
            relative_path = Path(subfolder) / f'{filename}_{counter:05}_.{file_format}'
        else:
            temp_file = Path(full_output_folder, f'hy3dtemp_.{file_format}')
            trimesh.export(temp_file, file_type=file_format)
            relative_path = Path(subfolder) / f'hy3dtemp_.{file_format}'
        
        return (str(relative_path), )   

class FaithCSimplifyMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_face_num": ("INT",{"default":2000000,"min":1,"max":20000000}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "FaithCWrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, target_face_num,):
        mesh_copy = copy.deepcopy(trimesh)
        new_vertices, new_faces = simplify_with_meshlib(vertices = mesh_copy.vertices, faces = mesh_copy.faces, target = target_face_num)
        mesh_copy.vertices = new_vertices
        mesh_copy.faces = new_faces
        
        return (mesh_copy, )       

class FaithCProcessMeshWithVoxel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "resolution": ([64,128,256,512,1024,2048],{"default":512}),
                "device": (["cpu","cuda"],{"default":"cuda"}),
                "verbose": ("BOOLEAN",{"default":False}),
                "compute_flux": ("BOOLEAN",{"default":True}),
                "clamp_anchors": ("BOOLEAN",{"default":True}), 
                "tri_mode": (['auto', 'simple_02', 'simple_13', 'length', 'angle', 'normal', 'normal_abs'],{"default":"auto"}),                
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "FaithCWrapper"
    OUTPUT_NODE = True

    def process(self, mesh, resolution, device, verbose, compute_flux, clamp_anchors, tri_mode):
        print('Normalize mesh ...')
        trimesh = Trimesh.Trimesh(mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy())
        trimesh = normalize_mesh(trimesh)
        
        print('Preprocess mesh ...')
        orig_faces = len(trimesh.faces)
        valid_mask = trimesh.nondegenerate_faces()
        if valid_mask.sum() < orig_faces:
            trimesh.update_faces(valid_mask)
            trimesh.remove_unreferenced_vertices()
            print(f"   ⚠️  Removed {orig_faces - len(trimesh.faces)} degenerate faces")        
        
        max_level = int(math.log2(resolution))
        min_level = min(4, max(1, max_level - 1))
        
        vertices = torch.tensor(trimesh.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(trimesh.faces, dtype=torch.long, device=device)
        
        print(f"\n🔧 Building spatial structures...")
        bvh = MeshBVH(vertices, faces)
        
        grid_bounds = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=device)
        octree = OctreeIndexer(max_level=max_level, bounds=grid_bounds, device=device)
        print(f"   • OctreeIndexer initialized (res={octree.res}, bounds=[-1, 1]^3)")        
        
        # --- FCT Encoding ---
        print(f"\n🔄 FCT Encoding...")        
        encoder = FCTEncoder(bvh, octree, device=device)        
        
        time_start = time.time()
        with SuppressPrint(turn_stdout=not verbose):
            solver_weights = {
                'lambda_n': 1.0,
                'lambda_d': 1e-3,
                'weight_power': 1
            }
            fct_result = encoder.encode(
                min_level=min_level,
                solver_weights=solver_weights,
                compute_flux=compute_flux,
                clamp_anchors=clamp_anchors
            )
        time_encode = time.time() - time_start        
        
        print(f"   ✅ Encoding completed in {time_encode:.3f}s")
        print(f"   • Active voxels: {fct_result.active_voxel_indices.shape[0]}")
        print(f"   • Anchor shape: {fct_result.anchor.shape}")
        print(f"   • Normal shape: {fct_result.normal.shape}")
        print(f"   • Edge flux shape: {fct_result.edge_flux_sign.shape}") 
        
        print(f"\n🔄 FCT Decoding (tri_mode={tri_mode})...")        
        decoder = FCTDecoder(resolution=resolution, bounds=grid_bounds, device=device)
        
        time_start = time.time()
        decoded_mesh = decoder.decode(
            active_voxel_indices=fct_result.active_voxel_indices,
            anchors=fct_result.anchor,
            edge_flux_sign=fct_result.edge_flux_sign,
            normals=fct_result.normal,
            triangulation_mode=tri_mode
        )
        time_decode = time.time() - time_start    

        print(f"   ✅ Decoding completed in {time_decode:.3f}s")
        print(f"   • Generated vertices: {decoded_mesh.vertices.shape[0]}")
        print(f"   • Generated faces: {decoded_mesh.faces.shape[0]}")        
        
        mesh_copy = copy.deepcopy(mesh)
        mesh_copy.vertices = decoded_mesh.vertices.float()
        mesh_copy.faces = decoded_mesh.faces.int()
        
        return (mesh_copy, )            

NODE_CLASS_MAPPINGS = {
    "FaithCLoadMesh": FaithCLoadMesh,
    "FaithCNormalizeMesh": FaithCNormalizeMesh,
    "FaithCPreProcessMesh": FaithCPreProcessMesh,
    "FaithCEncodeMesh": FaithCEncodeMesh,
    "FaithCDecodeMesh": FaithCDecodeMesh,
    "FaithCExportMesh": FaithCExportMesh,
    "FaithCSimplifyMesh": FaithCSimplifyMesh,
    "FaithCProcessMeshWithVoxel": FaithCProcessMeshWithVoxel,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaithCLoadMesh": "FaithC - Load Mesh",
    "FaithCNormalizeMesh": "FaithC - Normalize Mesh",
    "FaithCPreProcessMesh": "FaithC - PreProcess Mesh",
    "FaithCEncodeMesh": "FaithC - Encode Mesh",
    "FaithCDecodeMesh": "FaithC - Decode Mesh",
    "FaithCExportMesh": "FaithC - Export Mesh",
    "FaithCSimplifyMesh": "FaithC - Simplify Mesh",
    "FaithCProcessMeshWithVoxel": "FaithC - Process Mesh with Voxel",
    }        