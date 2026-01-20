"""
FaithContour: Faithful Contour Token Library

A library for encoding meshes into compact, topology-preserving tokens
and decoding them back to meshes.

Built on Atom3d primitives for efficient GPU-accelerated operations.

Main Functions:
    FCT_encoder: Mesh → FCT tokens
    FCT_decoder: FCT tokens → Mesh
    
Classes:
    FCTEncoder: Class-based encoder
    FCTDecoder: Class-based decoder
    UDFEncoder: UDF-based encoder (experimental)

Utilities:
    normalize_mesh: Normalize mesh to unit cube
"""

try:
    import torch
    assert torch.cuda.is_available(), "CUDA is not available."
except ImportError:
    raise ImportError(
        "\n\nPyTorch is not installed. \n"
        "Please install it manually from https://pytorch.org according to your CUDA version \n"
        "before installing or using the 'faithcontour' library.\n"
    )

# Class-based interface
from .encoder import FCTEncoder, FCTResult
from .decoder import FCTDecoder, DecodedMesh

# Utilities
from .utils.mesh import normalize_mesh, triangulate_quads_by_angle
from .qef_solver import solve_qef, solve_qef_differentiable

__all__ = [
    # Classes
    'FCTEncoder',
    'FCTDecoder',
    'FCTResult',
    'DecodedMesh',
    # Utilities
    'normalize_mesh',
    'triangulate_quads_by_angle',
    'solve_qef',
    'solve_qef_differentiable',
]

__version__ = '1.5.0'
