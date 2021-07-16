import dataclasses as dc

import numpy as np
import torch

import neural_renderer as nr


@dc.dataclass
class Mesh:
    vertices: torch.Tensor  # [num_vertices, 3], float32
    faces: torch.Tensor  # [num_faces, 3], int32

    def __post_init__(self):
        assert self.vertices.dim() == 2
        assert self.vertices.shape[1] == 3
        assert self.vertices.dtype == torch.float32
        assert self.faces.dim() == 2
        assert self.faces.shape[1] == 3
        assert self.faces.dtype == torch.int32

    @classmethod
    def from_file(cls, filename):
        vertices, faces = nr.load_obj(filename, normalization=False)
        return Mesh(
            vertices=vertices,
            faces=faces,
        )

    @classmethod
    def empty(self, device):
        return Mesh(
            vertices=torch.zeros((0, 3), dtype=torch.float32, device=device),
            faces=torch.zeros((0, 3), dtype=torch.int32, device=device),
        )

    def add_object(self, obj, p_WO):
        assert isinstance(obj, Mesh)
        # Transform vertices (with gradient).
        new_vertices = p_WO.unsqueeze(0) + obj.vertices
        # Offset face vertex indices (no gradient).
        prev_num_vertices = len(self.vertices)
        with torch.no_grad():
            new_faces = obj.faces + prev_num_vertices
        # Now concatenate.
        self.vertices = torch.cat([self.vertices, new_vertices])
        self.faces = torch.cat([self.faces, new_faces])

    def unsqueeze(self):
        # Unsqueeze to add batch dimension.
        return self.vertices.unsqueeze(0), self.faces.unsqueeze(0)


def make_fake_textures(faces):
    batch_size, num_faces, vertices_per_face = faces.shape
    assert vertices_per_face == 3
    num_channels = 3
    small_size = 2
    return torch.ones(
        (batch_size, num_faces, small_size, small_size, small_size, num_channels),
        dtype=torch.float32,
        device=faces.device,
    )


def intrinsic_matrix_from_fov(width, height, fov_y):
    # https://github.com/RobotLocomotion/drake/blob/v0.32.0/systems/sensors/camera_info.cc
    focal_x = height * 0.5 / np.tan(0.5 * fov_y)
    focal_y = focal_x
    pp_x = width / 2 - 0.5  # OpenGL half-pixel
    pp_y = height / 2 - 0.5
    return np.array([
        [focal_x, 0, pp_x],
        [0, focal_y, pp_y],
        [0, 0, 1],
    ])
