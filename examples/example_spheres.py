"""
Render spheres at positions.
Differentiate w.r.t. sphere origins.
"""
import dataclasses as dc
import os

import torch
import numpy as np
import tqdm
import imageio

import neural_renderer as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


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
        vertices, faces = nr.load_obj(filename, normalization=True)
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


def main():
    preview_image_file = os.path.join(data_dir, 'example_spheres.png')

    device = torch.device("cuda")

    # other settings
    camera_distance = 4.0
    elevation = 0.0
    azimuth = 0.0

    # load template once.
    template_file = os.path.join(data_dir, 'sphere_lowpoly.obj')
    template = Mesh.from_file(template_file)

    # create renderer
    renderer = nr.Renderer(
        camera_mode='look_at',
        image_size=128,
    )
    renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)

    # Fake particle positions.
    assert torch.is_grad_enabled()
    p_WOs = torch.tensor([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        # Out of view; should not have gradient.
        [-100.0, 0.0, 0.0],
    ], requires_grad=True, device=device)
    assert p_WOs.requires_grad

    # Assemble scene.
    scene = Mesh.empty(device)
    for p_WO in p_WOs:
        scene.add_object(template, p_WO)

    vertices, faces = scene.unsqueeze()
    fake_textures = make_fake_textures(faces)

    images = renderer(vertices, faces, textures=fake_textures, mode="rgb")
    image = images.squeeze(0)

    # Fake reduction to check gradients.
    fake_reduction = torch.mean(image)
    fake_reduction.backward()

    # Show gradients of "loss" w.r.t. each position.
    print(p_WOs.grad)

    # Save image.
    image_numpy = image.detach().cpu().numpy().transpose(1, 2, 0)
    image_numpy = (255 * image_numpy).astype(np.uint8)
    imageio.imwrite(preview_image_file, image_numpy)


if __name__ == '__main__':
    main()
