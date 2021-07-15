"""
Render spheres at positions.
Differentiate w.r.t. sphere origins.
"""
import dataclasses as dc
import os

import torch
import numpy as np
from skimage.io import imread, imsave
import h5py
import tqdm
import imageio
import glob

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
        # new_vertices = torch.mul(p_WO.unsqueeze(0),0.1) + obj.vertices
        new_vertices = p_WO.unsqueeze(0) + torch.mul(obj.vertices, 0.1)
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

def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()


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


def main():
    preview_image_file = os.path.join(data_dir, 'example_spheres.png')
    # image_ref_file = os.path.join(data_dir, "example_dough_ref.png")
    # # image_ref_file = os.path.join(data_dir, "example2_ref.png")
    # img = imread(image_ref_file).astype(np.float32)
    # # image_ref = torch.from_numpy(imread(image_ref_file).astype(np.float32).mean(-1) / 255.)[None, ::].cuda()
    # image_ref = torch.from_numpy(img/255.)[None, ::].cuda()

    device = torch.device("cuda")

    # other settings
    camera_distance = 4.0
    elevation = 0.0
    azimuth = 0.0

    # load template once.
    template_file = os.path.join(data_dir, 'my_sphere.obj')
    template = Mesh.from_file(template_file)
    image_size = 128
    # See:
    # https://github.com/jenngrannen-TRI/PyFleX/blob/5ea68201b20112debf9613eda464d751153fdff2/bindings/pyflex.cpp#L1235-L1239

    fov_y = np.pi / 4
    K = np.array([
        intrinsic_matrix_from_fov(image_size, image_size, fov_y),
    ])
    print("K", K)
    # R = np.array([np.eye(3)])
    R = np.array([[
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
        ]])
    # t = np.zeros((1,1,3))
    t = np.array([[1.0, 0, 1.0]])
    # create renderer
    renderer = nr.Renderer(
        camera_mode='projection',
        image_size=256,
        K = K,
        R = R,
        t = t
    )
    # renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)

    # Fake particle positions.
    assert torch.is_grad_enabled()
    # Particle positions.
    assert torch.is_grad_enabled()
    particles_filename = os.path.join(data_dir, "0.h5")
    hf = h5py.File(particles_filename, 'r')
    positions = np.array(hf.get('positions'))[-1]
    # normalize particle positions
    points_mean = np.median(positions, axis=0) 
    positions = positions - points_mean
    hf.close()

    p_WOs = torch.tensor(positions.astype(np.float32), requires_grad=True, device=device)

    assert p_WOs.requires_grad

    optimizer = torch.optim.Adam([p_WOs])
    loop = tqdm.tqdm(range(1))
    for i in loop:
        loop.set_description('Optimizing')
        optimizer.zero_grad()

        # Assemble scene.
        scene = Mesh.empty(device)
        for p_WO in p_WOs:
            scene.add_object(template, p_WO)

        vertices, faces = scene.unsqueeze()
        textures = make_fake_textures(faces)

        mode = "rgb"
        # renderer.eye = nr.get_points_from_angles(2.25, 0, 90) 
        # image = renderer(vertices, faces, textures=textures, mode="silhouettes").squeeze(0)
        image = renderer(vertices, faces, textures=textures, mode=mode).squeeze(0)
        # image = images
        image_save = image.detach().cpu().numpy()

        if i == 0:
            if mode=="rgb":
                image_save = np.transpose(image_save, (1,2,0))
            imsave("image_test_1.png", image_save)
            print("saved")
            exit(0)

        # Fake reduction to check gradients.
        # loss = torch.sum((image - image_ref[None, :, :])**2)
        # loss.backward()
        # optimizer.step()
        # image = images.detach().cpu().numpy()[0]
        imsave('/tmp/_tmp_%04d.png' % i, image_save)
    make_gif(os.path.join(data_dir, "proj_dough_optimization.gif"))

    # # Show gradients of "loss" w.r.t. each position.
    # print(p_WOs.grad)

    renderer = nr.Renderer(
        camera_mode='look_at',
        image_size=256
    )
    # draw object
    loop = tqdm.tqdm(range(0, 8, 4*2))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        renderer.eye = nr.get_points_from_angles(4, 30, azimuth)
        images, _, _ = renderer(vertices, faces, textures)
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        imsave('/tmp/_tmp_%04d.png' % num, image)
    make_gif(os.path.join(data_dir, "proj.gif"))

    particles = p_WOs.detach().cpu().numpy()
    np.save(os.path.join(data_dir, "proj_dough_particles.npy"), particles)

if __name__ == '__main__':
    main()