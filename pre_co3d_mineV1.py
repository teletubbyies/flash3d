import numpy
import torch
import torchvision
import numpy as np
from matplotlib import pyplot as plt

import math
import os
import tqdm
from PIL import Image
from omegaconf import DictConfig

from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from scipy.ndimage import binary_dilation, generate_binary_structure

from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import JsonIndexDatasetMapProviderV2
from pytorch3d.implicitron.tools.config import expand_args_fields
from torchvision import transforms

from datasets.bad_sequences import (
    NAN_SEQUENCES,
    NO_FG_COND_FRAME_SEQ,
    LARGE_FOCAL_FRAME_SEQ,
    EXCLUDE_SEQUENCE,
    CAMERAS_CLOSE_SEQUENCE,
    CAMERAS_FAR_AWAY_SEQUENCE,
    LOW_QUALITY_SEQUENCE
    )


CO3D_RAW_ROOT = os.path.normpath("E:\data\splatter-image-main\co3d")  # change to where your CO3D data resides
CO3D_OUT_ROOT = os.path.normpath("E:\data\splatter-image-main\co3d_ansV1_1024")  # change to your folder here
N_DESIRED = 1024

assert CO3D_RAW_ROOT is not None, "Change CO3D_RAW_ROOT to where your raw CO3D data resides"
assert CO3D_OUT_ROOT is not None, "Change CO3D_OUT_ROOT to where you want to save the processed CO3D data"


def exact_num_superpixels(image, n_desired, compactness=10, sigma=0):
    """
        Get the exact number of superpixels in an image.
    Args:
        image: (ndarray). [H, W, 3].
        n_desired: int. The expected number of superpixels.
        compactness: parameter used in slic.
        sigma: parameter used in slic.

    Returns:
        segments:[H, W]. Integer mask indicating segment labels and each pixel has its own superpixel(label).
    """
    image = np.ascontiguousarray(image, dtype=np.float32)
    # Start with more superpixels than desired
    n_segments = int(n_desired * 1.15)

    while True:
        # Perform SLIC
        segments = slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma)

        # Count unique labels
        unique_labels = np.unique(segments)
        n_current = len(unique_labels)

        if n_current >= n_desired:
            break

        # If we have too few, increase n_segments and try again
        n_segments = int(n_segments * 1.15)

    while n_current > n_desired:
        # Always have too many at this step, merge the smallest regions
        # If equals, skip this step, but still should rearrange SuperPixel label to [0, n_desired-1].

        # Find the smallest region.
        props = regionprops(segments)
        smallest_region = min(props, key=lambda x: x.area)
        smallest_label = smallest_region.label

        # Create a mask for the smallest region
        mask = segments == smallest_label

        # Dilate the mask to find neighbors
        struct = generate_binary_structure(2, 2)
        dilated = binary_dilation(mask, structure=struct)

        # Find unique labels in the dilated region, excluding the smallest label itself
        neighbor_labels = np.unique(segments[dilated & ~mask])

        if len(neighbor_labels) > 0:
            # Merge with the smallest neighbor
            merge_label = min(neighbor_labels, key=lambda x: np.sum(segments == x))
            segments[segments == smallest_label] = merge_label
            n_current -= 1 # Subtract the smallest region.
        else:
            raise ValueError("The smallest region has no neighbors.")

    # Visualize
    # plt.imshow(mark_boundaries(image, segments))
    # plt.show()

    # Rearrange SuperPixel label to [0, n_desired-1]
    segments = torch.from_numpy(segments)

    unique_labels = torch.unique(segments)
    label_to_index = torch.zeros(int(segments.max().item() + 1), dtype=torch.long)
    label_to_index[unique_labels] = torch.arange(n_desired)

    ans = label_to_index[segments]

    assert ans.max() < n_desired and len(torch.unique(ans)) == n_desired, 'something wrong, check'

    return ans

def savefig(t, file):
    to_pil = transforms.ToPILImage()
    image = to_pil(t)
    image.save(file)

def vis_picture(map, is_depth_or_mask=False):
    # depth_map: [1, H, W]
    # image_rgb: [3, H, W]
    if is_depth_or_mask:
        plt.imshow(map.squeeze(), cmap='gray')
        plt.colorbar(label='Depth Value')
        plt.title('Depth Map')
    else:
        # rgb
        map = np.transpose(map, (1, 2, 0))
        plt.imshow(map)
        plt.title('Image')

    # plt.axis('off')
    plt.show()


def vis_pointcloud(rgb, xyz):
    """

    Args:
        rgb: Tensor[128, 128, 3]
        xyz: ndarray[128*128, 3]
    Returns:

    """
    # plot pointcloud
    colors = (torch.clamp(rgb.permute(1, 2, 0).reshape(-1, 3), 0, 1) * 255).numpy().astype(np.uint8)
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "browser"
    fig = go.Figure(data=[go.Scatter3d(
        x=xyz.reshape(-1, 3)[:, 0],
        y=xyz.reshape(-1, 3)[:, 1],
        z=xyz.reshape(-1, 3)[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=colors,
            opacity=0.8
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='Visualize Pointcloud'
    )

    fig.show()


def pixel_to_world(depth, K, w2c=None):
    """

    Args:
        depth: [128, 128]
        K: [3, 3]
        w2c: [4, 4], the last row is [0, 0, 0, 1]

    Returns: [128, 128, 3]

    """
    pts_mask = depth > 0.0  # foreground
    # pts_mask = pts_mask.reshape(-1)

    height, width = depth.shape

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    pixels = np.stack((x.flatten(), y.flatten(), np.ones_like(x.flatten())), axis=-1).T

    camera_coords = np.linalg.inv(K) @ pixels

    camera_coords *= depth.flatten()

    if w2c is not None:
        camera_coords = np.vstack((camera_coords, np.ones((1, camera_coords.shape[1]))))
        world_coords = np.linalg.inv(w2c) @ camera_coords

        world_coords = world_coords / world_coords[3]

        world_coords = world_coords[:3].T
        world_coords = world_coords.reshape(height, width, -1)

        return world_coords, pts_mask.reshape(128, 128, 1)

    else:
        camera_coords = camera_coords.T.reshape(height, width, -1)
        pts_mask = pts_mask.reshape(height, width, 1)
        pts_mask = np.tile(pts_mask, (1, 1, 3))
        camera_coords[~pts_mask] = float('inf')
        return camera_coords, pts_mask


def save_depth_image(depth_map, filename, cmap='viridis'):
    plt.figure(figsize=(10, 10))
    plt.imshow(depth_map, cmap=cmap)
    plt.colorbar(label='Depth')
    plt.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def update_scores(top_scores, top_names, new_score, new_name):
    for sc_idx, sc in enumerate(top_scores):
        if new_score > sc:
            # shift scores and names to the right, start from the end
            for sc_idx_next in range(len(top_scores) - 1, sc_idx, -1):
                top_scores[sc_idx_next] = top_scores[sc_idx_next - 1]
                top_names[sc_idx_next] = top_names[sc_idx_next - 1]
            top_scores[sc_idx] = new_score
            top_names[sc_idx] = new_name
            break
    return top_scores, top_names


def normalize(seen_xyz):
    # [128, 128, 3]
    seen_xyz = seen_xyz / (seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].var(dim=0) ** 0.5).mean()
    seen_xyz = seen_xyz - seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].mean(axis=0)
    return seen_xyz


def main(dataset_name, category):
    subset_name = "fewview_dev"

    expand_args_fields(JsonIndexDatasetMapProviderV2)
    dataset_map = JsonIndexDatasetMapProviderV2(

        category=category,
        subset_name=subset_name,
        test_on_train=False,
        only_test_set=False,
        load_eval_batches=True,
        dataset_root=CO3D_RAW_ROOT,
        dataset_JsonIndexDataset_args=DictConfig(
            {"remove_empty_masks": False, "load_point_clouds": True}
        ),
    ).get_dataset_map()

    created_dataset = dataset_map[dataset_name]

    sequence_names = [k for k in created_dataset.seq_annots.keys()]
    

    bkgd = 0.0  # black background

    out_folder_path = os.path.join(CO3D_OUT_ROOT, "co3d_{}_for_gs".format(category),
                                   dataset_name)
    os.makedirs(out_folder_path, exist_ok=True)

    bad_sequences = []
    camera_Rs_all_sequences = {}
    camera_Ts_all_sequences = {}


    exclude_sequences = NO_FG_COND_FRAME_SEQ[category] + \
                        LARGE_FOCAL_FRAME_SEQ[category] + \
                        NAN_SEQUENCES[category] + \
                        EXCLUDE_SEQUENCE[category] + \
                        CAMERAS_CLOSE_SEQUENCE[category] + \
                        CAMERAS_FAR_AWAY_SEQUENCE[category] + \
                        LOW_QUALITY_SEQUENCE[category]


    for sequence_name in tqdm.tqdm(sequence_names, desc=f"Preparing: {dataset_name}-{category}"):

        if sequence_name in exclude_sequences:
            continue

        folder_outname = os.path.join(out_folder_path, sequence_name)

        frame_idx_gen = created_dataset.sequence_indices_in_order(sequence_name)
        frame_idxs = []

        fname_order = []

        depth_fg_this_sequence = []
        segments_fg_this_sequence = []
        focal_lengths_this_sequence = []
        rgb_fg_this_sequence = []
        xyz_fg_this_sequence = []

        # Preprocess cameras with Viewset Diffusion protocol
        cameras_this_seq = read_seq_cameras(created_dataset, sequence_name)

        camera_Rs_all_sequences[sequence_name] = cameras_this_seq.R
        camera_Ts_all_sequences[sequence_name] = cameras_this_seq.T

        while True:
            try:
                frame_idx = next(frame_idx_gen)
                frame_idxs.append(frame_idx)
            except StopIteration:
                break

        # Preprocess images
        for frame_idx in frame_idxs:
            # Read the original uncropped image
            # 读入的是原图
            """
            论文中“类似于近期的方法，我们从原始图像中以主点为中心裁剪出最大的区域，并使用Lanczos插值法将其调整为128 × 128分辨率。
            与许多单视图和少视图重建方法一样，我们也去除了背景。我们根据生成的变换相应地调整焦距。
            这是唯一的预处理步骤——CO3D对象的点云已经归一化为零均值和单位方差。”
            """
            frame = created_dataset[frame_idx]
            rgb_image = torchvision.transforms.functional.pil_to_tensor(
                Image.open(frame.image_path)).float() / 255.0
            # [3, 1251, 703]
            ## new
            depth_fg = torch.zeros_like(rgb_image)[:1, ...]  # [1, 1251(H), 703(W)]
            ##
            # [1, 1251, 703]
            # ============= Foreground mask =================
            # Initialise the foreground mask at the original resolution
            fg_probability = torch.zeros_like(rgb_image)[:1, ...]  # [1, 1251(H), 703(W)]
            # Find size of the valid region in the 800x800 image (non-padded)
            #
            # 800x800图片中物体的有效范围
            resized_image_mask_boundary_y = torch.where(frame.mask_crop > 0)[1].max() + 1  # H(y) 1是物体，0是边界(checked!)
            resized_image_mask_boundary_x = torch.where(frame.mask_crop > 0)[2].max() + 1  # W(x) Opencv
            # Resize the foreground mask to the original scale
            # 自带crop在原图中的标定框
            x0, y0, box_w, box_h = frame.crop_bbox_xywh
            resized_mask = torchvision.transforms.functional.resize(  # resize后的mask->raw_data的mask
                frame.fg_probability[:, :resized_image_mask_boundary_y, :resized_image_mask_boundary_x],
                # 800x800中的mask
                (box_h, box_w),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )  # 原图中 自带crop部分的mask(fg_probability)

            # new
            resized_depth = torchvision.transforms.functional.resize(
                frame.depth_map[:, :resized_image_mask_boundary_y, :resized_image_mask_boundary_x],
                (box_h, box_w),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )

            # Use rgb mask as depth mask due to many frame.depth_mask is all zeros!
            resized_depth_mask = torch.where(resized_mask > 0.4, torch.tensor(1.0), torch.tensor(0.0))
            resized_depth *= resized_depth_mask

            # Fill in the depth at the original scale in the correct location based
            # on where it was cropped.
            depth_fg[:, y0:y0 + box_h, x0:x0 + box_w] = resized_depth

            # Fill in the foreground mask at the original scale in the correct location based
            # on where it was cropped.
            fg_probability[:, y0:y0 + box_h, x0:x0 + box_w] = resized_mask

            # ============== Crop around principal point ================
            # compute location of principal point in Pytorch3D NDC coordinate system in pixels.（相机中心在成像平面上的坐标）
            # scaling * 0.5 is due to the NDC min and max range being +- 1
            principal_point_cropped = frame.camera.principal_point * 0.5 * frame.image_rgb.shape[
                1]  # frame.image_rgb.shape[1](800)
            # compute location of principal point from top left corer, i.e. in image grid coords
            scaling_factor = max(box_h, box_w) / 800
            principal_point_x = (frame.image_rgb.shape[2] * 0.5 - principal_point_cropped[0, 0]) * scaling_factor + x0
            principal_point_y = (frame.image_rgb.shape[1] * 0.5 - principal_point_cropped[0, 1]) * scaling_factor + y0

            # Get the largest center-crop that fits in the foreground
            max_half_side = get_max_box_side(
                frame.image_size_hw, principal_point_x, principal_point_y)
            # After this transformation principal point is at (0, 0)
            rgb = crop_image_at_non_integer_locations(rgb_image, max_half_side,
                                                      principal_point_x, principal_point_y)
            fg_probability_cc = crop_image_at_non_integer_locations(fg_probability, max_half_side,
                                                                    principal_point_x, principal_point_y)

            # new

            depth_fg = crop_image_at_non_integer_locations(depth_fg, max_half_side,
                                                           principal_point_x, principal_point_y)
            ##
            assert frame.image_rgb.shape[1] == frame.image_rgb.shape[2], "Expected square images"

            # =============== Resize to 128 and save =======================
            # Resize raw rgb
            pil_rgb = torchvision.transforms.functional.to_pil_image(rgb)
            pil_rgb = torchvision.transforms.functional.resize(pil_rgb,
                                                               128,
                                                               interpolation=torchvision.transforms.InterpolationMode.LANCZOS)
            rgb = torchvision.transforms.functional.pil_to_tensor(pil_rgb) / 255.0
            # Resize mask
            fg_probability_cc = torchvision.transforms.functional.resize(fg_probability_cc,
                                                                         128,
                                                                         interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            # new
            depth_fg = torchvision.transforms.functional.resize(depth_fg,
                                                                128,
                                                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)


            # Save masked rgb
            rgb_fg = rgb[:3, ...] * fg_probability_cc + bkgd * (1 - fg_probability_cc)
            rgb_fg_this_sequence.append(rgb_fg)

            segments_fg = exact_num_superpixels(rgb_fg.permute(1, 2, 0).numpy(), n_desired=N_DESIRED)
            segments_fg_this_sequence.append(segments_fg)

            # new
            # Save masked depth
            depth_fg_this_sequence.append(depth_fg)

            fname_order.append("{:05d}.png".format(frame_idx))

            # ============== Intrinsics transformation =================
            # Transform focal length according to the crop
            # Focal length is in NDC conversion so we do not need to change it when resizing
            # We should transform focal length to non-cropped image and then back to cropped but
            # the scaling factor of the full non-cropped image cancels out.
            transformed_focal_lengths = frame.camera.focal_length * max(box_h, box_w) / (2 * max_half_side)
            focal_lengths_this_sequence.append(transformed_focal_lengths)

            # new
            K = np.array(
                [[transformed_focal_lengths[0, 0].item(), 0, 64], [0, transformed_focal_lengths[0, 1].item(), 64],
                 [0, 0, 1]])
            xyz, pts_mask = pixel_to_world(depth_fg.squeeze(0).numpy(), K)
            # print(f'datasetname:{dataset_name}, catgory:{category}', pts_mask[..., 0].sum())
            # colors = (torch.clamp(rgb.permute(1,2, 0).reshape(-1, 3), 0, 1) * 255).numpy().astype(np.uint8)
            # import plotly.graph_objects as go
            # import plotly.io as pio
            # pio.renderers.default = "browser"
            # fig = go.Figure(data=[go.Scatter3d(
            #     x=xyz[pts_mask[..., 0]].reshape(-1, 3)[:, 0],
            #     y=xyz[pts_mask[..., 0]].reshape(-1, 3)[:, 1],
            #     z=xyz[pts_mask[..., 0]].reshape(-1, 3)[:, 2],
            #     mode='markers',
            #     marker=dict(
            #         size=5,
            #         color=colors.reshape(-1, 3)[pts_mask[..., 0].reshape(-1)],
            #         opacity=0.8
            #     )
            # )])
            #
            # fig.update_layout(
            #     scene=dict(
            #         xaxis_title='X',
            #         yaxis_title='Y',
            #         zaxis_title='Z'
            #     ),
            #     title='Visualize Pointcloud'
            # )
            #
            # fig.show()

            # save xyz
            xyz_fg = normalize(torch.from_numpy(xyz)).permute(-1, 0, 1)
            if torch.any(xyz_fg.isnan()):
                print(f'{sequence_name}-{frame_idx} has nan xyz')

            xyz_fg_this_sequence.append(xyz_fg)

        os.makedirs(folder_outname, exist_ok=True)
        focal_lengths_this_sequence = torch.stack(focal_lengths_this_sequence)

        if torch.all(torch.logical_not(torch.stack(rgb_fg_this_sequence).isnan())) and \
                torch.all(torch.logical_not(focal_lengths_this_sequence.isnan())) and \
                torch.all(torch.logical_not(torch.stack(depth_fg_this_sequence).isnan())) and \
                torch.all(torch.logical_not(torch.stack(xyz_fg_this_sequence).isnan())) and \
                torch.all(torch.logical_not(torch.stack(segments_fg_this_sequence).isnan())): # xyz may have nan

            np.save(os.path.join(folder_outname, "images_fg.npy"), torch.stack(rgb_fg_this_sequence).numpy())
            np.save(os.path.join(folder_outname, "focal_lengths.npy"), focal_lengths_this_sequence.numpy())
            np.save(os.path.join(folder_outname, "depth_fg.npy"), torch.stack(depth_fg_this_sequence).numpy())
            np.save(os.path.join(folder_outname, "xyz_fg.npy"), torch.stack(xyz_fg_this_sequence).numpy())
            np.save(os.path.join(folder_outname, "segments.npy"), torch.stack(segments_fg_this_sequence).numpy())

            with open(os.path.join(folder_outname, "frame_order.txt"), "w+") as f:
                f.writelines([fname + "\n" for fname in fname_order])
        else:
            print("Warning! bad sequence {}".format(sequence_name))
            bad_sequences.append(sequence_name)

    # convert camera data to numpy archives and save
    for dict_to_save, dict_name in zip([camera_Rs_all_sequences,
                                        camera_Ts_all_sequences],
                                       ["camera_Rs",
                                        "camera_Ts"]):
        np.savez(os.path.join(out_folder_path, dict_name + ".npz"),
                 **{k: v.numpy() for k, v in dict_to_save.items()})

    return bad_sequences


def get_max_box_side(hw, principal_point_x, principal_point_y):
    # assume images are always padded on the right - find where the image ends
    # find the largest center crop we can make
    max_x = hw[1]  # x-coord of the rightmost boundary
    min_x = 0.0  # x-coord of the leftmost boundary
    max_y = hw[0]  # y-coord of the top boundary
    min_y = 0.0  # y-coord of the bottom boundary

    max_half_w = min(principal_point_x - min_x, max_x - principal_point_x)
    max_half_h = min(principal_point_y - min_y, max_y - principal_point_y)
    max_half_side = min(max_half_h, max_half_w)

    return max_half_side


def crop_image_at_non_integer_locations(img,
                                        max_half_side: float,
                                        principal_point_x: float,
                                        principal_point_y: float):
    """
    Crops the image so that its center is at the principal point.
    The boundaries are specified by half of the image side.
    """
    # number of pixels that the image spans. We don't want to resize
    # at this stage. However, the boundaries might be such that
    # the crop side is not an integer. Therefore there will be
    # minimal resizing, but it's extent will be sub-pixel.
    # We don't apply low-pass filtering at this stage and cropping is
    # done with bilinear sampling
    max_pixel_number = math.floor(2 * max_half_side)
    half_pixel_side = 0.5 / max_pixel_number
    x_locations = torch.linspace(principal_point_x - max_half_side + half_pixel_side,
                                 principal_point_x + max_half_side - half_pixel_side,
                                 max_pixel_number)
    y_locations = torch.linspace(principal_point_y - max_half_side + half_pixel_side,
                                 principal_point_y + max_half_side - half_pixel_side,
                                 max_pixel_number)
    grid_locations = torch.stack(torch.meshgrid(x_locations, y_locations, indexing='ij'), dim=-1).transpose(0, 1)
    grid_locations[:, :, 1] = (grid_locations[:, :, 1] - img.shape[1] / 2) / (img.shape[1] / 2)
    grid_locations[:, :, 0] = (grid_locations[:, :, 0] - img.shape[2] / 2) / (img.shape[2] / 2)
    image_crop = torch.nn.functional.grid_sample(img.unsqueeze(0), grid_locations.unsqueeze(0))
    return image_crop.squeeze(0)


def read_seq_cameras(dataset, sequence_name):
    """
    没有改变任何的R/T
    Args:
        dataset:
        sequence_name:

    Returns:

    """

    frame_idx_gen = dataset.sequence_indices_in_order(sequence_name)
    frame_idxs = []
    while True:
        try:
            frame_idx = next(frame_idx_gen)
            frame_idxs.append(frame_idx)
        except StopIteration:
            break

    cameras_start = []
    for frame_idx in frame_idxs:
        cameras_start.append(dataset[frame_idx].camera)
    cameras_start = join_cameras_as_batch(cameras_start)
    cameras = cameras_start.clone()

    return cameras


if __name__ == "__main__":
    for category in ["hydrant"]:
        for split in ["train", "val"]:
            bad_sequences_val = main(split, category)
