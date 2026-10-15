import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

import configparser
import ast


config = configparser.ConfigParser()
config.read("./config.ini")


GROUND_IMAGE_SIZE = ast.literal_eval(
    config.get("KITTI", "ground_image_size")
)
SATELLITE_IMAGE_SIZE = ast.literal_eval(
    config.get("KITTI", "satellite_image_size")
)

DATASET_ROOT = config.get("KITTI", "dataset_root")
TRAIN_FILE = config.get("KITTI", "train_file")
TEST1_FILE = config.get("KITTI", "test1_file")
TEST2_FILE = config.get("KITTI", "test2_file")

SHIFT_RANGE_LAT = config.getfloat("KITTI", "shift_range_lat")
SHIFT_RANGE_LON = config.getfloat("KITTI", "shift_range_lon")
ROTATION_RANGE = config.getfloat("KITTI", "rotation_range")


ORIGINAL_GROUND_SIZE = (375, 1242)
ORIGINAL_SATELLITE_SIZE = 512
SATELLITE_ZOOM = 18
REFERENCE_LATITUDE = 49.015

CAMERA_GPS_SHIFT = (1.08, 0.26)


satellite_transform = transforms.Compose(
    [
        transforms.Resize(SATELLITE_IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)

ground_transform = transforms.Compose(
    [
        transforms.Resize(GROUND_IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)


def get_meter_per_pixel(
    latitude=REFERENCE_LATITUDE,
    zoom=SATELLITE_ZOOM,
    scale=None,
):
    if scale is None:
        scale = (
            SATELLITE_IMAGE_SIZE[0]
            / ORIGINAL_SATELLITE_SIZE
        )

    resolution = (
        156543.03392
        * np.cos(np.deg2rad(latitude))
        / (2 ** zoom)
    )

    resolution /= 2.0
    resolution /= scale

    return resolution


def _read_split_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def _frame_parts(frame_path):
    day = frame_path[:10]
    drive = frame_path[:38]
    image_name = frame_path[38:]

    return day, drive, image_name


def _load_camera_parameters(root, day):
    calibration_path = os.path.join(
        root,
        "raw_data",
        day,
        "calib_cam_to_cam.txt",
    )

    projection_values = None

    with open(
        calibration_path,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.startswith("P_rect_02"):
                projection_values = [
                    float(value)
                    for value in line.split(":", 1)[1].split()
                ]
                break

    if projection_values is None:
        raise ValueError(
            f"P_rect_02 was not found in {calibration_path}"
        )

    original_height, original_width = ORIGINAL_GROUND_SIZE
    target_height, target_width = GROUND_IMAGE_SIZE

    fx = projection_values[0] * target_width / original_width
    cx = projection_values[2] * target_width / original_width
    fy = projection_values[5] * target_height / original_height
    cy = projection_values[6] * target_height / original_height

    camera_matrix = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    horizontal_fov = 2.0 * np.arctan(
        target_width / (2.0 * fx)
    )

    fov_deg = torch.tensor(
        [np.rad2deg(horizontal_fov)],
        dtype=torch.float32,
    )

    return camera_matrix, fov_deg


def _load_heading(root, drive, image_name):
    oxts_path = os.path.join(
        root,
        "raw_data",
        drive,
        "oxts",
        "data",
        image_name.lower().replace(".png", ".txt"),
    )

    with open(
        oxts_path,
        "r",
        encoding="utf-8",
    ) as file:
        values = file.readline().split()

    return float(values[5])


def _load_ground_image(root, drive, image_name):
    path = os.path.join(
        root,
        "raw_data",
        drive,
        "image_02",
        "data",
        image_name.lower(),
    )

    with Image.open(path) as image:
        return image.convert("RGB")


def _load_satellite_image(root, frame_path):
    path = os.path.join(
        root,
        "satmap",
        frame_path,
    )

    with Image.open(path) as image:
        return image.convert("RGB")


def _align_satellite_to_camera(satellite, heading, meter_per_pixel):
    """Rotate and translate the satellite image into the camera-centered frame."""
    heading_deg = np.rad2deg(heading)

    aligned = satellite.rotate(
        90.0 - heading_deg,
        resample=Image.BILINEAR,
    )

    shift_x = CAMERA_GPS_SHIFT[0] / meter_per_pixel
    shift_y = CAMERA_GPS_SHIFT[1] / meter_per_pixel

    return aligned.transform(
        aligned.size,
        Image.AFFINE,
        (1.0, 0.0, shift_x, 0.0, 1.0, shift_y),
        resample=Image.BILINEAR,
    )


def _apply_pose_perturbation(
    satellite,
    shift_x,
    shift_y,
    rotation_deg,
    shift_pixels_lon,
    shift_pixels_lat,
):
    """Apply translation and rotation perturbations to a satellite image."""
    shifted = satellite.transform(
        satellite.size,
        Image.AFFINE,
        (
            1.0,
            0.0,
            shift_x * shift_pixels_lon,
            0.0,
            1.0,
            -shift_y * shift_pixels_lat,
        ),
        resample=Image.BILINEAR,
    )

    rotated = shifted.rotate(
        rotation_deg,
        resample=Image.BILINEAR,
    )

    return TF.center_crop(
        rotated,
        ORIGINAL_SATELLITE_SIZE,
    )


def _translation_target(
    shift_x,
    shift_y,
    rotation_deg,
    shift_pixels_lon,
    shift_pixels_lat,
    quantize=False,
):
    """Convert normalized perturbations into ARC-Loc satellite pixel offsets."""
    angle = np.deg2rad(rotation_deg)

    x_offset = (
        shift_x * shift_pixels_lon * np.cos(angle)
        - shift_y * shift_pixels_lat * np.sin(angle)
    )

    y_offset = (
        -shift_y * shift_pixels_lat * np.cos(angle)
        - shift_x * shift_pixels_lon * np.sin(angle)
    )

    if quantize:
        x_offset = int(x_offset)
        y_offset = int(y_offset)

    scale = (
        SATELLITE_IMAGE_SIZE[0]
        / ORIGINAL_SATELLITE_SIZE
    )

    return torch.tensor(
        [[y_offset * scale, x_offset * scale]],
        dtype=torch.float32,
    )


def _rotation_target(rotation_deg):
    """Return the 2-D rotation from satellite coordinates to ground coordinates."""
    yaw = -np.deg2rad(rotation_deg)

    cosine = np.cos(yaw)
    sine = np.sin(yaw)

    return torch.tensor(
        [
            [cosine, -sine],
            [sine, cosine],
        ],
        dtype=torch.float32,
    )


class _KITTIBaseDataset(Dataset):
    def __init__(
        self,
        root,
        split_file,
        transform=None,
        shift_range_lat=20.0,
        shift_range_lon=20.0,
        rotation_range=10.0,
    ):
        self.root = root
        self.entries = _read_split_file(split_file)

        self.shift_range_lat = shift_range_lat
        self.shift_range_lon = shift_range_lon
        self.rotation_range = rotation_range

        # Perturbations are defined in the coordinate system of
        # the original satellite map.
        self.meter_per_pixel = get_meter_per_pixel(scale=1.0)
        self.shift_pixels_lat = (
            shift_range_lat / self.meter_per_pixel
        )
        self.shift_pixels_lon = (
            shift_range_lon / self.meter_per_pixel
        )

        if transform is None:
            self.satellite_transform = None
            self.ground_transform = None
        else:
            self.satellite_transform, self.ground_transform = transform

    def __len__(self):
        return len(self.entries)

    def _load_common(self, frame_path):
        day, drive, image_name = _frame_parts(frame_path)

        camera_matrix, fov_deg = _load_camera_parameters(
            self.root,
            day,
        )

        heading = _load_heading(
            self.root,
            drive,
            image_name,
        )

        ground = _load_ground_image(
            self.root,
            drive,
            image_name,
        )

        satellite = _load_satellite_image(
            self.root,
            frame_path,
        )

        satellite = _align_satellite_to_camera(
            satellite,
            heading,
            self.meter_per_pixel,
        )

        return (
            satellite,
            ground,
            camera_matrix,
            fov_deg,
        )

    def _finalize_sample(
        self,
        satellite,
        ground,
        camera_matrix,
        fov_deg,
        shift_x,
        shift_y,
        rotation_deg,
        quantize_target=False,
    ):
        satellite = _apply_pose_perturbation(
            satellite=satellite,
            shift_x=shift_x,
            shift_y=shift_y,
            rotation_deg=rotation_deg,
            shift_pixels_lon=self.shift_pixels_lon,
            shift_pixels_lat=self.shift_pixels_lat,
        )

        if self.satellite_transform is not None:
            satellite = self.satellite_transform(satellite)

        if self.ground_transform is not None:
            ground = self.ground_transform(ground)

        if not torch.is_tensor(satellite):
            raise TypeError(
                "Satellite transform must return a torch.Tensor."
            )

        if not torch.is_tensor(ground):
            raise TypeError(
                "Ground transform must return a torch.Tensor."
            )

        gt_loc = _translation_target(
            shift_x=shift_x,
            shift_y=shift_y,
            rotation_deg=rotation_deg,
            shift_pixels_lon=self.shift_pixels_lon,
            shift_pixels_lat=self.shift_pixels_lat,
            quantize=quantize_target,
        )

        rotation_matrix = _rotation_target(rotation_deg)

        return (
            satellite,
            ground,
            camera_matrix,
            gt_loc,
            rotation_matrix,
            fov_deg,
        )


class KITTITrainDataset(_KITTIBaseDataset):
    def __getitem__(self, index):
        frame_path = self.entries[index]

        satellite, ground, camera_matrix, fov_deg = (
            self._load_common(frame_path)
        )

        shift_x = np.random.uniform(-1.0, 1.0)
        shift_y = np.random.uniform(-1.0, 1.0)
        rotation_deg = (
            np.random.uniform(-1.0, 1.0)
            * self.rotation_range
        )

        return self._finalize_sample(
            satellite=satellite,
            ground=ground,
            camera_matrix=camera_matrix,
            fov_deg=fov_deg,
            shift_x=shift_x,
            shift_y=shift_y,
            rotation_deg=rotation_deg,
            quantize_target=True,
        )


class KITTITestDataset(_KITTIBaseDataset):
    def __getitem__(self, index):
        fields = self.entries[index].split()

        if len(fields) != 4:
            raise ValueError(
                "KITTI test split entries must contain "
                "'frame shift_x shift_y rotation'."
            )

        frame_path, shift_x, shift_y, rotation = fields

        satellite, ground, camera_matrix, fov_deg = (
            self._load_common(frame_path)
        )

        # Preserve the coordinate convention used by the
        # pre-generated KITTI evaluation splits.
        shift_x = -float(shift_x)
        shift_y = -float(shift_y)
        rotation_deg = (
            float(rotation)
            * self.rotation_range
        )

        return self._finalize_sample(
            satellite=satellite,
            ground=ground,
            camera_matrix=camera_matrix,
            fov_deg=fov_deg,
            shift_x=shift_x,
            shift_y=shift_y,
            rotation_deg=rotation_deg,
            quantize_target=True,
        )


train_set = KITTITrainDataset(
    root=DATASET_ROOT,
    split_file=TRAIN_FILE,
    transform=(satellite_transform, ground_transform),
    shift_range_lat=SHIFT_RANGE_LAT,
    shift_range_lon=SHIFT_RANGE_LON,
    rotation_range=ROTATION_RANGE,
)

test1_set = KITTITestDataset(
    root=DATASET_ROOT,
    split_file=TEST1_FILE,
    transform=(satellite_transform, ground_transform),
    shift_range_lat=SHIFT_RANGE_LAT,
    shift_range_lon=SHIFT_RANGE_LON,
    rotation_range=ROTATION_RANGE,
)

test2_set = KITTITestDataset(
    root=DATASET_ROOT,
    split_file=TEST2_FILE,
    transform=(satellite_transform, ground_transform),
    shift_range_lat=SHIFT_RANGE_LAT,
    shift_range_lon=SHIFT_RANGE_LON,
    rotation_range=ROTATION_RANGE,
)