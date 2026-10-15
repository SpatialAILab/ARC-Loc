import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


VIGOR_CITIES = (
    "NewYork",
    "Seattle",
    "SanFrancisco",
    "Chicago",
)


class VIGORArcDataset(Dataset):
    """VIGOR dataset interface used by ARC-Loc."""

    def __init__(
        self,
        root,
        label_root,
        split,
        train,
        transform=None,
        random_orientation=False,
        use_filtered=False,
    ):
        super().__init__()

        if split not in {"same", "cross"}:
            raise ValueError(
                f"split must be 'same' or 'cross', but received {split!r}"
            )

        if train not in {"train", "test"}:
            raise ValueError(
                f"train must be 'train' or 'test', but received {train!r}"
            )

        self.root = root
        self.label_root = label_root
        self.split = split
        self.train = train
        self.random_orientation = random_orientation
        self.use_filtered = use_filtered

        if transform is None:
            self.ground_transform = None
            self.satellite_transform = None
        else:
            self.ground_transform, self.satellite_transform = transform

        self.samples = self._build_sample_list()

    def _selected_cities(self):
        if self.split == "same":
            return VIGOR_CITIES

        if self.train == "train":
            return VIGOR_CITIES[:2]

        return VIGOR_CITIES[2:]

    def _label_filename(self):
        if self.split == "same":
            if self.train == "train":
                return "same_area_balanced_train.txt"

            if self.use_filtered:
                return "Filtered_same_area_balanced_test.txt"

            return "same_area_balanced_test.txt"

        if self.use_filtered:
            return "Filtered_pano_label_balanced.txt"

        return "pano_label_balanced.txt"

    def _build_sample_list(self):
        samples = []
        label_filename = self._label_filename()

        for city in self._selected_cities():
            label_path = os.path.join(
                self.root,
                self.label_root,
                city,
                label_filename,
            )

            if not os.path.isfile(label_path):
                raise FileNotFoundError(
                    f"VIGOR label file was not found: {label_path}"
                )

            with open(label_path, "r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    line = line.strip()

                    if not line:
                        continue

                    sample = self._parse_label_line(
                        city=city,
                        line=line,
                        line_number=line_number,
                        label_path=label_path,
                    )
                    samples.append(sample)

        return samples

    @staticmethod
    def _parse_label_line(city, line, line_number, label_path):
        fields = line.split()

        # VIGOR labels contain one panorama followed by four
        # satellite candidates and their corresponding 2-D offsets.
        if len(fields) < 13:
            raise ValueError(
                f"Malformed VIGOR label at {label_path}:{line_number}"
            )

        ground_name = fields[0]
        satellite_name = fields[1]

        try:
            offset_y = float(fields[2])
            offset_x = float(fields[3])
        except ValueError as exc:
            raise ValueError(
                f"Invalid offset at {label_path}:{line_number}"
            ) from exc

        return {
            "city": city,
            "ground_name": ground_name,
            "satellite_name": satellite_name,
            "offset": np.asarray(
                [offset_y, offset_x],
                dtype=np.float32,
            ),
        }

    def __len__(self):
        return len(self.samples)

    def _ground_path(self, sample):
        return os.path.join(
            self.root,
            sample["city"],
            "panorama",
            sample["ground_name"],
        )

    def _satellite_path(self, sample):
        return os.path.join(
            self.root,
            sample["city"],
            "satellite",
            sample["satellite_name"],
        )

    @staticmethod
    def _sample_rotation():
        return random.uniform(-0.5, 0.5)

    @staticmethod
    def _rotation_matrix(rotation):
        angle = -2.0 * np.pi * rotation

        cosine = np.cos(angle)
        sine = np.sin(angle)

        return torch.tensor(
            [
                [cosine, -sine],
                [sine, cosine],
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def _roll_panorama(image, rotation):
        if not torch.is_tensor(image):
            raise TypeError(
                "Ground transform must return a torch.Tensor."
            )

        shift = int(round(rotation * image.shape[-1]))
        return torch.roll(image, shifts=shift, dims=-1)

    @staticmethod
    def _scale_offset(offset, original_size, transformed_size):
        original_width, original_height = original_size
        transformed_height, transformed_width = transformed_size

        scale_y = transformed_height / float(original_height)
        scale_x = transformed_width / float(original_width)

        scaled = offset.copy()
        scaled[0] *= -scale_y
        scaled[1] *= scale_x

        return torch.from_numpy(scaled).unsqueeze(0)

    def __getitem__(self, index):
        sample = self.samples[index]

        ground = Image.open(
            self._ground_path(sample)
        ).convert("RGB")

        satellite = Image.open(
            self._satellite_path(sample)
        ).convert("RGB")

        satellite_original_size = satellite.size

        if self.ground_transform is not None:
            ground = self.ground_transform(ground)

        if self.satellite_transform is not None:
            satellite = self.satellite_transform(satellite)

        if not torch.is_tensor(ground):
            raise TypeError(
                "Ground transform must return a torch.Tensor."
            )

        if not torch.is_tensor(satellite):
            raise TypeError(
                "Satellite transform must return a torch.Tensor."
            )

        if self.random_orientation:
            rotation = self._sample_rotation()
        else:
            rotation = 0.0

        ground = self._roll_panorama(ground, rotation)
        rotation_matrix = self._rotation_matrix(rotation)


        gt_loc = self._scale_offset(
            sample["offset"],
            original_size=satellite_original_size,
            transformed_size=satellite.shape[-2:],
        )

        auxiliary = torch.zeros(1, dtype=torch.float32)

        return (
            ground,
            satellite,
            gt_loc,
            rotation_matrix,
            sample["city"],
            auxiliary,
        )