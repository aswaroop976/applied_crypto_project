"""List of dataloaders."""

import collections
import json
from typing import Optional
import os


ImageDetails = collections.namedtuple("ImageDetails", ("name", "path", "hash", "rel_path"))


class Imagenet(object):
    """Dataloader for ImageNet dataset.

    It loads the images from the dataset_path and excludes the images
    mentioned in the JSON file, as well as the duplicates if the filepaths are passed.
    The JSON file should be the one containing
    annotations of faces as found by Yang. et. al. 2021.

    Args:
        dataset_path: Path to the ImageNet dataset.
        face_annotations_json_path: Path to the JSON file containing face annotations.
        duplicates_path: Path to the json file containing duplicate image paths.
    """

    def __init__(
        self,
        dataset_path: str,
        face_annotations_json_path: Optional[str] = None,
        duplicates_path: Optional[str] = None,
    ):
        """Initialize dataloader."""
        self.dataset_path = dataset_path
        self.face_annotations_json_path = face_annotations_json_path
        self.duplicates_path = duplicates_path
        self.images = self.load_dataset()

    def load_dataset(self):
        """Generate a list of all images in the folder.

        It creates a list of images that are to be excluded.
        Then it generates a list of all images, excluding those detected
        in the previous step.
        """
        face_annotations = []
        print("Loading face annotations JSON file.")
        if self.face_annotations_json_path:
            with open(self.face_annotations_json_path) as file_path:
                face_annotations = json.load(file_path)
        print("Processing face annotations JSON file.")
        images_to_exclude = set([x["url"] for x in face_annotations if len(x["bboxes"]) > 0])
        print("Creating valid image list.")
        all_images = set()
        for dirpath, _, fnames in os.walk(self.dataset_path):
            for fname in fnames:
                path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(path, self.dataset_path)
                if fname.endswith(".JPEG") and rel_path not in images_to_exclude:
                    all_images.add(ImageDetails(name=fname, path=path, hash=None, rel_path=rel_path))
        print("Image list created.")
        all_images = list(all_images)
        print(f"Number of images: {len(all_images)}")
        if self.duplicates_path:
            with open(
                self.duplicates_path,
                "r",
            ) as f:
                duplicate_image_paths = json.load(f)["duplicate_images"]
            all_images = [im for im in all_images if im.rel_path not in duplicate_image_paths]
            print(f"Number of images after duplicate removal:{len(all_images)}")
        return all_images
