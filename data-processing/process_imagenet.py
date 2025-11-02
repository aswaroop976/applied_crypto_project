"""Process imagenet data.

This files extracts the images from the ImageNet into the folder structure.

- top_folder
    - train
        - images
    - val
        - images
    - test
        - images
"""
import os
from posixpath import splitext
import tarfile
import tqdm


def extract_tar(tarfile_path, directory_path, keep_original=True):
    """Extract tarfile into the directory_path."""
    tar = tarfile.open(tarfile_path)
    os.makedirs(directory_path, exist_ok=True)
    tar_members = tar.getmembers()
    for member in tqdm.tqdm(tar_members):
        tar.extract(member, path=directory_path)
    tar.close()
    if not keep_original:
        os.remove(tarfile_path)

def extract_tars_recursively(input_directory, output_directory, keep_original=True):
    """Extract all the tars in the folder recursively to the directory.

    Note:
        - A tar is extracted a folder (by name of the tar) created in specified directory
            under the relative path of the tar w.r.t the input directory.
    """
    for (dirpath, _, fnames) in os.walk(input_directory):
        for fname in fnames:
            if fname.endswith('.tar'):
                extract_from = os.path.join(dirpath, fname)
                extract_to = os.path.join(
                    output_directory, os.path.relpath(dirpath, input_directory), 
                    os.path.splitext(fname)[0])
                print(f"Extracting from {extract_from} to {extract_to}")
                extract_tar(
                    extract_from,
                    extract_to,
                    keep_original=keep_original
                )


if __name__ == "__main__":
    # extract_tar("ILSVRC2012_img_val.tar", './datasets/imagenet/val/')
    # extract_tar("ILSVRC2012_img_test_v10102019.tar", './datasets/imagenet/test/')
    # extract_tar("ILSVRC2012_img_train.tar", './datasets/imagenet/train/')
    extract_tars_recursively('./datasets/imagenet/train/', './datasets/imagenet/train/')
