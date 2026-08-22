""" image_utils.py for the image processiong."""

from pathlib import Path
import numpy as np
import cv2
# default extension support
_DEFAULT_IMAGE_EXTS = {
    '.png'

}

def get_image_paths(directory: str, 
                    recursive: bool = False, 
                    extensions: set[str]| None = None)->list[Path]:
    """ get the path list of the target folder

    Args:
        directory: the directory path
        recursive: include child folder
        extensions: extention set
    Returns:
        the image path list with the defined extension
    Raises:
        ValueError: the folder not found
    """
    # print(f"debug directory type={type(directory)}, value={repr(directory)}")
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"the folder not found:{directory}")

    exts = extensions or _DEFAULT_IMAGE_EXTS
    normalized_exts = {
        ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
        for ext in exts
    }

    iterator_paths = dir_path.rglob('*') if recursive else dir_path.glob('*')

    return [
        f for f in iterator_paths
            if f.is_file() and f.suffix.lower() in normalized_exts
    ]

def get_bgr_img(img_raw: np.ndarray) -> np.ndarray|None:

    if len(img_raw.shape) == 3 and img_raw.shape[2] == 4:
        img_bgr = img_raw[:, :, :3]
    else:
        img_bgr = img_raw
    return img_bgr 


def get_mask(img: np.ndarray) -> np.ndarray | None:

    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        return np.where(alpha > 0, 255, 0).astype(np.uint8)
    else:
        return None


def load_img(img_raw: str|Path|np.ndarray) -> np.ndarray|None:

    if isinstance(img_raw, Path):
        img_raw = str(img_raw)
         
    if isinstance(img_raw, str):
        img_alpha = cv2.imread(img_raw, cv2.IMREAD_UNCHANGED)
        if img_alpha is None:
            print(f"warning: image not found {img_raw}, ignored")
            return None
    elif isinstance(img_raw, np.ndarray):
        img_alpha = img_raw
    else:
        print(f"warning: the type not supported: {type(img_raw)}, ignored")
        return None
    
    img_bgr = get_bgr_img(img_alpha)
    img_mask = get_mask(img_alpha)

    return img_bgr, img_mask

def imgs_flatten(arr_bgr, arr_mask):

    imgs_reshape = arr_bgr.reshape(-1, 1, arr_bgr[0].shape[2])
    imgs_mask_reshape = arr_mask.reshape(-1, 1, 1)
    return imgs_reshape, imgs_mask_reshape

def load_imgs(img_paths: list[str|Path|np.ndarray]):

    imgs_bgr = []
    imgs_mask = []
    for img_path in img_paths:
        img_bgr, img_mask = load_img(img_path)
        imgs_bgr.append(img_bgr)
        imgs_mask.append(img_mask)

    arr_bgr = np.array(imgs_bgr)
    arr_mask = np.array(imgs_mask)

    return arr_bgr, arr_mask 


def load_imgs_flatten(img_paths: list[str|Path|np.ndarray]):

    imgs_bgr = []
    imgs_mask = []
    for img_path in img_paths:
        img_bgr, img_mask = load_img(img_path)
        img_reshape = img_bgr.reshape(-1, img_bgr.shape[2])
        img_mask_reshape = img_mask.reshape(-1, 1)

        imgs_bgr.append(img_reshape)
        imgs_mask.append(img_mask_reshape)

    arr_bgr = np.array(imgs_bgr)
    arr_mask = np.array(imgs_mask)

    return arr_bgr, arr_mask


