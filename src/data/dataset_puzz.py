
import pandas as pd
from pathlib import Path
import os
from PIL import Image
import torch
import torch.nn.functional as nnFun
from torch.utils.data import Dataset
import numpy as np

def load_pos(folder_name):
    required_cols = ['rpf', 'x', 'y', 'rot']
    all_data = []
    for file in Path(folder_name).iterdir():
        if file.is_file() and file.suffix.lower() == '.txt':
            try: 
                df = pd.read_csv(file, sep = ',', skipinitialspace = True, encoding = 'utf-8')
            except:
                df = pd.read_csv(file, sep = ',', skipinitialspace = True, encoding = 'gbk')

            # extract columns
            df_sub = df.reindex(columns = required_cols)
            df_sub['file_name'] = file.stem
            all_data.append(df_sub)
    if all_data:
        final_df = pd.concat(all_data, ignore_index = True)
        # keep the first value if rpf repeated in the same file
        final_df = final_df.drop_duplicates(subset=['file_name', 'rpf'], keep = 'first')
        # set index file name and rpf
        final_df.set_index(['file_name', 'rpf'], inplace = True)
    else:
        print("no data")
    return final_df


def load_fragment_path(img_dir, max_num = -1):
    samples = []
    paths = os.listdir(img_dir)
    if max_num > -1:
        paths = paths[: max_num]        
    for img_path in paths:
        frags_dir = os.path.join(img_dir, img_path)
        if not os.path.isdir(frags_dir):
            continue
        frag_imgs = [frag for frag in os.listdir(frags_dir) if frag.lower().endswith('.png')]
        if not frag_imgs:
            continue
        samples.append({
            'frag_files':[os.path.join(frags_dir, frag) for frag in frag_imgs],
            'sample_id': img_path
        })
    return samples

def calculate_new_pos(sample_img, preload, pos_imgs):
    frag_tensors = []
    frag_coords = []
    canvas_w, canvas_h = 0, 0
    img_name = Path(sample_img['sample_id']).stem
    print(img_name)
    if img_name not in pos_imgs.index:
        return None
    pos_frags = pos_imgs.loc[img_name]
    fragments = []
    for frag_path in sample_img['frag_files']:
        frag_img = Image.open(frag_path).convert('RGBA')
        frag_name = os.path.basename(frag_path)
        if frag_name not in pos_frags.index:
            return None
        orig_x, orig_y = pos_frags.loc[frag_name,['x','y']]
        frag_bbox = frag_img.getbbox()
        if frag_bbox is None:
            continue
        left, top, right, bottom = frag_bbox
        # cropped the image
        frag_cropped = frag_img.crop(frag_bbox)
        # new position = ori - offset
        new_x = orig_x + left
        new_y = orig_y + top
        
        fragments.append({
            'name': frag_name,
            'orig_img': frag_img,
            'cropped_img': frag_cropped,
            'orig_pos': (orig_x, orig_y),
            'new_pos': (new_x, new_y),
            'orig_size': frag_img.size,
            'crop_size': frag_cropped.size,
            'crop_offset':(left, top)
        })
    return fragments


class PuzzleDataset(Dataset):
    def __init__(self, imgs_dir, ground_dir, patch_size = 64, stride = None, expand = 8,
                max_frag_size = 512, min_frag_size = 80, preload = True):
        """
            Args:
                imgs_dir: the root dirtory of the fragment folder
                ground_dir: the dirtory of the ground truth
                patch_size: the size of the patch the fragment split into
                stride: the step of the sliding window
                expand: the overlaped pixels for restoring
                max_frag_size: the max size of the fragment, othersize, downsample
                min_frag_size: the min size of the fragment, othersize, upsample
                preload: preload image
        """
        self.patch_size = patch_size
        self.stride = stride if stride else patch_size
        self.expand = expand
        self.max_frag_size = max_frag_size
        self.min_frag_size = min_frag_size
        self.new_patch_size = patch_size + 2 * expand

        # sample
        self.samples = load_fragment_path(imgs_dir, 10)
        self.positions = load_pos(ground_dir)

        # load imgs
        self.preload = preload
        self.frag_tensors_list = [] # fragments of each sample
        self.frag_coords_list = [] # position of each fragment
        self.image_sizes = [] # canvas of each sample

        for sample in self.samples:
            fragments_data = calculate_new_pos(sample, preload, self.positions)
            if fragments_data is None:
                continue
            frag_tensors = []
            frag_coords = []
            canvas_w, canvas_h = 0, 0
            for frag_data in fragments_data:
                cropped_frag = frag_data['cropped_img']
                cropped_frag_np = np.array(cropped_frag)
                rgb_tensor = torch.tensor(cropped_frag_np, dtype = torch.float32).permute(2, 0, 1) / 255.0
                new_x, new_y = frag_data['crop_size']
                w, h = cropped_frag.size
                canvas_w = max(canvas_w, new_x + w)
                canvas_h = max(canvas_h, new_y + h)
                frag_tensors.append(rgb_tensor)
                frag_coords.append((new_x, new_y, w, h))
            self.frag_tensors_list.append(frag_tensors)
            self.frag_coords_list.append(frag_coords)
            self.image_sizes.append((canvas_h, canvas_w))

    
    def __len__(self):
        return len(self.frag_tensors_list)

    def __getitem__(self, idx):
        frag_tensors = self.frag_tensors_list[idx]
        frag_coords = self.frag_coords_list[idx]
        height, width = self.image_sizes[idx]
        pad = self.expand
        patch_height, patch_width = self.patch_size, self.patch_size
        step = self.stride

        all_patches = []
        all_patch_centers = []
        frag_centers = []
        patch_frag_idx = []

        # upsample
        min_size = self.patch_size + 2 * self.expand
        for i in range(len(frag_tensors)):
            frag_tensor = frag_tensors[i]
            pos_x, pos_y, f_width, f_height = frag_coords[i]
            if f_width < min_size or f_height < min_size:
                scale = max(min_size / f_height, min_size / f_width)
                new_h = int(f_height * scale) + 1
                new_w = int(f_width * scale) + 1
                frag_tensor = nnFun.interpolate(
                    frag_tensor.unsqueeze(0),
                    size = (new_h, new_w),
                    mode = 'bilinear',
                    align_corners = False                
                ).squeeze(0)
                f_width, f_height = new_w, new_h
                frag_tensors[i] = frag_tensor
                frag_coords[i] = (pos_x, pos_y, f_width, f_height)
                
        for frag_idx, (frag_tensor, (pos_x, pos_y, f_width, f_height)) in enumerate(zip(frag_tensors, frag_coords)):
            frag_centers.append(((pos_y + f_height / 2) / height, (pos_x + f_width / 2) / width))
            # padding
            padded = nnFun.pad(frag_tensor, (pad, pad, pad, pad), mode = 'reflect')
            # sliding window
            for y in range(0, f_height - patch_height + 1, step):
                for x in range(0, f_width - patch_width + 1, step):
                    patch = padded[:, y:y + patch_height + 2 * pad, x:x + patch_width + 2 * pad]
                    valid_center = patch[:, pad:pad + patch_height, pad:pad + patch_width]
                    if valid_center.mean() < 0.05 or valid_center.var() < 0.001:
                        continue
                    all_patches.append(patch)
                    patch_center_x = (pos_x + x + patch_width / 2) / width
                    patch_center_y = (pos_y + y + patch_height / 2) / height
                    all_patch_centers.append((patch_center_x, patch_center_y))
                    patch_frag_idx.append(frag_idx)
        if len(all_patches) == 0:
            return self.__getitem__((idx + 1) % len(self))

        patches = torch.stack(all_patches) # [N, 3, new_patch_size, new_patch_size]
        patch_centers = torch.tensor(all_patch_centers, dtype = torch.float32) #[N, 2]
        frag_centers = torch.tensor(frag_centers, dtype = torch.float32) #[M, 2]
        patch_frag_idx = torch.tensor(patch_frag_idx, dtype = torch.long) #[N]

        #offset
        frag_centers_expand = frag_centers[patch_frag_idx] #[N, 2]
        offset_labels = patch_centers - frag_centers_expand

        return{
            'patches': patches,
            'patch_centers': patch_centers,
            'frag_centers': frag_centers,
            'patch_frag_idx': patch_frag_idx,
            'offset_labels': offset_labels,
            'num_patches': len(patches),
            'num_frags': len(frag_centers),
            'image_size':(height, width),
        }
        
def collate_puzzle(batches):
    max_patches = max([batch['num_patches'] for batch in batches])
    max_frag = max([batch['num_frags'] for batch in batches])
    batch_size = len(batches)

    patches_list, patch_center_list, frag_center_list, patch_idx_list, off_list = [], [], [], [], []
    masks_patch = []
    masks_frag = []
    image_sizes = []
    for batch in batches:
        N = batch['num_patches']
        M = batch['num_frags']
        image_sizes.append(batch['image_size'])

        patch = batch['patches']
        patch_centers = batch['patch_centers']
        frag_centers = batch['frag_centers']
        patch_idx = batch['patch_frag_idx']
        offset = batch['offset_labels']
        
        if N < max_patches:
            pad_n = max_patches - N
            patch = torch.cat([patch, torch.zeros(pad_n, *patch.shape[1:], dtype = patch.dtype)], dim = 0)
            patch_centers = torch.cat([patch_centers, torch.zeros(pad_n, 2, dtype = patch_centers.dtype)], dim = 0)
            patch_idx = torch.cat([patch_idx, torch.full((pad_n,), -1, dtype = patch_idx.dtype)], dim = 0)
            offset = torch.cat([offset, torch.zeros(pad_n, 2, dtype = offset.dtype)], dim = 0)
        patches_list.append(patch)
        patch_center_list.append(patch_centers)
        patch_idx_list.append(patch_idx)
        off_list.append(offset)
        
        if M < max_frag:
            pad_m = max_frag - M
            frag_centers = torch.cat([frag_centers, torch.zeros(pad_m, 2, dtype = frag_centers.dtype)], dim = 0)
        frag_center_list.append(frag_centers)

        mask_patch = torch.zeros(max_patches, dtype = torch.bool)
        mask_patch[:N] = True
        masks_patch.append(mask_patch)

        mask_frag = torch.zeros(max_frag, dtype = torch.bool)
        mask_frag[:M] = True
        masks_frag.append(mask_frag)
    return{
        'patches': torch.stack(patches_list),
        'patch_centers': torch.stack(patch_center_list),
        'frag_centers': torch.stack(frag_center_list),
        'patch_frag_idx': torch.stack(patch_idx_list),
        'offset_labels': torch.stack(off_list),
        'mask_patch': torch.stack(masks_patch),
        'mask_frag': torch.stack(masks_frag),
        'num_patches': torch.tensor([batch['num_patches'] for batch in batches]),
        'num_frags': torch.tensor([batch['num_frags'] for batch in batches]),
        'image_size':image_sizes,
    }
                