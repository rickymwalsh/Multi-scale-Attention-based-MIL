from collections import defaultdict

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torch.utils.data import Dataset

import os 
import h5py

import json
import ast

def convert_to_float_list(string_list):
    # Strip brackets and split by comma, then convert to floats
    return [float(value) for value in string_list.strip('[]').split(', ')]
    

def filter_bounding_boxes(finding_categories, boxes, label_name):

    finding_categories_list = json.loads(finding_categories)
    
    cleaned_categories = []
    for sublist in finding_categories_list:

        cleaned_sublist = ast.literal_eval(sublist)
        
        # Append the cleaned sublist as a whole if it contains more than one element
        cleaned_categories.append(cleaned_sublist if isinstance(cleaned_sublist, list) else [cleaned_sublist])

    # Get bounding box lists
    xmin_list, ymin_list, xmax_list, ymax_list = boxes

    # Create an empty list to store filtered bounding boxes
    filtered_boxes = []
    
    # Iterate over the bounding boxes and associated findings
    for i in range(len(xmin_list)):
        # Extract the finding categories for this box
        box_categories = cleaned_categories[i]
        
        # Check if label_name is present
        if any(category == label_name for category in (box_categories if isinstance(box_categories, list) else [box_categories])):
                
            xmin = xmin_list[i]
            xmax = xmax_list[i]
            ymin = ymin_list[i]
            ymax = ymax_list[i]
            
            # Swap xmin and xmax if xmin > xmax
            if float(xmin) > float(xmax):
                xmin, xmax = xmax, xmin
    
            # Swap ymin and ymax if ymin > ymax
            if float(ymin) > float(ymax):
                ymin, ymax = ymax, ymin
                
            filtered_boxes.append([xmin, ymin, xmax, ymax])

    return filtered_boxes
    
class BagDataset(Dataset):
    """
    Dataset for loading images and extracting patches
    """
    def __init__(self, args, df, transform=None):
        self.args = args
        self.df = df
        self.dir_path = args.data_dir / args.img_dir
        self.dataset = args.dataset
        self.transform = transform
        
        print(transform)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        data = self.df.iloc[idx]
        img_path = self.dir_path / str(self.df.iloc[idx]['patient_id']) / str(self.df.iloc[idx]['image_id'])
            
        if (self.args.arch.lower() == "upmc_breast_clip_det_b5_period_n_ft" or
            self.args.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_ft" or
            self.args.arch.lower() == "upmc_breast_clip_det_b5_period_n_lp" or
            self.args.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_lp" or
            self.args.arch.lower() == "upmc_breast_clip_det_b2_period_n_ft" or
            self.args.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_ft" or
            self.args.arch.lower() == "upmc_breast_clip_det_b2_period_n_lp" or
            self.args.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_lp"
        ):
            img = Image.open(img_path).convert('RGB')
            
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        patches, patch_coords, padding = self.transform(img)
            
        return {
        'x': patches.unsqueeze(0), 
        'coords': patch_coords, 
        'padding': padding,
        'patient_id': str(self.df.iloc[idx]['patient_id']),
        'image_id': str(self.df.iloc[idx]['image_id'])
        }
            

def collate_patch_features(batch):
	return {
        'x': torch.stack([item['x'] for item in batch]),
        'coords': np.vstack([item['coords'] for item in batch]),
        'padding_left': [item['padding'][0] for item in batch],  # Gather left padding values
        'padding_right': [item['padding'][1] for item in batch], # Gather right padding values
        'padding_top': [item['padding'][2] for item in batch],  # Gather top padding values
        'padding_bottom': [item['padding'][3] for item in batch],  # Gather bottom padding values
        'patient_id': [item['patient_id'] for item in batch],
        'image_id': [item['image_id'] for item in batch]
    }

class Generic_MIL_Dataset(Dataset):
    def __init__(self, args, df, transform, split='train'):
        self.args = args
        self.df = df

        if args.multi_scale_model is None:
            # single-scale mil models
            self.dir_path = args.data_dir / args.feat_dir / f'patch_size-{args.scales[0]}' if args.feature_extraction == 'offline' else args.data_dir / args.img_dir

        elif args.multi_scale_model == 'msp':
            # multi-scale patch-based mil models
            self.dir_path = args.data_dir / args.feat_dir

        elif args.multi_scale_model in ['fpn', 'backbone_pyramid']:
            self.dir_path = args.data_dir / args.feat_dir / 'multi_scale' if args.feature_extraction == 'offline' else args.data_dir / args.img_dir

        self.split = split
        self.dataset = args.dataset
        self.label = args.label
        self.transform = transform

        self.multi_scale_model = args.multi_scale_model
        self.scales = args.scales
        # Preload training/val features into memory to avoid per-batch file I/O.
        self._feat_cache = None
        self._label_cache = None
        if split in ['train', 'val'] and self.transform is None and args.feature_extraction == 'offline':
            self._preload_features()

    def _ensure_sorted_file(self, bag_dir, feat_pyramid_level=None):
        """Return path to a pre-sorted feature file, creating it if it does not exist.

        The sorted file is saved alongside the original feature file as
        ``{prefix}patch_features_sorted.pt``.  The write is performed via a
        temporary file followed by an atomic ``os.rename()`` so that two
        concurrent processes racing to create the same file cannot produce a
        corrupt result.
        """
        import pathlib
        bag_dir = pathlib.Path(bag_dir)
        prefix = f'{feat_pyramid_level}_' if feat_pyramid_level else ''
        sorted_path = bag_dir / f'{prefix}patch_features_sorted.pt'
        if sorted_path.exists():
            # Check dtype: existing float32 sorted files (from old code) must be
            # recreated as float16 to halve DRAM and PCIe bandwidth.
            try:
                _existing = torch.load(sorted_path, mmap=True, weights_only=True)
            except TypeError:
                _existing = torch.load(sorted_path, weights_only=True)
            if _existing.dtype == torch.float16:
                return sorted_path
            # float32 sorted file found — fall through to recreate as float16

        # Load original features and sort by patch coordinates.
        orig_path = bag_dir / f'{prefix}patch_features.pt'
        x = torch.load(orig_path, weights_only=True)
        coords_path = bag_dir / 'info_patches.h5'
        with h5py.File(coords_path, 'r') as _f:
            bag_coords = np.array(_f['coords'])
        sorted_indices = np.lexsort((bag_coords[:, 0], bag_coords[:, 1]))
        x_sorted = x[sorted_indices].half()  # cast to float16: halves DRAM/PCIe bandwidth

        # Write atomically: another process may win the race, which is fine
        # because both produce the identical tensor.
        tmp_path = sorted_path.with_suffix('.pt.tmp')
        torch.save(x_sorted, tmp_path)
        os.replace(str(tmp_path), str(sorted_path))  # atomic on POSIX
        return sorted_path

    def _load_sorted_mmap(self, bag_dir, feat_pyramid_level=None):
        """Load a pre-sorted feature file as a file-backed mmap tensor.

        Two processes that open the same file with ``mmap=True`` share the
        same physical pages via the OS page cache, halving DRAM bandwidth
        when two training jobs run concurrently.  We intentionally do NOT
        call ``posix_fadvise(DONTNEED)`` here — keeping pages resident is
        what enables sharing.

        Falls back to a regular heap load on PyTorch < 2.1 where the
        ``mmap`` keyword is not available (no sharing benefit, but no crash).
        """
        sorted_path = self._ensure_sorted_file(bag_dir, feat_pyramid_level)
        try:
            return torch.load(sorted_path, mmap=True, weights_only=True)
        except TypeError:
            # mmap kwarg unavailable (PyTorch < 2.1)
            return torch.load(sorted_path, weights_only=True)

    def _preload_features(self):
        """Load all pre-extracted features into contiguous stacked tensors.

        Rather than storing a Python list of per-sample objects, all features
        are copied into a small number of large pre-allocated tensors:

          single-scale : _feat_cache  shape (N, P, D)           float16
          fpn          : _feat_cache  [Tensor(N,P,C4,H,W),
                                       Tensor(N,P,C5,H,W)]      float16
          msp          : _feat_cache  {scale: Tensor(N,P,D)}    float16
          all modes    : _label_cache shape (N,)                 int64

        __getitem__ returns zero-copy views (tensor[idx]) so no data is copied
        at batch time.  Replacing ~5×N Python objects with 2–4 large tensors
        reduces the GC workload during training to near zero.
        """
        N = len(self.df)
        print(f"Preloading {N} samples into memory...")

        # Read first sample to determine shapes and pre-allocate stores.
        row0 = self.df.iloc[0]
        pid0, iid0 = str(row0['patient_id']), str(row0['image_id'])

        if self.multi_scale_model is None:
            x0 = self._load_sorted_mmap(self.dir_path / pid0 / iid0)
            store = torch.empty(N, *x0.shape, dtype=torch.float16)
            store[0].copy_(x0)
            del x0

        elif self.multi_scale_model == 'msp':
            store = {}
            for scale in self.scales:
                x0 = self._load_sorted_mmap(
                    self.dir_path / f'patch_size-{scale}' / pid0 / iid0)
                store[scale] = torch.empty(N, *x0.shape, dtype=torch.float16)
                store[scale][0].copy_(x0)
                del x0

        elif self.multi_scale_model in ['fpn', 'backbone_pyramid']:
            c4_0 = self._load_sorted_mmap(self.dir_path / pid0 / iid0, feat_pyramid_level='C4')
            c5_0 = self._load_sorted_mmap(self.dir_path / pid0 / iid0, feat_pyramid_level='C5')
            store = [
                torch.empty(N, *c4_0.shape, dtype=torch.float16),
                torch.empty(N, *c5_0.shape, dtype=torch.float16),
            ]
            store[0][0].copy_(c4_0)
            store[1][0].copy_(c5_0)
            del c4_0, c5_0

        label_store = torch.empty(N, dtype=torch.long)
        label_store[0] = int(row0[self.label])

        # Fill remaining samples.
        for idx in range(1, N):
            row = self.df.iloc[idx]
            pid, iid = str(row['patient_id']), str(row['image_id'])

            if self.multi_scale_model is None:
                store[idx].copy_(self._load_sorted_mmap(self.dir_path / pid / iid))

            elif self.multi_scale_model == 'msp':
                for scale in self.scales:
                    store[scale][idx].copy_(
                        self._load_sorted_mmap(
                            self.dir_path / f'patch_size-{scale}' / pid / iid))

            elif self.multi_scale_model in ['fpn', 'backbone_pyramid']:
                store[0][idx].copy_(
                    self._load_sorted_mmap(self.dir_path / pid / iid, feat_pyramid_level='C4'))
                store[1][idx].copy_(
                    self._load_sorted_mmap(self.dir_path / pid / iid, feat_pyramid_level='C5'))

            label_store[idx] = int(row[self.label])

            if (idx + 1) % 2000 == 0:
                print(f"  Preloaded {idx + 1}/{N} samples")

        self._feat_cache = store
        self._label_cache = label_store

        print(f"Preloading complete ({N} samples)")
        import gc as _gc
        _gc.collect()
        _gc.freeze()

    def __len__(self):
        return len(self.df)

    def load_data(self, bag_dir, feat_pyramid_level = None):
        """
        Load pre-extracted instance features and sort them by patch coordinates.

        Args:
            bag_dir (Path or str): Directory containing instance features and metadata.
            feat_pyramid_level (str or None): Specific pyramid level (e.g., 'C4', 'C5') for multi-scale features.

        Returns:
            torch.Tensor: Sorted instance features tensor.
        """

        # Load patch features tensor
        if feat_pyramid_level is None:
            feat_path = os.path.join(bag_dir, 'patch_features.pt')
        else:
            feat_path = os.path.join(bag_dir, f'{feat_pyramid_level}_patch_features.pt')

        x = torch.load(feat_path, weights_only=True)

        # Load patch coordinates from HDF5 file
        coords_path = os.path.join(bag_dir, 'info_patches.h5')
        with h5py.File(coords_path, 'r') as file:
            bag_coords = np.array(file['coords'])

        # Drop the coords file from page cache (h5py reads into heap numpy array,
        # so this is always safe immediately after the with-block closes the file).
        try:
            with open(coords_path, 'rb') as _f:
                os.posix_fadvise(_f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass

        sorted_indices = np.lexsort((bag_coords[:, 0], bag_coords[:, 1]))  # Sort by y, then x

        # Fancy indexing always allocates a new heap-backed tensor (never a view).
        # This ensures x is in anonymous heap memory regardless of whether torch.load()
        # returned an mmap-backed tensor or a regular heap tensor.
        x = x[sorted_indices]

        # Drop the feature file from the kernel page cache NOW, after x[sorted_indices]
        # has copied the data to heap and the original mmap storage has been freed.
        # Calling DONTNEED before the indexing would evict the mmap pages and cause
        # page faults on every element access during the copy — exactly the wrong order.
        # Doing it here prevents val/test on-demand reads from accumulating in the page
        # cache and competing with the preloaded train cache for physical RAM.
        try:
            with open(feat_path, 'rb') as _f:
                os.posix_fadvise(_f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass

        return x


    def __getitem__(self, idx):

        # Fast path: return preloaded features
        if self._feat_cache is not None:
            if isinstance(self._feat_cache, torch.Tensor):
                x = self._feat_cache[idx]               # zero-copy view (P, D)
            elif isinstance(self._feat_cache, list):
                x = [t[idx] for t in self._feat_cache]  # list of views
            else:  # dict (msp)
                x = {k: v[idx] for k, v in self._feat_cache.items()}
            label = self._label_cache[idx]
            return {'x': x, 'y': label}

        data = self.df.iloc[idx]

        if self.transform is not None:  # Online instance feature extraction mode

            bag_dir = self.dir_path / str(self.df.iloc[idx]['patient_id']) / str(self.df.iloc[idx]['image_id']) if self.args.dataset != 'ddsm' else data['image_file_path']

            img = Image.open(bag_dir).convert('RGB')

            x, bag_coords, padding = self.transform(img)

        else: # Offline instance feature extraction mode: load pre-extracted features from disk

            if self.multi_scale_model is None:

                bag_dir = self.dir_path / str(self.df.iloc[idx]['patient_id']) / str(self.df.iloc[idx]['image_id'])

                x = self.load_data(bag_dir)

            elif self.multi_scale_model == 'msp':
                # Multi-scale pyramid: load features for all scales

                bag_dir_small = self.dir_path / f'patch_size-{self.scales[0]}' / str(self.df.iloc[idx]['patient_id']) / str(self.df.iloc[idx]['image_id'])
                x_small  = self.load_data(bag_dir_small)

                bag_dir_medium = self.dir_path / f'patch_size-{self.scales[1]}' / str(self.df.iloc[idx]['patient_id']) / str(self.df.iloc[idx]['image_id'])
                x_medium = self.load_data(bag_dir_medium)

                bag_dir_large = self.dir_path / f'patch_size-{self.scales[2]}' / str(self.df.iloc[idx]['patient_id']) / str(self.df.iloc[idx]['image_id'])
                x_large = self.load_data(bag_dir_large)

                x = {
                    self.scales[0]: x_small,
                    self.scales[1]: x_medium,
                    self.scales[2]: x_large
                }

            elif self.multi_scale_model in ['fpn', 'backbone_pyramid']:
                # Load specific feature pyramid levels

                dir_path = self.dir_path / str(self.df.iloc[idx]['patient_id']) / str(self.df.iloc[idx]['image_id'])

                c4_feats = self.load_data(dir_path, feat_pyramid_level = 'C4')

                c5_feats = self.load_data(dir_path, feat_pyramid_level = 'C5')

                x = [c4_feats, c5_feats]

        return {
        'x': x,
        'y': torch.tensor(data[self.label], dtype=torch.long)
        }
            

def collate_MIL_patches(batch):
    """
    Custom collate function for MIL datasets to batch variable input formats
    """

    if isinstance(batch[0]['x'], dict): 
        ## Multi-scale patch-based MIL model: batch each scale separately
        x = {scale: torch.stack([item['x'][scale] for item in batch]) 
               for scale in batch[0]['x']
              } 
        
    elif isinstance(batch[0]['x'], list): 
        # FPN-based MIL model: batch each feature level separately
        x = [torch.stack([item['x'][i] for item in batch], dim=0) for i in range(len(batch[0]['x']))]
        
    else:
        # Single-scale MIL model: batch all tensors
        x = torch.stack([item['x'] for item in batch])

    return {
        'x': x, 
        'y': torch.from_numpy(np.array([item["y"] for item in batch], dtype=np.float32))
    }


class Generic_MIL_Dataset_Detection(Dataset):
    def __init__(self, args, df, transform=None):
        self.args = args
        self.df = df

        # Determine if feature extraction is online (from images) or offline (from features)
        self.feature_extraction = True if args.feature_extraction == 'online' else False 

        if self.feature_extraction: 
            # For online feature extraction, only image directory is needed
            self.img_dir = args.data_dir / args.img_dir
        else: 
            # For offline feature extraction, set feature directory based on MIL model type
            if args.multi_scale_model == 'msp': 
                # multi-scale patch-based MIL model 
                self.feat_dir = args.data_dir / args.feat_dir
            elif args.multi_scale_model in ['fpn', 'backbone_pyramid']: 
                # FPN-based MIL model 
                self.feat_dir = args.data_dir / args.feat_dir / 'multi_scale'
            else:
                # single-scale MIL model 
                self.feat_dir = args.data_dir / args.feat_dir / f'patch_size-{args.scales[0]}'
                    
            self.img_dir = args.data_dir / args.img_dir
            
        self.dataset = args.dataset
        self.transform = transform
        self.label = args.label
        
        self.multi_scale_model = args.multi_scale_model
        self.scales = args.scales 
        
        if args.label == 'Suspicious_Calcification':
            self.label_type = 'Suspicious Calcification' 
        else: 
            self.label_type = args.label 

        self.image_dict = self._generate_image_dict()

    def _generate_image_dict(self):
        """
        Builds a dictionary holding image paths, feature paths (if offline),
        bounding boxes and labels.
        """

        # Initialize dictionary keys depending on feature extraction mode and MIL model type
        if self.feature_extraction: 
            image_dict = {"img_path": [], "boxes": [], "labels": []}
        else: 
            if self.multi_scale_model == 'msp':
                image_dict = {"img_path": [], 
                              "feat_path_small": [], 
                              "feat_path_medium": [], 
                              "feat_path_large": [], 
                              "boxes": [], 
                              "labels": []}
                
            else:    
                image_dict = {"img_path": [], "feat_path": [], "boxes": [], "labels": []}
            
        for idx, row in self.df.iterrows():
            
            finding_categories = row["finding_categories"]

            # Extract bounding boxes, convert coordinates to floats
            boxes = [
                convert_to_float_list(row['resized_xmin']),
                convert_to_float_list(row['resized_ymin']),
                convert_to_float_list(row['resized_xmax']),
                convert_to_float_list(row['resized_ymax'])
            ]

            # Filter bounding boxes based on label type
            boxes = filter_bounding_boxes(finding_categories, boxes, self.label_type) 

            # If any valid bounding boxes exist, append data to image_dict
            if any(boxes): 
                img_path = self.img_dir / str(self.df.iloc[idx]['patient_id']) / str(self.df.iloc[idx]['image_id'])
                image_dict["img_path"].append(img_path) 
                image_dict["boxes"].append(boxes)
                image_dict["labels"].append(1) # Label is 1 indicating presence of the finding

                # For offline extraction, append corresponding feature paths based on MIL model type
                if not self.feature_extraction:
                    if self.multi_scale_model == 'msp':
                        image_dict["feat_path_small"].append(
                            self.feat_dir / f'grid_size-{int(512/self.scales[0])}x{int(512/self.scales[0])}' / str(self.df.iloc[idx]['patient_id']) / str(self.df.iloc[idx]['image_id'])
                        )

                        image_dict["feat_path_medium"].append(
                            self.feat_dir / f'grid_size-{int(512/self.scales[1])}x{int(512/self.scales[1])}' / str(self.df.iloc[idx]['patient_id']) / str(self.df.iloc[idx]['image_id'])
                        )

                        image_dict["feat_path_large"].append(
                            self.feat_dir / f'grid_size-{int(512/self.scales[2])}x{int(512/self.scales[2])}' / str(self.df.iloc[idx]['patient_id']) / str(self.df.iloc[idx]['image_id'])
                        )
                        
                    else: 
                        image_dict["feat_path"].append(
                            self.feat_dir / str(self.df.iloc[idx]['patient_id']) / str(self.df.iloc[idx]['image_id'])
                        )
                    
        return image_dict

    def load_data(self, bag_dir, feat_pyramid_level = None): 
        """
        Loads pre-extracted patch features and associated patch info from disk.
        
        Args:
            bag_dir: Directory path where patch features and info are stored.
            feat_pyramid_level: Optional; specifies pyramid level for multi-scale features.

        Returns:
            x: Tensor of patch features sorted spatially.
            bag_coords: Numpy array of patch coordinates sorted spatially.
            bag_info: Dictionary of additional info about the patch coordinates.
        """

        # Load the patch features tensor
        if feat_pyramid_level is None:
            # patch-based MIL models
            x = torch.load(os.path.join(bag_dir, 'patch_features.pt'), weights_only=True)
        else:
            # FPN-based MIL model
            x = torch.load(os.path.join(bag_dir, f'{feat_pyramid_level}_patch_features.pt'), weights_only=True)
        
        # Load patch coordinate info from H5 file
        with h5py.File(os.path.join(bag_dir, 'info_patches.h5'), 'r') as file:
            
            # Getting the data
            bag_coords = np.array(file['coords'])
            bag_info = {key: file['coords'].attrs[key] for key in file['coords'].attrs.keys()}

        # Sort coordinates (and corresponding features) 
        sorted_indices = np.lexsort((bag_coords[:, 0], bag_coords[:, 1]))  # Sort by y, then x

        bag_coords = bag_coords[sorted_indices]
        x = x[sorted_indices]
        
        return x, bag_coords, bag_info
        
    def __len__(self):
        return len(self.image_dict["img_path"])

    def __getitem__(self, idx):

        img_path = self.image_dict["img_path"][idx]
        
        label = self.image_dict["labels"][idx]

        boxes = self.image_dict["boxes"][idx]

        boxes = np.array(boxes)

        if self.feature_extraction: # Online feature extraction from raw image
            img = Image.open(img_path).convert('RGB')
                
            x, bag_coords, padding = self.transform(img)

            # Unpack padding (left, right, top, bottom)
            padding_left, padding_right, padding_top, padding_bottom = padding
                
            if self.multi_scale_model == 'msp': 
    
                bag_info = {}
                    
                for idx, patch_size in enumerate(self.scales):
                        
                    bag_info[patch_size] = {
                                'patch_size': patch_size,  
                                'step_size': patch_size - int(patch_size * self.args.overlap[idx]),
                            }

                bag_info['img_height'] = self.args.img_size[0] + padding_top + padding_bottom
                bag_info['img_width']= self.args.img_size[1] + padding_left + padding_right
                bag_info['img_dir'] = img_path
        
            else: 

                patch_size = self.args.scales[0] if self.multi_scale_model is None else self.args.patch_size
                    
                bag_info = {
                        'patch_size': patch_size, 
                        'step_size': patch_size - int(patch_size * self.args.overlap[0]), 
                        'img_height': self.args.img_size[0] + padding_top + padding_bottom,
                        'img_width': self.args.img_size[1] + padding_left + padding_right,
                        'img_dir': img_path
                    }

            # Adjust bounding boxes by adding padding offsets
            if len(boxes) > 0:
                boxes[:, 0] += padding_left  # Adjust xmin
                boxes[:, 1] += padding_top   # Adjust ymin
                boxes[:, 2] += padding_left  # Adjust xmax
                boxes[:, 3] += padding_top   # Adjust ymax
    
        else: # Offline feature extraction from pre-extracted features

            if self.multi_scale_model is None: 
                # single-scale patch-based (ssp)-mil models 

                feat_path = self.image_dict["feat_path"][idx]
                
                x, bag_coords, bag_info = self.load_data(feat_path)

                # Adjust bounding boxes by padding info stored in bag_info
                if len(boxes) > 0:
                    boxes[:, 0] += bag_info['padding_left'] # Adjust xmin
                    boxes[:, 1] += bag_info['padding_top'] # Adjust ymin
                    boxes[:, 2] += bag_info['padding_left'] # Adjust xmax
                    boxes[:, 3] += bag_info['padding_top'] # Adjust ymax
                    
            elif self.multi_scale_model == 'msp': 
                # multi-scale patch-based (msp)-mil models 

                # Load features for each scale separately
                feat_path_small = self.image_dict["feat_path_small"][idx]
                x_small, bag_coords_small, bag_info_temp = self.load_data(feat_path_small) 
                    
                feat_path_medium = self.image_dict["feat_path_medium"][idx]
                x_medium, bag_coords_medium, _ = self.load_data(feat_path_medium)
                    
                feat_path_large = self.image_dict["feat_path_large"][idx]
                x_large, bag_coords_large, _ = self.load_data(feat_path_large)

                x = torch.stack([x_small, x_medium, x_large], dim=-1) 
                                        
                bag_coords = {
                        self.scales[0]: bag_coords_small,
                        self.scales[1]: bag_coords_medium,
                        self.scales[2]: bag_coords_large
                    }
        
                bag_info = {}
                
                for idx, patch_size in enumerate(self.scales):
                    bag_info[patch_size] = {
                                'patch_size': patch_size,  
                                'step_size': patch_size - int(patch_size * self.args.overlap[idx]),
                            }
                    
                bag_info['img_height'] = bag_info_temp['img_height']
                bag_info['img_width'] = bag_info_temp['img_width']
                bag_info['img_dir'] = img_path

                # Adjust bounding boxes with padding info
                if len(boxes) > 0:
                    boxes[:, 0] += bag_info_temp['padding_left'] # Adjust xmin
                    boxes[:, 1] += bag_info_temp['padding_top']   # Adjust ymin
                    boxes[:, 2] += bag_info_temp['padding_left']  # Adjust xmax
                    boxes[:, 3] += bag_info_temp['padding_top']   # Adjust ymax

            elif self.multi_scale_model in ['fpn', 'backbone_pyramid']: 
                # fpn-based mil models 
                
                # Load features at different pyramid levels (e.g., C4, C5)
                feat_path = self.image_dict["feat_path"][idx]

                c4_feats, bag_coords, bag_info = self.load_data(feat_path, feat_pyramid_level = 'C4')
                
                c5_feats, bag_coords, bag_info = self.load_data(feat_path, feat_pyramid_level = 'C5')

                x = [c4_feats, c5_feats] 

                bag_info['img_dir'] = img_path

                # Adjust bounding boxes with padding info
                if len(boxes) > 0:
                    boxes[:, 0] += bag_info['padding_left']  # Adjust xmin
                    boxes[:, 1] += bag_info['padding_top']  # Adjust ymin
                    boxes[:, 2] += bag_info['padding_left']  # Adjust xmax
                    boxes[:, 3] += bag_info['padding_top']   # Adjust ymax
                
        return {
            'x': x,
            'y': torch.tensor(label, dtype=torch.long),
            'coords': bag_coords, 
            'bag_info':bag_info,
            'boxes': boxes
        }


def collate_MIL_patches_detection(batch):

    if isinstance(batch[0]['x'], dict): 
        x = {scale: torch.stack([item['x'][scale] for item in batch]) 
               for scale in batch[0]['x']
              } 

    elif isinstance(batch[0]['x'], list): 
        x = [torch.stack([item['x'][i] for item in batch], dim=0) for i in range(len(batch[0]['x']))]
    
    else:
        x = torch.stack([item['x'] for item in batch])

    if isinstance(batch[0]['coords'], dict): 
        coords = {scale: np.vstack([item['coords'][scale] for item in batch]) 
                   for scale in batch[0]['coords']
                  }
    else: 
        coords = np.vstack([item['coords'] for item in batch])
        
    return {
        'x': x,
        'y': torch.from_numpy(np.array([item["y"] for item in batch], dtype=np.float32)),
        'coords': coords,
        'bag_info': [item['bag_info'] for item in batch],
        'boxes': np.vstack([item['boxes'] for item in batch])
    }




