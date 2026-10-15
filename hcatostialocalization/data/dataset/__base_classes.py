# PyTorch Datasets Base class for any dataset (both 2d, 2d3s, 3d)
from typing import Literal, Callable
import os
from abc import ABC, abstractmethod
import numpy
import SimpleITK as sitk
from sqlalchemy import case
import torch
import logging as logger

from hcatostialocalization.data.transform.transformations_2d import augment_2d
from hcatostialocalization.data.transform.transformations_3d import augment_3d




class LabeledDataset(torch.utils.data.Dataset, ABC):
    '''
    # LabeledDataset
    
    Base class for labeled datasets handling 3D CT volumes, designed to provide training, validation, and test data for deep learning models. 
    Supports 2D, 3D, and a special "2d3s" case where 2D slices are stacked as 3-channel images with contiguous slices. 
    The class manages data loading, patch sampling, augmentation, and batching, with a focus on efficient memory usage and flexibility for different dataset structures.
    
    ## Key Features:
    
    - Handles 3D CT volumes and supports 2D, 3D, and 2d3s (3-channel 2D) training.
    - Provides a unified interface for train, validation, and test splits.
    - Efficient memory usage by loading one image at a time for training.
    - Patch sampling and augmentation for training batches.
    - Abstracts data pool initialization and patient ID formatting for dataset-specific customization.
    
    ## Abstract Methods:
    
    - `_init_data_pool(self) -> list[tuple[str, str]]`:
        Must be implemented by child classes to return a list of (image_path, label_path) tuples for the dataset.
    - `_get_patient_id(self, current_patient: tuple[str, str]) -> str`:
        Must be implemented by child classes to return a readable patient ID string for logging and output.
    
    # Semi-Abstract Methods (Template Methods):
    
    These methods have default implementations but are designed to be overridden by child classes for advanced use cases:
    - `_check_sample_inclusion_criterion(self, *args, **kwargs) -> bool`:
        Determines whether a sampled patch should be included in the batch (e.g., based on label presence). Can be overridden for more complex sampling strategies.
    - `_load_volume(self, f) -> dict`:
        Loads a volume from file and returns a dictionary with the tensor and original spacing. Can be customized for different file formats or metadata handling.
    
    ## Typical Usage:
    
    Child classes should implement the abstract methods to define how data is discovered and how patient IDs are formatted.
    Optionally, they can override the semi-abstract methods to customize data loading, patch sampling, or inclusion criteria.
    '''
    labels: dict[int, str] = {} # "abstract" class attribute to be defined in child classes: mapping from label index to label name

    def __init__(
            self, 
            split: Literal['train', 'val', 'test'], 
            nsdim: Literal[2, 3], 
            sample_sdim: int|tuple = 128,
            minibatch_size: int = 1,
            change_after_n_sampling: int = 64, 
            new_spacing: float|tuple|None = None,
            label_transform_fn: Callable|None = None,
            train_sampling_strategy: Literal['random', 'empty-balanced', 'class-balanced'] = 'random',
            train_proportion_with_labels: float = 0.75,
            **kwargs
        ):
        super().__init__()
        self.nsdim = int(nsdim) # number of spatial dimensions required for training
        self.sample_sdim = sample_sdim if isinstance(sample_sdim, tuple) else tuple([sample_sdim]*self.nsdim) # these will be the two spatial dimensions of the minibatch in train
        self.sample_sdim = tuple([int(d) for d in self.sample_sdim])
        self._sample_sdim_3d = self.sample_sdim if self.nsdim == 3 else tuple([1] + list(self.sample_sdim)) # this is the dimension-agnostic minibatch dimension used in internal computations
        if len(self.sample_sdim) != self.nsdim:
            raise ValueError(f"argument minibatch_sample_size should be int or tuple with length equal to argument nsdim ({self.nsdim}), got {self.sample_sdim}.")
        self.minibatch_size = int(max(minibatch_size, 1))
        self.split = split
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"'split' must be in ['train', 'val', 'test'], got {self.split}") 
        self.new_spacing = new_spacing # (K1, K2, K3)
        # label transform function upon load
        self._label_transform_fn = label_transform_fn
        # Internal random number generator
        self.rng: numpy.random.Generator = numpy.random.default_rng(seed=42)
        # Save any additional kwarg passed to the init, child classes may need that
        self._kwargs = kwargs if kwargs else {}
        # Get data paths 
        self.data_pool: list[tuple[str, str]] = self._init_data_pool() # this is a blueprint function, classes inheriting this should implement this fucntion themselves
        # slit-wise configs
        if self.split == "train":
            # Low memory training:
            # load in memory just one image at a time.
            # After a counter is depleted, the loaded image changes
            self.change_after_n_sampling = int(change_after_n_sampling)
            self.counter = 0
            self.current_patient: tuple[str, str] = None
            self.loaded: dict = None    # -> dict with 'image', 'label', 'spacing', 'spacing_orig'
                                        # image dtype is float16
                                        # label dtype is int16
            self.next() # loads current_patient and loaded correctly
            self.train_sampling_strategy = train_sampling_strategy
            self.train_proportion_with_labels = numpy.clip(train_proportion_with_labels, 0.0, 1.0)
        elif self.split in ["val", "test"]:
            # In validation and test, the get method will get one full image
            # in the form of (1, 1, n_slices, 512, 512)
            # self.current_patient is used to cycle among all patients
            # and then restart from the beginning
            # loading of the actual image and label is deferred to the get method and submethods
            self.current_patient = self.data_pool[0]
        else:
            raise NotImplementedError(f"'split' must be in ['train', 'val', 'test'], got {self.split}")
    
        
        if (
            self.labels is None or
            self.labels == {} or 
            any([not isinstance(k, int) for k in self.labels.keys()]) or
            any([not isinstance(v, str) for v in self.labels.values()])
        ):
            raise TypeError('Subclasses must set \'label\' to a dictionary mapping integer label indices to string label names. It must contain the background label.')
        if 'background' not in self.labels.values():
            raise ValueError('The \'labels\' dictionary must contain a mapping for the background label (any integer).')
        # Debug print when finished loading
        logger.info(f"Loaded {self.__class__.__name__}")
    
    @abstractmethod
    def _init_data_pool(self) -> list[tuple[str, str]]|list[tuple[str, tuple[str, str]]]:
        # Method in whcih each dataset will be able to load a list of image-label data path pairs
        raise NotImplementedError(f"Abstract/Template method inherited by {self.__class__.__name__} in {__file__}")
    
    @abstractmethod
    def _get_patient_id(self, current_patient: tuple[str, str]|tuple[str, tuple[str, str]]) -> str:
        # Method in which to define the id of the patient in a readable, nice way.
        # A couple of examples:
        # ASOCA: 
        #   f"{self.split}-ASOCA-{out_id}"
        # ImageCAS:
        #   f"{self.split}-ImageCAS-{out_id}"
        raise NotImplementedError(f"Abstract/Template method inherited by {self.__class__.__name__} in {__file__}")
        
    def _transform_label_upon_load(self, label_tensor: torch.Tensor) -> torch.Tensor:
        '''Template method with standard definition to transform the label upon loading.
           Intended use: no geometrical transform, just label meaning and channel expansion transforms.
           By default, no transformation is applied.
           If label_transform_fn is provided in the init, it is applied here.
        '''
        if self._label_transform_fn is not None:
            label_tensor = self._label_transform_fn(label_tensor)
        return label_tensor

    def choose_random_from_datapool(self, n: int = 1) -> tuple[str,str]|list[tuple[str, str]]|tuple[str, tuple[str, str]]|list[tuple[str, tuple[str, str]]]:
        """
        Compatibility function to choose random samples from the data pool
        (self.rng.choice does not work if we have tuples as labels inside the data pool)
        """
        n = max(1, int(n))
        if n == 1:
            idx = self.rng.integers(low=0, high=len(self.data_pool))
            return self.data_pool[idx]
        else:
            idxs = self.rng.integers(low=0, high=len(self.data_pool), size=n)
            return [self.data_pool[i] for i in idxs]

    def next(self) -> None:
        """
        Chooses a new patient from the data pool and loads in in memory in self.loaded (dict)
        - self.loaded: dict with keys 'image', 'label', 'spacing', 'spacing_orig'
        """
        self.current_patient = self.choose_random_from_datapool(n=1)
        self.loaded = self._load_data_and_transform_label(self.current_patient, self.new_spacing)

    def _load_volume(self, f) -> dict:
        # CAN BE TEMPLATE, JUST ALWAYS OUTPUT A DICT LIKE {"tensor": volume, "spacing_orig": spacing_orig}
        # if spacing_orig is not present in the volume info, use: spacing_orig = (0.54321, 0.54321, 0.54321)
        #
        # out: volume as tensor, spacing as tuple
        file_ext = os.path.splitext(f)[-1].lower()
        if ".nii.gz" in f.lower():
            file_ext = ".nii.gz"
        match file_ext:
            case ".mhd":
                # .mhd * .raw + .zraw file
                # volume file
                data = sitk.ReadImage(f)
                # image tensor
                volume = sitk.GetArrayFromImage(data)  # -> (slices, K2, K3) or (slices, H, W) or (depth, height, width)
                volume = torch.from_numpy(volume)[None, None, ...]  # -> (1, 1, slices, K2, K3) or (1, 1, K1, K2, K3)
                # spacing (original)
                try:
                    spacing_orig = data.GetSpacing() # -> x,y,z but volume is z,y,x
                    # - bring spacing in same order of volume dims
                    spacing_orig = spacing_orig[::-1]
                except:
                    spacing_orig = (0.54321, 0.54321, 0.54321) # i assign it myself if i cannot find it
                spacing_orig = tuple([float(s) for s in spacing_orig])
            case _:
                # volume file
                data = sitk.ReadImage(f)
                # image tensor
                volume = sitk.GetArrayFromImage(data)  # -> (slices, K2, K3) or (slices, H, W) or (depth, height, width)
                volume = torch.from_numpy(volume)[None, None, ...]  # -> (1, 1, slices, K2, K3) or (1, 1, K1, K2, K3)
                # spacing (original)
                try:
                    spacing_orig = data.GetSpacing() # -> x,y,z but volume is z,y,x
                    # - bring spacing in same order of volume dims
                    spacing_orig = spacing_orig[::-1]
                except:
                    spacing_orig = (0.54321, 0.54321, 0.54321) # i assign it myself if i cannot find it
                spacing_orig = tuple([float(s) for s in spacing_orig])
        return {"tensor": volume, "spacing_orig": spacing_orig}
    
    def _load_volume_resize(self, f, which: Literal['image', 'label'], new_spacing: float|tuple|None = None) -> dict:
        # out: dict with 'tensor', 'spacing', 'spacing_orig'
        # load image with its dedicated loader
        data = self._load_volume(f)
        # get image/label
        img = data['tensor']
        if which == 'image':
            img = img.float().clone()
        elif which == 'label':
            img = img.long().clone()
        # get spacing (original)
        spacing_orig = data['spacing_orig']
        # spacing
        resize = False
        if new_spacing is None:
            spacing = spacing_orig
        elif isinstance(new_spacing, float):
            spacing = tuple([new_spacing]*len(spacing_orig))
            resize = True
        elif isinstance(new_spacing, tuple):
            spacing = tuple([float(s) for s in new_spacing])
            if len(spacing) != len(spacing_orig):
                raise ValueError(f"arg new_spacing should be of length {len(spacing_orig)}, got: {spacing}")
            resize = True
        else:
            raise ValueError(f"Wrong new_spacing: {new_spacing}, {type(new_spacing)}")
        # resize volume if needed
        if resize:
            img = torch.nn.functional.interpolate(
                img.float(),
                scale_factor=tuple([spacing_orig[i]/spacing[i] for i in range(len(spacing))]), # new_size = old_size * (old_sp/new_sp), that is the scale_factor
                mode="nearest" if which=='label' else "trilinear",
            )
            if which=='label':
                # interpolate does not work with long, even if mode is nearest
                img = img.long()
        # done
        out = {
            'tensor': img,
            'spacing': spacing,
            'spacing_orig': spacing_orig
        }
        return out
    
    def _load_data(self, image_label_f: tuple[str, str]|tuple[str,tuple[str]], new_spacing: float|tuple|None = None) -> dict:
        # img
        dict_img = self._load_volume_resize(image_label_f[0], which='image', new_spacing=new_spacing)
        # - if training, if volume is too large, randomly select a subvolume
        #   for both image and label
        # Do it here right away, otherwise it can happen that label volume transformations with torch.where()
        # saturate memory already.
        reduce_volume = False
        if self.split in ['train']:
            max_memory_GB = 1.0 # for each
            shape_axial = dict_img['tensor'].shape[-2:] # for images of shape (B,C,K1,K2,K3), most times K2,K3 are axial (square) images, and K1 is the index of slices
            bytes_per_element = 2 # images are stores in self.loaded as base 16 data (float16 and int16)
            max_slices = max(
                int((max_memory_GB * 1024**3) / (bytes_per_element * shape_axial[0] * shape_axial[1])), 
                self._sample_sdim_3d[0], 
                32
            )
            if max_slices < dict_img['tensor'].shape[2]:
                start_slice = self.rng.integers(low=0, high=dict_img['tensor'].shape[2] - max_slices + 1)
                end_slice = start_slice + max_slices
                dict_img['tensor'] = dict_img['tensor'][:, :, start_slice:end_slice, ...]
                reduce_volume = True
        # label
        # label can be a single file or a tuple/list of files. If the latter, they must be merged together
        # TEST SET: label can be "None"
        try:
            if isinstance(image_label_f[1], str):
                dict_lab = self._load_volume_resize(image_label_f[1], which='label', new_spacing=new_spacing)
                if reduce_volume:
                    # reduce volume right away to avoid memory issues
                    dict_lab['tensor'] = dict_lab['tensor'][:, :, start_slice:end_slice, ...]
            else:
                # multi-volume label
                dict_lab_list = [
                    self._load_volume_resize(f, which='label', new_spacing=new_spacing)
                    for f in image_label_f[1]
                ]
                if reduce_volume:
                    # reduce volume right away to avoid memory issues
                    for i in range(len(dict_lab_list)):
                        dict_lab_list[i]['tensor'] = dict_lab_list[i]['tensor'][:, :, start_slice:end_slice, ...] 
                # merge into single label
                # - check that spacing, spacing_orig and tensor shape are the same
                _err = False
                for d in dict_lab_list[1:]:
                    if list(d['spacing']) != list(dict_lab_list[0]['spacing']):
                        _err = True
                        logger.error(f"ERROR: Label volumes to be merged do not have the same spacing: {d['spacing']} vs {dict_lab_list[0]['spacing']}")
                    if list(d['spacing_orig']) != list(dict_lab_list[0]['spacing_orig']):
                        _err = True
                        logger.error(f"ERROR: Label volumes to be merged do not have the same original spacing: {d['spacing_orig']} vs {dict_lab_list[0]['spacing_orig']}")
                    if list(d['tensor'].shape) != list(dict_lab_list[0]['tensor'].shape):
                        _err = True
                        logger.error(f"ERROR: Label volumes to be merged do not have the same shape: {d['tensor'].shape} vs {dict_lab_list[0]['tensor'].shape}")
                if _err:
                    raise ValueError("Cannot merge label volumes with different spacing/shape. See errors above.")
                # - merge - supports multiple volumes, each with multiple classes inside
                label_tensor = torch.zeros_like(dict_lab_list[0]['tensor'], dtype=torch.long)
                for d in dict_lab_list:
                    stored_max = label_tensor.max().item()
                    _where = d['tensor'] > 0
                    label_tensor[_where] = d['tensor'][_where] + stored_max
                dict_lab = {
                    'tensor': label_tensor,
                    'spacing': dict_lab_list[0]['spacing'],
                    'spacing_orig': dict_lab_list[0]['spacing_orig']
                }
            # correct label
            if list(dict_img['tensor'].shape[2:]) != list(dict_lab['tensor'].shape[2:]):
                # something is wrong with the spacing read in the label file probably.
                # Make sure label spacing is correct
                dict_lab['tensor'] = torch.nn.functional.interpolate(
                    dict_lab['tensor'].float(), 
                    tuple(list(dict_img['tensor'].shape[2:])), 
                    mode='nearest'
                ).long()
            # special case: 2d3s
            if self._kwargs.get('is_2d3s', False):
                dict_img['tensor'] = dict_img['tensor'].repeat((1, 3, 1, 1, 1))
                dict_img['tensor'][:,0,1:,...] = dict_img['tensor'][:,1,:-1,...]
                dict_img['tensor'][:,2,:-1,...] = dict_img['tensor'][:,1,1:,...]
        except Exception as e:
            if os.path.exists(image_label_f[1]) or self.split != 'test':
                raise RuntimeError(f"Following exception was thrown even if test set label file exists ({image_label_f[1]})\n\n", e)
            # fill out['label'] with "None", so that downstream logic can take that into account without stopping the program
            # if the label for a test set image is not found
            dict_lab = {
                'tensor': None
            }
        # Modify data type
        if self.split == 'train':
            dict_img['tensor'] = dict_img['tensor'].to(torch.float16)
            dict_lab['tensor'] = dict_lab['tensor'].to(torch.int16) if torch.is_tensor(dict_lab['tensor']) else None
        # out
        out = {
            'image': dict_img['tensor'], 
            'label': dict_lab['tensor'], 
            'spacing': dict_img['spacing'], 
            'spacing_orig': dict_img['spacing_orig']
        }
        return out

    def _load_data_and_transform_label(self, image_label_f: tuple[str, str]|tuple[str,tuple[str]], new_spacing: float|tuple|None = None) -> dict:
        # This is the function that must be called to load data INSIDE the class.
        # Form OUTSIDE (an end user) just use the next() method.
        # THIS IS NOT TEMPLATE-LIKE and must not be tambered with
        out = self._load_data(image_label_f, new_spacing)
        # transfrom label:
        # label is transformed here by a method that can be templated, so
        # each dataset can define its own transformation. By default, no
        # transformation is applied. It is up to the method to keep the labels
        # correct and coherent.
        # Do it after volume reduction
        if out['label'] is not None:
            out['label'] = self._transform_label_upon_load(out['label'])
        return out
        

    def __getitem__(self, index):
        # index is ignored in this kind of dataset
        # This is kept defined here only to make it compliant with DataLoaders.
        # In the dataLoader, specify that the batching is performed by the Dataset class and should not
        # be performed by the DataLoader.
        # Mostly useful with nfs file systems and with DistibutedDataParallel or similar
        return self.get()
    
    def get(self, minibatch_size: int|None = None):
        """
        Gets a sample depending on the split

        # Input

        - minibatch_size (int): only for the `train` split, the number of 3d samples to output. If not provided, self.minibatch_size is used. It's value will be >= 1.
        
        # Output

        - `train`: a tuple of img_minibatch, label_minibatch.
        - `val` & `test`: a dictionary composed of:
            - "id": "<split>.ImageCAS.<patient id>",
            - "image": image (torch.Tensor (float32) of shape (B, C, K1, K2[, K3])) where B is 1 for 3d volumes, B is number of slices for 2d and 2dns images,
            - "label": label (torch.Tensor (longint) of shape (B, 1, K1, K2[, K3])),
            - "spacing": tuple,
            - "spacing_orig": tuple.

        # Data Types

        While the training dataset stores the loaded images as float16 and int16 to save considerable space, 
        all datasets return data in the standard float32 for images and int64 for labels.

        """
        if minibatch_size is None:
            minibatch_size = self.minibatch_size
        minibatch_size = int(max(minibatch_size, 1))
        match self.split:
            case "train":
                return self._get_train(minibatch_size)
            case "val":
                return self._get_val()
            case "test":
                return self._get_test()
            case _:
                return None

    def _pad_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        shape = tensor.shape
        sshape = shape[2:]
        if any([k < s for k, s in zip(sshape, self._sample_sdim_3d)]):
            pads = [max(0, s-k) for k, s in zip(sshape, self._sample_sdim_3d)]
            # define the padding tuple: (pad_left_k3, pad_right_k3, pad_left_k2, pad_right_k2, pad_left_k1, pad_right_k1)
            # pad applies padding in reverse dimension order (last dim first)
            padding_config = [
                [p//2, p-p//2] for p in pads[::-1]
            ]
            padding_config = [x for xs in padding_config for x in xs]
            try:
                return torch.nn.functional.pad(tensor, padding_config, mode='reflect')
            except:
                try:
                    return torch.nn.functional.pad(tensor, padding_config, mode='constant', value=float(tensor.min().item()))
                except:
                    return torch.nn.functional.pad(tensor, padding_config, mode='constant', value=0.0)
        return tensor

    def _check_sample_inclusion_criterion(self, *args, **kwargs) -> bool: 
        """
        Check whether a given sample (or sample in a minibatch position) should be included
        in the training stream according to a target proportion of labelled (non-empty)
        samples.
        Returns:
            bool: True to include the sample, False to reject it.
        Parameters (via kwargs):
            lab (torch.Tensor | None):
                A tensor containing the sample label. The function treats the sample as
                "non-empty" when (lab != 0).sum().item() > 0. If lab is None or not a
                tensor, the current implementation will raise; callers must pass a tensor.
            minibatch_size (int | None):
                Size of the minibatch being assembled. When minibatch_size == 1 the
                function uses a sequential extraction plan (self._extraction_plan) and
                a position counter (self._train_draw_history) to decide acceptance.
                When minibatch_size > 1 a deterministic index-based rule is used:
                accept samples where minibatch_index < int(minibatch_size * proportion_with_labels).
            minibatch_index (int | None):
                Index inside the current minibatch used only when minibatch_size > 1.
        """
        # POSSIBLE TEMPLATE
        # This can be changed to complicate it in the case of more advance checks
        #
        # true: sample is to be included
        # false: sample is NOT to be included
        lab: torch.Tensor = kwargs.get('lab', None)
        minibatch_size: int = kwargs.get('minibatch_size', None)
        minibatch_index: int = kwargs.get('minibatch_index', None)
        # if proportion_with_slabel is not give, it is set to 0, which means that
        # the inclusion criterion will always be true (take anything)
        # The more proportion_with_labels get closer to 1.0, the more samples with labels
        # will be chosen.
        proportion_with_labels = self._kwargs.get('proportion_with_labels', 0.0)
        proportion_with_labels = numpy.clip(proportion_with_labels, 0.0, 1.0)
        # How "emptyness" is computed:
        # - basic naive approach: any label != 0 is good
        is_non_empty = (lab != 0).sum().item() > 0
        # - advancd balancing attempt: cycling through the volume held classes
        #   (label unique values), we want to extract that class
        
        if minibatch_size == 1:
            # different logic
            # we want ot feed training samples once with labels, 1-n without
            # label inside, n with label, depending on proportion_with_labels
            if (
                (not hasattr(self, "_train_draw_history")) or 
                (not hasattr(self, "_extraction_plan")) or 
                (self._train_draw_history) is None or 
                (self._extraction_plan is None)
            ):
                # init them here, because this rfunction is template-able, meaning that if overwritten, these
                # two will have no use elsewhere -> so I do not initialize them in the class init
                self._train_draw_history: int = 0
                self._extraction_plan = self.rng.permutation(
                    [ True if i < 1000*proportion_with_labels else False for i in range(1000) ]
                    # True: accept with non-empty label
                    # False: accept with empty label
                )
            accept_non_empty = self._extraction_plan[self._train_draw_history]
            if accept_non_empty == is_non_empty:
                pass_ = True
            else:
                pass_ = False
            self._train_draw_history = (self._train_draw_history + 1) % len(self._extraction_plan)
        else:
            # I have a batch of size minibatch_size, of which I want a proportion of them
            # to have non-empty labels, some empty
            non_empty_index = minibatch_index <= int(minibatch_size*proportion_with_labels)
            if is_non_empty == non_empty_index:
                # try with new sample to find one with labels inside
                pass_ = True
            else:
                pass_ = False
        return pass_

    def _get_train(self, minibatch_size: int):
        # here, we load one single image that will stay the same untill
        # self.change_after_n_sampling quota is reached (circa).
        # Here we have augmentations and such.

        # pad original image and label if sample_sdim is too big
        # -> this will aply only once iff there's need for padding
        ##########################################################

        self.loaded['image'] = self._pad_tensor(self.loaded['image'])
        self.loaded['label'] = self._pad_tensor(self.loaded['label'])

        # get minibatch_size samples
        ############################

        out_size_img = (minibatch_size, self.loaded['image'].shape[1], *self.sample_sdim)
        out_size_lab = (minibatch_size, self.loaded['label'].shape[1], *self.sample_sdim)
        img_minibatch = torch.zeros(out_size_img, dtype=torch.float32)
        lab_minibatch = torch.zeros(out_size_lab, dtype=torch.long)

        b = 0
        stuck_counter = 0
        while b < minibatch_size:
            # get patches
            if self.train_sampling_strategy in ['random','empty-balanced']:
                # random sampling strategy:
                # just sample 
                low = [
                    self.rng.integers(low=0, high=self.loaded['image'].shape[i+2]-self._sample_sdim_3d[i]+1)
                    for i in range(len(self._sample_sdim_3d))
                ]
                high = [
                    low[i]+self._sample_sdim_3d[i]
                    for i in range(len(self._sample_sdim_3d))
                ]
                # Slicing: integer index for dim 0 if 2D (to squeeze), else slice
                s_idx = [low[0] if self.nsdim == 2 else slice(low[0], high[0])]
                s_idx += [slice(l, h) for l, h in zip(low[1:], high[1:])]
                img_ = self.loaded['image'][(slice(None), slice(None), *s_idx)]
                lab_ = self.loaded['label'][(slice(None), slice(None), *s_idx)]
                if self.train_sampling_strategy == 'empty-balanced':
                    # check inclusion condition
                    if not self._check_sample_inclusion_criterion(
                        lab=lab_, 
                        minibatch_size=minibatch_size, 
                        minibatch_index=b,
                        proportion_with_labels=self.train_proportion_with_labels
                    ):
                        if stuck_counter < 20:
                            stuck_counter += 1
                            continue
                        # else, probably stuck, just save that
            elif self.train_sampling_strategy == 'class-balanced':
                # class-balanced:
                # - get the unique classes contained in this dataset, and prepare an extraction plan for it
                #   if it is a new training sample, prepare the extraction plan, otherwise, use the already existing plan
                #   we can use self.counter
                # - also take into account all-empty slices following the provided self.train_proportion_with_labels
                # - find suitable slices, draw randomly from them
                # Note: the extraction plan is a list meaning : in this batch element, there has to be at least one pixel with the class
                #
                # extraction plan
                if self.counter == 0:
                    unique = self.loaded['label'].unique().cpu().numpy().astype(int)
                    unique_fg = unique[unique != 0]
                    if len(unique_fg) == 0:
                        self._train_class_balanced_extraction_plan = numpy.zeros(2*self.change_after_n_sampling, dtype=int)
                    else:
                        self._train_class_balanced_extraction_plan = self.rng.choice(unique_fg, size=2*self.change_after_n_sampling)
                        n_empty = int(len(self._train_class_balanced_extraction_plan) * (1.0 - self.train_proportion_with_labels))
                        self._train_class_balanced_extraction_plan[:n_empty] = 0
                        self.rng.shuffle(self._train_class_balanced_extraction_plan)
                    # create indices only once as to where find classes
                    self._train_class_balanced_positions = {}
                    for c in unique:
                        where = (self.loaded['label'] == c).nonzero()
                        self._train_class_balanced_positions[c] = where
                # extraction
                target_class = self._train_class_balanced_extraction_plan[self.counter + b]
                if target_class == 0:
                    # Try to find empty patch
                    positions = self._train_class_balanced_positions.get(0, None)
                    for _ in range(50):
                        if positions is None or len(positions) == 0:
                            low = [
                                self.rng.integers(low=0, high=self.loaded['image'].shape[i+2]-self._sample_sdim_3d[i]+1)
                                for i in range(len(self._sample_sdim_3d))
                            ]
                        else:
                            idx = self.rng.integers(0, len(positions))
                            center_pos = positions[idx]
                            low = []
                            for i in range(len(self._sample_sdim_3d)):
                                dim_len = self.loaded['image'].shape[i+2]
                                patch_size = self._sample_sdim_3d[i]
                                center = center_pos[i+2].item()
                                l = max(0, min(center - patch_size // 2, dim_len - patch_size))
                                low.append(l)
                        
                        high = [l + s for l, s in zip(low, self._sample_sdim_3d)]
                        s_idx = [low[0] if self.nsdim == 2 else slice(low[0], high[0])]
                        s_idx += [slice(l, h) for l, h in zip(low[1:], high[1:])]
                        
                        lab_ = self.loaded['label'][(slice(None), slice(None), *s_idx)]
                        if (lab_ == 0).sum() / lab_.numel() > 0.95:
                            break
                    # If loop finishes without break, we just use the last sampled patch (give up)
                    img_ = self.loaded['image'][(slice(None), slice(None), *s_idx)]
                    # lab_ is already set
                else:
                    positions = self._train_class_balanced_positions.get(target_class, None)
                    if positions is None or len(positions) == 0:
                        # fallback to random if class not found
                        low = [
                            self.rng.integers(low=0, high=self.loaded['image'].shape[i+2]-self._sample_sdim_3d[i]+1)
                            for i in range(len(self._sample_sdim_3d))
                        ]
                    else:
                        # pick a random pixel of that class
                        idx = self.rng.integers(0, len(positions))
                        center_pos = positions[idx] # (B, C, D, H, W) indices
                        low = []
                        for i in range(len(self._sample_sdim_3d)):
                            dim_len = self.loaded['image'].shape[i+2]
                            patch_size = self._sample_sdim_3d[i]
                            center = center_pos[i+2].item()
                            l = max(0, min(center - patch_size // 2, dim_len - patch_size))
                            low.append(l)
                    high = [l + s for l, s in zip(low, self._sample_sdim_3d)]
                    s_idx = [low[0] if self.nsdim == 2 else slice(low[0], high[0])]
                    s_idx += [slice(l, h) for l, h in zip(low[1:], high[1:])]
                    # get
                    img_ = self.loaded['image'][(slice(None), slice(None), *s_idx)]
                    lab_ = self.loaded['label'][(slice(None), slice(None), *s_idx)]
            else:
                raise ValueError(f"Unknown train_sampling_strategy: {self.train_sampling_strategy}")
            # augment patches
            if self.nsdim == 2:
                img_, lab_ = augment_2d(img_, lab_)
            else:
                img_, lab_ = augment_3d(img_, lab_)
            # insert into minibatch
            img_minibatch[b:b+1] = img_
            lab_minibatch[b:b+1] = lab_
            # next one
            stuck_counter = 0
            b += 1

        # decide wether it is the case to change loaded image-label
        ###########################################################

        self.counter += minibatch_size
        if self.counter >= self.change_after_n_sampling:
            # restart counter and choose another patient from which to draw the next samples
            self.counter = 0
            self.next()

        # out
        #####
        return (img_minibatch, lab_minibatch)
# ALLA FINE FAI USCIRE UN DIZIONARIO
    def _get_val(self):
        return self._get_val_test()

    def _get_test(self):
        return self._get_val_test()

    def _get_val_test(self) -> dict:
        # load current patient
        data = self._load_data_and_transform_label(self.current_patient, self.new_spacing)
        # - transform, based on wether the dataset is 2d (or 2d3s) or 3d
        if self.nsdim == 2:
            # 2d case: (1, C, K1, K2, K3) reshape to (K1, C, K2, K3)
            # 2dns case (2d with multiple slices as channels) is already taken care of in the loading phase,
            # no further step required here
            data['image'] = data['image'][0].permute(1, 0, 2, 3).contiguous()
            if data['label'] is None and self.split != 'test':
                raise ValueError("data['label'] is None when it is not allowed to be.")
            if data['label'] is not None:
                data['label'] = data['label'][0].permute(1, 0, 2, 3).contiguous()
        out_id = self._get_patient_id(self.current_patient)
        # go to next patient, ready for next iteration
        position = self.data_pool.index(self.current_patient)
        if position == -1:
            logger.debug(f"position is {position}, something is wrong.")
        next_position = (position + 1) % len(self)
        self.current_patient = self.data_pool[next_position]
        # make sure data type is correct
        data['image'] = data['image'].to(torch.float32)
        if data['label'] is not None:
            data['label'] = data['label'].to(torch.int64)
        # output current patient dict
        data['id'] = out_id
        return data

    def __len__(self):
        # Number of volumes (patients) stored
        return len(self.data_pool)

    


class LabeledDatasetsCollection(torch.utils.data.Dataset):
    '''
    # LabeledDatasetsCollection

    A PyTorch Dataset class that aggregates multiple LabeledDataset instances, allowing for combined training, validation, and testing across diverse datasets.
    It supports balanced sampling from each dataset during training and sequential access during validation and testing.

    ## Args:
    - datasets (list[LabeledDataset|tuple[LabeledDataset]]): A list of LabeledDataset instances or tuples of LabeledDataset instances to be included in the collection.
    - split (Literal['train', 'val', 'test']): The data split to use ('train', 'val', or 'test'). For each datasets, splits are defined in the corresponding yaml config file.
    - nsdim (Literal[2, 3]): The number of spatial dimensions for training
    - sample_sdim (int|tuple): The size of the spatial dimensions for sampling during training. Can be an int or a tuple of ints.
    - minibatch_size (int): The number of samples to include in each training minibatch. If lower than len(datasets), datasets will be chosen randomly to fill the minibatch.
    - draws_proportion (list[float] | None): Available only for train split. Optional list specifying the proportion of samples to draw from each dataset list item during training.
      A dataset list item can be a single LabeledDataset or a tuple of LabeledDataset instances. If a tuple, it means that one of the datasets in the tuple will be chosen randomly at each draw.
      The grand total must sum to 1.0. If not, it will be normalized automatically. If not provided, all items in the datasets list will be weighted equally.
    - multichoice_probability (list[float|tuple[float]] | None): Available only for train split.Optional list specifying the probability of choosing a dataset from a tuple of datasets during training.
      It must have the lame exact layout of datasets, meaning that, if datasets[i] is a tuple of n datasets, then multichoice_probability[i] must be a tuple of n floats summing to 1.0.
      If they do not sum to 1, they will be normalized automatically.
    '''
    def __init__(
            self, 
            datasets: list[LabeledDataset|tuple[LabeledDataset]],
            split: Literal['train', 'val', 'test'], 
            nsdim: Literal[2, 3], 
            sample_sdim: int|tuple = 128,
            minibatch_size: int = 1,
            draws_proportion: list[float] | None = None,
            multichoice_probability: list[float|tuple[float]] | None = None,
        ):    
        super().__init__()
        self.nsdim = int(nsdim) # number of spatial dimensions required for training
        self.sample_sdim = sample_sdim if isinstance(sample_sdim, tuple) else tuple([sample_sdim]*self.nsdim)
        if len(self.sample_sdim) != self.nsdim:
            raise ValueError(f"argument minibatch_sample_size should be int or tuple with length equal to argument nsdim ({self.nsdim}), got {self.sample_sdim}.")
        self.minibatch_size = int(max(minibatch_size, 1))
        self.split = split
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"'split' must be in ['train', 'val', 'test'], got {self.split}") 
        self.rng = numpy.random.default_rng(seed=42)

        self.datasets = datasets
        if self.split != 'train': 
            if any([isinstance(d, (tuple,list)) for d in self.datasets]):
                raise ValueError(f"When split is '{self.split}', datasets cannot contain tuples of datasets, got: {self.datasets}")
        
        if draws_proportion is None:
            draws_proportion = [1.0/len(self.datasets) for _ in self.datasets]
        if len(draws_proportion) != len(self.datasets) or any([not isinstance(p, float) for p in draws_proportion]) or any([p < 0.0 for p in draws_proportion]):
            raise ValueError(f"draws_proportion is wrong, got: {draws_proportion}")
        _t = sum(draws_proportion)
        self.draws_proportion = [p/_t for p in draws_proportion]

        if multichoice_probability is None:
            multichoice_probability = []
            for d in self.datasets:
                if isinstance(d, (tuple, list)):
                    n = len(d)
                    multichoice_probability.append(tuple([1.0/n for _ in d]))
                else:
                    multichoice_probability.append(1.0)
        if len(multichoice_probability) != len(self.datasets):
            raise ValueError(f"multichoice_probability is wrong, got: {multichoice_probability}")
        self.multichoice_probability = []
        for p, d in zip(multichoice_probability, self.datasets):
            if isinstance(d, (tuple, list)):
                if not isinstance(p, (tuple, list)):
                    raise ValueError(f"multichoice_probability entry must be a tuple/list when dataset entry is a tuple/list, got: {p} for dataset {d}")
                if len(p) != len(d) or any([not isinstance(pp, float) for pp in p]) or any([pp < 0.0 for pp in p]):
                    raise ValueError(f"multichoice_probability entry is wrong, got: {p} for dataset {d}")
                _t = sum(p)
                self.multichoice_probability.append(tuple([pp/_t for pp in p]))
            else:
                if not isinstance(p, float) or p < 0.0:
                    raise ValueError(f"multichoice_probability entry is wrong, got: {p} for dataset {d}")
                self.multichoice_probability.append(float(p))
        
        self.datasets_lenghts = []
        for d in self.datasets:
            if isinstance(d, LabeledDataset):
                self.datasets_lenghts.append(len(d))
            else:
                _l = tuple([len(dd) for dd in d])
                self.datasets_lenghts.append(_l)
        
        # Use this to associate each batch element in training with its generator dataset
        if self.split == 'train':
            self.training_last_draws: list[LabeledDataset] = []
        
        if self.split in ['val', 'test']:
            # datasets is just a list of LabeledDataset instances
            self._val_test_counter = 0
            self._dataset_to_draw_from = [] # a list long as len(Dataset_3d) holding the correct class for each draw 
            for d in self.datasets:
                # this does not weight on memory, as ffectively the list only holds references (pointers) to the objects
                self._dataset_to_draw_from.extend(
                    [d]*len(d)
                )

    def __getitem__(self, index):
        # Alias for .get() method, index is ignored
        return self.get()

    def get(self, minibatch_size: int|None = None):
        """
        Gets a sample depending on the split

        # Input

        - minibatch_size (int): only for the `train` split, the number of 3d samples to output.
                                If specified in the init of the class, that one is used.
        
        # Output

        - `train`: a tuple of img_minibatch, label_minibatch.
                   Each dataset contributes with (close to) the same number of samples to the minibatch.
        - `val` & `test`: a dictionary composed of:
            - "id": "<split>.<dataset name>.<patient id>",
            - "image": image (torch.Tensor (float32) of shape (1, C, K1, K2[, K3])),
            - "label": label (torch.Tensor (longint) of shape (1, 1, K1, K2[, K3]))
            - "spacing": tuple,
            - "spacing_orig": tuple.        
        """
        if minibatch_size is None:
            minibatch_size = self.minibatch_size
        minibatch_size = int(max(minibatch_size, 1))
        match self.split:
            case "train":
                return self._get_train(minibatch_size)
            case "val":
                return self._get_val()
            case "test":
                return self._get_test()
            case _:
                return None

    def _get_train(self, minibatch_size: int = 1):
        """Get training samples from held datasets"""
        # Initialize var to store which final samples come from which dataset:
        # list of datasets of same length as img_minibatch
        self.training_last_draws = []
        # build the minibatch - allocate memory all at once for better performance
        ch_im = (self.datasets[0] if isinstance(self.datasets[0], LabeledDataset) else self.datasets[0][0]).loaded['image'].shape[1]
        ch_lb = (self.datasets[0] if isinstance(self.datasets[0], LabeledDataset) else self.datasets[0][0]).loaded['label'].shape[1]
        #
        if self.nsdim == 2:
            out_size_img = (minibatch_size, ch_im, self.sample_sdim[0], self.sample_sdim[1])
            out_size_lab = (minibatch_size, ch_lb, self.sample_sdim[0], self.sample_sdim[1])
        elif self.nsdim == 3:
            out_size_img = (minibatch_size, ch_im, self.sample_sdim[0], self.sample_sdim[1], self.sample_sdim[2])
            out_size_lab = (minibatch_size, ch_lb, self.sample_sdim[0], self.sample_sdim[1], self.sample_sdim[2])
        else:
            raise ValueError(f"nsdim must be 2 or 3, got: {self.nsdim}")
        img_minibatch = torch.zeros(out_size_img, dtype=torch.float32)
        lab_minibatch = torch.zeros(out_size_lab, dtype=torch.long)
        
        # Decide how many samples to draw from each dataset statistically respecting draws_proportion and multichoice_probability.
        # dataset_draws is built to help extract images from each dataset at once, increasing performance
        dataset_draws: dict[int:int|dict[int:int]] = {}
        for i in range(len(self.datasets)):
            if not isinstance(self.datasets[i], (tuple, list)):
                dataset_draws[i] = 0
            else:
                dataset_draws[i] = {
                    j: 0 for j in range(len(self.datasets[i]))
                }
        
        for _ in range(minibatch_size):
            # Choose dataset according to draws_proportion
            dataset_index = self.rng.choice(
                a=len(self.datasets),
                p=self.draws_proportion
            )
            # get the dataset or one of the datasets in the tuple
            if isinstance(self.datasets[dataset_index], (tuple, list)):
                # choose one dataset from the tuple according to multichoice_probability
                probs = list(self.multichoice_probability[dataset_index])
                chosen_dataset_idx = self.rng.choice(
                    a=len(self.datasets[dataset_index]),
                    p=probs
                )
                dataset_draws[dataset_index][chosen_dataset_idx] += 1
            else:
                dataset_draws[dataset_index] += 1
        
        # draw samples
        c_ = 0
        for dataset_index, draws in dataset_draws.items():
            if not isinstance(draws, dict):
                # single dataset
                if draws == 0:
                    continue
                im, lb = self.datasets[dataset_index].get(minibatch_size=draws)
                img_minibatch[c_:c_+draws] = im
                lab_minibatch[c_:c_+draws] = lb
                self.training_last_draws.extend([self.datasets[dataset_index]]*im.shape[0])
                c_ += draws
            else:
                # multiple datasets
                for chosen_dataset_idx, n_draws in draws.items():
                    if n_draws == 0:
                        continue
                    im, lb = self.datasets[dataset_index][chosen_dataset_idx].get(minibatch_size=n_draws)
                    img_minibatch[c_:c_+n_draws] = im
                    lab_minibatch[c_:c_+n_draws] = lb
                    self.training_last_draws.extend([self.datasets[dataset_index][chosen_dataset_idx]]*im.shape[0])
                    c_ += n_draws
            if c_ >= minibatch_size:
                break
        if c_ < minibatch_size:
            raise ValueError(f"Something went wrong in building the minibatch, only filled {c_} samples out of {minibatch_size}.") 

        # out
        #####
        return (img_minibatch, lab_minibatch)
# ALLA FINE FAI USCIRE UN DIZIONARIO
    def _get_val(self):
        return self._get_val_test()

    def _get_test(self):
        return self._get_val_test()

    def _get_val_test(self) -> dict:
        data = self._dataset_to_draw_from[self._val_test_counter].get()
        self._val_test_counter = (self._val_test_counter + 1) % len(self)
        return data

    def __len__(self):
        # Number of 3D volumes (patients) stored
        return int(sum(self.datasets_lenghts))
