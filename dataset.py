import os
import pickle
from pathlib import Path
from typing import NamedTuple,Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import random
class SliceDataset_train(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """
    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        transform: Optional[Callable] = None,
    ):
        import glob
        paths=glob.glob(str(root)+"/*.nii.gz")
        self.examples=[]
        for path in paths:
            for j in range(384):
                self.examples.append((path,j))
        self.transform = transform
        print("train",len(self.examples))
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, index_slice = self.examples[i]
        data_slice=sitk.GetArrayFromImage(sitk.ReadImage(fname))


        if self.transform is None:
            sample = (data_slice, fname, index_slice)
        else:
            sample = self.transform(data_slice,fname, index_slice)

        return sample
class SliceDataset_val(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        transform: Optional[Callable] = None,
    ):
        if os.path.exists(str(root)+"/val_data_supervise.pickle"):
            with open(str(root)+"/val_data_supervise.pickle", "rb") as f:
                    self.examples = pickle.load(f)
        else:
            import glob
            paths=glob.glob(str(root)+"/*.nii.gz")
            self.examples=[]
            for path in paths:
                max_value=np.max(sitk.GetArrayFromImage(sitk.ReadImage(path)))
                for j in range(384):
                        self.examples.append((path,j,max_value))
            with open(str(root)+"/val_data_supervise.pickle", "wb") as f:
                        pickle.dump(self.examples, f)
        self.transform = transform
        print("val",len(self.examples))
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, index_slice,max_value = self.examples[i]
        data_slice=sitk.GetArrayFromImage(sitk.ReadImage(fname))

        if self.transform is None:
            sample = (data_slice, fname, index_slice,max_value)
        else:
            sample = self.transform(data_slice,fname, index_slice,max_value)

        return sample

class EdsrSample(NamedTuple):
    """
    Args:
        image: Subsampled image after inverse FFT.
        target: The target image (if applicable).
        mean: Per-channel mean values used for normalization.
        std: Per-channel standard deviations used for normalization.
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
    """

    image: torch.Tensor
    # target: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    fname: str
    slice_num: int
    max_value:float

def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)
def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)
def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, Union[torch.Tensor], Union[torch.Tensor]]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std

class EdsrDataTransform_train:

    def __init__(
        self,
        use_seed: bool = True,
    ):
        self.use_seed = use_seed
      
    def __call__(
        self,
        data_slice: np.ndarray,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        random_list=[0,1]
        direction=random.choice(random_list)
        if direction==0:
            slice=data_slice[:,slice_num,:]
        else:
            slice=data_slice[...,slice_num]
        hr_slice = to_tensor(slice.astype('float32'))
        hr_slice=hr_slice.transpose(1,0)
        return hr_slice.unsqueeze(0),fname,direction

class EdsrDataTransform_val:
    def __init__(
        self,
        use_seed: bool = True,
    ):
        self.use_seed = use_seed

    def __call__(
        self,
        data_slice: np.ndarray,
        fname: str,
        slice_num: int,
        max_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:

        slice=data_slice[:,slice_num,:]
        hr_slice = to_tensor(slice.astype('float32'))
        hr_slice=hr_slice.transpose(1,0)
        return hr_slice.unsqueeze(0)  
     
class EdsrDataTransform_train_b:
     def __init__(
         self,
         use_seed: bool = True,
     ):
         self.use_seed = use_seed  
     def __call__(
         self,
         data_slice: np.ndarray,
         fname: str,
         slice_num: int,
     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
         random_list=[0,1]
         direction=random.choice(random_list)
         if direction==0:
             slice=data_slice[:,slice_num,:]
         else:
             slice=data_slice[...,slice_num]
         lr_slice = to_tensor(slice.astype('float32'))
         lr_slice=lr_slice.transpose(1,0)
         return lr_slice,fname,direction     

class EdsrDataTransform_val_b:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        use_seed: bool = True,
    ):
        self.use_seed = use_seed

    def __call__(
        self,
        data_slice: np.ndarray,
        fname: str,
        slice_num: int,
        max_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        ##coronal
        #slice=data_slice[:,:,slice_num]
        ##axical
        slice=data_slice[:,slice_num,:]
        lr_slice = to_tensor(slice.astype('float32'))
        lr_slice=lr_slice.transpose(1,0)
        return lr_slice,fname,slice_num     
     
class Getdataloader:
    def __init__(
        self,
        data_path: Path,
        batch_size: int = 1,
        num_workers: int = 4,
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _create_data_loader(
        self,
        data_transform: Callable,
        data_partition: str,
    ) -> torch.utils.data.DataLoader:

        data_path = self.data_path
        if data_partition == "train":
            is_train = True
            dataset = SliceDataset_train(
                root=data_path+'/singlecoil_train/',
                transform=data_transform,
            )
        else:
            is_train = False
            dataset = SliceDataset_val(
                root=data_path+'/singlecoil_val/',
                transform=data_transform
            )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=is_train
        )
        return dataloader
    def _create_data_loader_b(
        self,
        data_transform: Callable,
        data_partition: str,
    ) -> torch.utils.data.DataLoader:

        data_path = self.data_path
        if data_partition == "train":
            is_train = True
            dataset = SliceDataset_train(
                root=data_path+'/singlecoil_train_b/',
                transform=data_transform,
            )
        else:
            is_train = False
            dataset = SliceDataset_val(
                root=data_path+'/singlecoil_val_b/',
                transform=data_transform
            )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=is_train
        )
        return dataloader


    def train_dataloader_a(self):
        train_transform=EdsrDataTransform_train()
        return self._create_data_loader(train_transform, data_partition="train")

    def val_dataloader_a(self):
        val_transform=EdsrDataTransform_val()
        return self._create_data_loader(
            val_transform, data_partition="val")

    def test_dataloader_a(self):
        val_transform=EdsrDataTransform_val()
        return self._create_data_loader(
            val_transform,data_partition="test",
        )
    def train_dataloader_b(self):
        train_transform=EdsrDataTransform_train_b()
        return self._create_data_loader_b(train_transform, data_partition="train")

    def val_dataloader_b(self):
        val_transform=EdsrDataTransform_val_b()
        return self._create_data_loader_b(
            val_transform, data_partition="val")

    def test_dataloader_b(self):
        val_transform=EdsrDataTransform_val_b()
        return self._create_data_loader_b(
            val_transform,data_partition="test",
        )
