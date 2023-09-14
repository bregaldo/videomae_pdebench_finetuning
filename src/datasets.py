import os
from torchvision import transforms
from torch.utils.data import Dataset
from transforms import *
from masking_generator import TubeMaskingGenerator, LastFrameMaskingGenerator
from kinetics import VideoClsDataset, VideoMAE
from ssv2 import SSVideoClsDataset
import h5py
import shutil
import utils
import time


class CustomNormalize(torch.nn.Module):
    def __init__(self, minv, maxv, dim_from_the_end=3, eps=1e-5):
        super().__init__()
        self.min = torch.Tensor(minv).float()
        self.max = torch.Tensor(maxv).float()
        self.dim_from_the_end = dim_from_the_end
        self.min = torch.reshape(self.min, (-1,) + (1,) * self.dim_from_the_end)
        self.max = torch.reshape(self.max, (-1,) + (1,) * self.dim_from_the_end)
        self.eps = eps

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.min) / ((self.max - self.min) + self.eps)

    def unnormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * ((self.max - self.min) + self.eps) + self.min

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.normalize(tensor)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min={self.min}, max={self.max})"

class DataAugmentationForPDEBench(object):
    def __init__(self, args):
        if args.data_set in ['compNS_turb', 'compNS_rand']:
            basic_stats = utils.get_pdebench_basic_stats(args.data_set)
            self.input_min = [basic_stats[field]['min'] for field in args.fields]
            self.input_max = [basic_stats[field]['max'] for field in args.fields]
        else:
            raise ValueError('Dataset name not recognized.')
        normalize = CustomNormalize(self.input_min, self.input_max)
        crop = transforms.CenterCrop(args.input_size)
        self.transform = transforms.Compose([crop, normalize])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type == 'last_frame':
            self.masked_position_generator = LastFrameMaskingGenerator(args.window_size)

    def __call__(self, images):
        process_data = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

class PDEBenchDataset(Dataset):
    def __init__(self, data_set,
                 transform=None,
                 fields=['Vx', 'Vy', 'density'],
                 timesteps=16,
                 random_start=True,
                 shuffle=False,
                 set_type='train',
                 split_ratios=(0.8, 0.1, 0.1),
                 split_seed=42,
                 data_tmp_copy=False,
                 local_rank=None):
        self.dataset_name = data_set
        self.timesteps = timesteps
        self.fields = fields # Possible choices are: 'Vx', 'Vy', 'density', 'pressure'
        assert len(self.fields) > 0, "At least one field must be specified."

        self.random_start = random_start
        self.shuffle = shuffle

        self.shape = None
        self.num_samples = None
        self.num_timesteps = None
        self.sample_shape = None

        self.transform = transform

        self.data_tmp_copy = data_tmp_copy
        self.local_rank = local_rank

        self.set_type = set_type
        self.split_ratios = split_ratios
        assert len(split_ratios) == 3, "Split ratios must be a tuple of length 3."
        assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0."
        self.split_seed = split_seed

        self.load_dataset()

    def load_dataset(self):
        data_base_path = "/mnt/home/gkrawezik/ceph/AI_DATASETS/PDEBench/2D/CFD/"
        data_rand_path = os.path.join(data_base_path, "2D_Train_Rand")
        data_turb_path = os.path.join(data_base_path, "2D_Train_Turb")
        #data_rand_filename = '2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5'
        data_rand_filename = '2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5'
        #data_turb_filename = '2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5'
        data_turb_filename = '2D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5'

        if self.dataset_name == 'compNS_turb':
            filename = os.path.join(data_turb_path, data_turb_filename)
            
        elif self.dataset_name == 'compNS_rand':
            filename = os.path.join(data_rand_path, data_rand_filename)
        else:
            raise ValueError('Dataset name not recognized.')
        print("Loading dataset file %s" % filename)
        if self.data_tmp_copy:
            if self.local_rank == 0:
                stime = time.time()
                print("Copying target file to /scratch...")
                shutil.copy(filename, '/scratch/')
                print("Done in %.2f seconds." % (time.time() - stime))
            elif self.local_rank is None:
                raise ValueError("local_rank must be specified if data_tmp_copy is True.")
            torch.distributed.barrier()
            self.file = h5py.File(os.path.join('/scratch/', filename.split('/')[-1]), 'r')
        else:
            self.file = h5py.File(filename, 'r')

        for field in self.fields:
            assert field in self.file.keys(), "Field %s not found in dataset." % field
        self.shape = self.file[self.fields[0]].shape
        for field in self.fields:
            assert self.file[field].shape == self.shape, "Fields must have same shape."
        
        self.num_samples = self.shape[0]
        self.num_timesteps = self.shape[1]
        self.sample_shape = self.shape[2:]

        assert self.num_timesteps >= self.timesteps, "Dataset has fewer timesteps than specified timesteps."

        print("Raw dataset %s has %d samples of shape %s and %d timesteps." % (self.dataset_name, self.num_samples, str(self.sample_shape), self.num_timesteps))

        # Split dataset according to split_ratios
        self.split_indices = {}
        indices = np.arange(self.num_samples)
        rng = np.random.default_rng(seed=self.split_seed)
        if self.shuffle:
            rng.shuffle(indices)
        self.split_indices['train'] = indices[:int(self.num_samples*self.split_ratios[0])]
        self.split_indices['val'] = indices[int(self.num_samples*self.split_ratios[0]):int(self.num_samples*(self.split_ratios[0]+self.split_ratios[1]))]
        self.split_indices['test'] = indices[int(self.num_samples*(self.split_ratios[0]+self.split_ratios[1])):]
    
    def __len__(self):
        return len(self.split_indices[self.set_type])
    
    def __getitem__(self, idx):
        sample_idx = self.split_indices[self.set_type][idx]
        if self.random_start:
            start_idx = np.random.randint(0, self.num_timesteps-self.timesteps+1)
        else:
            start_idx = 0
        end_idx = start_idx + self.timesteps
        sample = torch.zeros((len(self.fields), self.timesteps, *self.sample_shape))
        for i, field in enumerate(self.fields):
            sample[i] = torch.tensor(self.file[field][sample_idx, start_idx:end_idx])
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample

class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pdebench_dataset(args, set_type='train'):
    transform = DataAugmentationForPDEBench(args)
    train_split_ratio = args.train_split_ratio
    test_split_ratio = args.test_split_ratio
    assert train_split_ratio + test_split_ratio <= 1.0
    val_split_ratio = 1.0 - train_split_ratio - test_split_ratio
    dataset = PDEBenchDataset(args.data_set,
                              fields=args.fields,
                              split_ratios=(train_split_ratio, val_split_ratio, test_split_ratio),
                              set_type=set_type,
                              timesteps=args.num_frames,
                              transform=transform,
                              data_tmp_copy=args.data_tmp_copy,
                              local_rank=args.gpu)
    return dataset

def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
