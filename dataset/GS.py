import torch # type: ignore
import gin # type: ignore
import numpy as np # type: ignore
from typing import Optional, Union
import pickle
import cv2, os # type: ignore
from utils.transform_utils import remove_outliers, MinMaxScaler
from dataset import colmap_utils
from utils import gs_utils
import glob, yaml # type: ignore
from pathlib import Path
import random
from PIL import Image # type: ignore
from time import time
import math


SF = False
ALPHA=0.3 # 0.1     0.3     0.5     0.7     0.9
RATIO = 1.0 #

#torch.manual_seed(0)

@gin.configurable
class SplatfactoDataset(torch.utils.data.IterableDataset):
    def __init__(self, 
                 train_or_test,
                 nerfstudio_folder,
                 colmap_folder,
                 load_pose_src, #[colmap or nerfstudio]
                 sample_ratio_test: Optional[float],
                 image_per_scene: Optional[int],
                 remove_outlier_ndevs: float,
                 max_gs_num: int,
                 cache_steps: int,
                 cache_num_scenes: int, #Default: cache_num_scenes=1, cache_steps=1
                 split_across_gpus: bool,
                 background_color: list=[0,0,0],
                 ):
        self.test_id=0
        self.train_or_test = train_or_test
        self.image_per_scene = image_per_scene
        self.sample_ratio_test = sample_ratio_test  

        self.nerfstudio_folders = sorted([os.path.join(nerfstudio_folder, ls, 'splatfacto') for ls in os.listdir(nerfstudio_folder)])
        #print(colmap_folder)
        if colmap_folder.endswith('.txt'):
            self.colmap_folders = [os.path.join(colmap_path, ls) for ls in open(colmap_folder).read().splitlines()] # type: ignore
        elif os.path.isdir(colmap_folder):
            self.colmap_folders = sorted([os.path.join(colmap_folder, ls) for ls in os.listdir(colmap_folder)])

        assert len(self.nerfstudio_folders) == len(self.colmap_folders), 'The number of folders in nerfstudio and colmap should be the same'
        self.folders = list(zip(self.nerfstudio_folders, self.colmap_folders))
        self.load_pose_src = load_pose_src

        self.remove_outlier_ndevs = remove_outlier_ndevs
        self.cache_steps, self.cache_num_scenes = cache_steps, cache_num_scenes
        self.split_across_gpus = split_across_gpus
        self.max_gs_num = max_gs_num
        self.cache_scenes = []
        self.background_color = background_color
        self.gs_params= None

        if train_or_test in ['test']: 
            # For test set, we need to split data across device deterministically
            self.remaining_scenes = list(range(len(self.folders)))
            assert self.cache_num_scenes==1 and self.cache_steps==1, 'For test, we do not cache'
            # For DDP evaluation, we need to chunk the data
            try:
                world_size = torch.cuda.device_count()
                rank = torch.distributed.get_rank()
            except:
                world_size, rank = 1, 0
            chunk_size = len(self.remaining_scenes)//world_size
            if rank == world_size-1:
                self.remaining_scenes = self.remaining_scenes[rank*chunk_size:]
            else:
                self.remaining_scenes = self.remaining_scenes[rank*chunk_size:(rank+1)*chunk_size]
        else:
            self.counter = 0

    def refresh_remaining_training(self,):
        if self.split_across_gpus:
            self.random_split_to_remaining()
        else:
            self.remaining_scenes = self.get_thisworker_split(N=len(self.folders))
            random.shuffle(self.remaining_scenes)
        self.counter += 1
        return

    def get_thisworker_split(self, N):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return list(range(N))
        per_worker = N // worker_info.num_workers
        worker_id = worker_info.id
        if worker_id == worker_info.num_workers-1:
            return list(range(worker_id*per_worker, N))
        else:
            return list(range(worker_id*per_worker, (worker_id+1)*per_worker))
        
    def random_split_to_remaining(self, ):
        """
        #Generate a new permutation of the folders
        """
        # 1. Split self.folders across processes
        np.random.seed(torch.distributed.get_rank())
        torch.manual_seed(torch.distributed.get_rank())
        rng_state = np.random.get_state()
        world_size = torch.cuda.device_count()
        rank = torch.distributed.get_rank()     
        np.random.seed(self.counter)
        #Pad to world_size*k
        permutation = np.random.permutation(len(self.folders))
        pad_num = world_size - len(self.folders)%world_size
        if pad_num > 0 and world_size > 1:
            permutation = np.concatenate([permutation, permutation[:pad_num]])
        np.random.set_state(rng_state)

        chunk_size = len(permutation)//world_size
        if rank == world_size-1:
            remaining_scenes_for_thisprocess = permutation[rank*chunk_size:]
        else:
            remaining_scenes_for_thisprocess = permutation[rank*chunk_size:(rank+1)*chunk_size]

        # 2. Split scenes across works
        split_id = self.get_thisworker_split(N=len(remaining_scenes_for_thisprocess))
        self.remaining_scenes = [remaining_scenes_for_thisprocess[i] for i in split_id]
        worker_info = torch.utils.data.get_worker_info()
        return 

    @gin.configurable
    def read_image(self, path, background):
        try:
            pil_image = Image.open(path)
        except:
            print(f'Warning: {path} cannot be opened')

        image = np.array(pil_image, dtype="uint8").astype(np.float32) / 255.0
        if 'real' in path.lower():
            possible_mask_filename = path.replace('images','masks') #TODO: hardcoded
            if os.path.exists(possible_mask_filename):
                mask = np.array(Image.open(possible_mask_filename)).astype(image.dtype)/255.0
                mask = torch.from_numpy(mask)
        else:
            mask = None

        image = torch.from_numpy(image)
        if image.shape[2] == 4:
            image = image[:, :, :3] * image[:, :, -1:] + background * (1.0 - image[:, :, -1:])
        elif mask is not None:
            image_rgb = image * mask[...,None] + background * (1.0 - mask[...,None])
            # As we need to preserve the mask for evaluation, we save the image RGBA
            image = torch.concat([image_rgb, mask[...,None]], axis=-1) # Hardcoded here, only for the real dataset
        return image

    def load_gs_params_fromnerfstudio(self, nerfstudio_dir, idx):
        skip_params = gin.query_parameter('FeaturePredictor.input_features')
        if gin.query_parameter("training.pretrain_steps") > 0:
            skip_params = skip_params+gin.query_parameter('create_pseudo_target.take_from_input')
        try:
            ckpt_file = glob.glob(nerfstudio_dir + '/nerfstudio_models/step-*.ckpt')[-1] #Take the last checkpoint
        except:
            print(f'Warning: {nerfstudio_dir} does not have nerfstudio_models/step-*.ckpt')
            exit()
        ckpt = torch.load(ckpt_file, map_location='cpu')['pipeline']
        # print(ckpt_file)
        
        ckpt = {k.replace('_model.gauss_params.',''):v for k,v in ckpt.items() if 'gauss_params' in k}
        gs_params = {k:ckpt[k] for k in set(skip_params)}
        # print(gs_params['means'].shape[0])

        # Remove inf or nan
        select = torch.ones(gs_params['means'].shape[0], dtype=torch.bool)
        for key in gs_params:
            if key=='features_rest':
                select = select & ~torch.isnan(gs_params[key].sum(dim=1)).any(dim=1)
            else:
                select = select & ~torch.isnan(gs_params[key]).any(dim=1)
        for key in gs_params:
            gs_params[key] = gs_params[key][select]

        # Filter the outliers
        if self.remove_outlier_ndevs > 0:
            _,inlier_mask = remove_outliers(gs_params['means'], n_devs=self.remove_outlier_ndevs)
            for key in gs_params:
                gs_params[key] = gs_params[key][inlier_mask]

        # Truncate gs params if num > self.max_gs_num
        N = gs_params['means'].shape[0]
        # print(N)
        # print('*********')
        #print(self.train_or_test)
        # if self.train_or_test == "train":
        #     ratio = random.choice([0.3,0.2,0.1])
        # else:
        #     ratio = 0.2
        
        
        #print(ratio)
        self.max_gs_num = int(N*RATIO)
        
        # if self.max_gs_num <= 1:
        #     self.max_gs_num = int(N*self.max_gs_num)
            
       
        if N > self.max_gs_num:
            inlier_mask = torch.zeros(N, dtype=torch.bool)

            if SF == True:
                inlier_mask[:self.max_gs_num] = True
            else:
            
                opacities_norm = ((gs_params['opacities'] - gs_params['opacities'].mean()) / (gs_params['opacities'].std() + 1e-6)).squeeze(1)
                scales_sum = (4/3 * math.pi) * torch.prod(gs_params['scales'] + 1e-6, dim=-1) #torch.sum(gs_params['scales'], dim=1)
                #scales_sum = torch.sum(gs_params['scales'], dim=1)
                scales_norm = (scales_sum - scales_sum.mean()) / (scales_sum.std() + 1e-6)
                #rank_tensor = 0.8*opacities_norm + 0.2*scales_norm
                rank_tensor = ALPHA*opacities_norm + (1-ALPHA)*scales_norm #0.50.5
                sorted_indices = torch.argsort(rank_tensor, descending=True)
                keep_indices = sorted_indices[:self.max_gs_num] 


                inlier_mask[keep_indices] = True

            

            # gen = torch.Generator()
            # gen.manual_seed(torch.seed())

            # keep_indices = torch.randperm(N, generator=gen)[:self.max_gs_num]
            # print(random_indices)
            # if self.train_or_test == 'test':
            #    print(gs_params['means'].shape)

            #--------------------------------
            
            for key in gs_params:
                gs_params[key] = gs_params[key][inlier_mask]

            random_indices = None #keep_indices #None #
            #---------------------------------
        
        else:
            random_indices=None
        # Normalize the means and scales (we need to use the scaler to transform the camera later)
        scaler = MinMaxScaler()
        
        gs_params['means'] = scaler.fit_transform(gs_params['means']) 
        gs_params['scales'] = gs_params['scales'] + torch.log(scaler.scale_)

        inf_mask = torch.isinf(gs_params['scales']).sum(dim=1).bool()
        valid_mask = (~inf_mask).bool()
        inrange_mask = torch.all((gs_params['means'] >= 0) & (gs_params['means'] <= 1), dim=1)
        valid_mask = valid_mask & inrange_mask
        for key in gs_params:
            gs_params[key] = gs_params[key][valid_mask]
            if torch.isnan(gs_params[key]).any():
                print(f'Warning: {key} contains nan', nerfstudio_dir)

        # print(gs_params['means'].shape[0])
        # print('----')
        return gs_params, scaler, random_indices

    def load_images_cameras_fromnerfstudio(self, nerfstudio_dir, colmap_dir):
        with open(nerfstudio_dir + '/camera_for-3d-denoise.pkl', 'rb') as f:
            meta = pickle.load(f)
        train_imgs_path, test_imgs_path = [], []

        # print(nerfstudio_dir.split('/')[-2])
        try:
            #images_test_filename = TEST_dict[nerfstudio_dir.split('/')[-2]].split(',')
            
            images_test_filename = []
            # eval_lists
            with open(os.path.join(colmap_dir,'eval_lists.txt')) as f:
                for line in f.readlines():
                    if line!='':
                        images_test_filename.append(line.split('/')[-1].split('.')[0])
        except:
            images_test_filename = []
        # print(images_test_filename)
        # print(len(images_test_filename))
        image_names = os.listdir(colmap_dir + '/images')

        TESTSET_ELEVATION = False
        import re
        def natural_sort_key(text):
            return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]

        
        # for i, name in enumerate(sorted(image_names)): # TODO [The order is not necessarily aligned with the camera_to_worlds]
        for i, name in enumerate(sorted(image_names, key=natural_sort_key)):
            if self.train_or_test == 'test':
                #test_imgs_path.append(os.path.join(colmap_dir, 'images', name)) #---
                if name.split('.')[0] in images_test_filename:
                    test_imgs_path.append(os.path.join(colmap_dir, 'images', name))
            else:
                if name.split('.')[0] in images_test_filename: #'elevation' in name: # Hardcoded: test set (TODO)
                    continue #---
                    #test_imgs_path.append(os.path.join(colmap_dir, 'images', name))
                else:
                    train_imgs_path.append(os.path.join(colmap_dir, 'images', name))

        # Check: align the order of camer poses with the images 
        # print(test_imgs_path)
        # print(len(test_imgs_path))
        if TESTSET_ELEVATION:
            meta['test_camera_to_worlds'] = meta['test_camera_to_worlds'][-3*3:]
        # if ood_test_img_names!=None:
        #     ood_ids = [i for i, path in enumerate(test_imgs_path) if os.path.basename(path) in ood_test_img_names]
        #     # selected test_imgs_path and test_camera_to_worlds
        #     test_imgs_path = [test_imgs_path[i] for i in ood_ids]
        #     meta['test_camera_to_worlds'] = meta['test_camera_to_worlds'][ood_ids]
        return meta, train_imgs_path, test_imgs_path

    def load_images_cameras_fromcolmap(self, colmap_dir):
        recon_dir = Path(os.path.join(colmap_dir, "sparse/0"))
        if (recon_dir / "cameras.txt").exists():
            cam_id_to_camera = colmap_utils.read_cameras_text(recon_dir / "cameras.txt")
            im_id_to_image = colmap_utils.read_images_text(recon_dir / "images.txt")
        elif (recon_dir / "cameras.bin").exists():
            cam_id_to_camera = colmap_utils.read_cameras_binary(recon_dir / "cameras.bin")
            im_id_to_image = colmap_utils.read_images_binary(recon_dir / "images.bin")
        else:
            raise ValueError(f"Could not find cameras.txt or cameras.bin in {recon_dir}")

        # Parse cameras
        assert len(cam_id_to_camera) == 1, "Only one camera is supported"
        for cam_id, cam_data in cam_id_to_camera.items():
            camera = colmap_utils.parse_colmap_camera_params(cam_data)
        assert camera['camera_model'] in ['SIMPLE_PINHOLE', 'PINHOLE'], "Only pinhole camera is supported"

        meta = {
            'fx':camera['fl_x'], 'fy':camera['fl_y'],
            'cx':camera['cx'], 'cy':camera['cy'],
            'width':camera['w'], 'height':camera['h']
        }

        for key in meta:
            meta[key] = torch.tensor(meta[key], dtype=torch.float32)
        c2ws, image_names = [], []

        ordered_im_id = sorted(im_id_to_image.keys(), key=lambda x: im_id_to_image[x].name)
        for im_id in ordered_im_id:
            im_data = im_id_to_image[im_id]
            # NB: COLMAP uses Eigen / scalar-first quaternions
            # * https://colmap.github.io/format.html
            # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
            # the `rotation_matrix()` handles that format for us.
            rotation = colmap_utils.qvec2rotmat(im_data.qvec)
            translation = im_data.tvec.reshape(3, 1)
            w2c = np.concatenate([rotation, translation], 1)
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
            c2w = np.linalg.inv(w2c)
            # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
            c2w[0:3, 1:3] *= -1
            c2ws.append(c2w)
            image_names.append(im_data.name) #test_XXX.png

        poses = torch.from_numpy(np.array(c2ws).astype(np.float32))
        train_poses, test_poses = [], []
        train_imgs_path, test_imgs_path = [], []
        for i, name in enumerate(image_names):
            
            if name.startswith('test'):
                test_poses.append(poses[i])
                test_imgs_path.append(os.path.join(colmap_dir, 'images', name))
            else:
                train_poses.append(poses[i])
                train_imgs_path.append(os.path.join(colmap_dir, 'images', name))
        if len(train_poses)!=0:
            meta['train_camera_to_worlds'] = torch.stack(train_poses, dim=0)
        else:
            print("Warning: No training images, set the 1st test poses as training poses as placeholder")
            meta['train_camera_to_worlds'] = torch.stack(test_poses[:1], dim=0)
            train_imgs_path = test_imgs_path[:1]
        meta['test_camera_to_worlds'] = torch.stack(test_poses, dim=0)
        return meta, train_imgs_path, test_imgs_path

    def load_scene(self, idx):
        nerfstudio_dir, colmap_dir = self.folders[idx]
        #if self.gs_params == None:
        #    self.gs_params  = self.load_gs_params_fromnerfstudio(nerfstudio_dir, idx)
        gs_params, scaler,random_indices = self.load_gs_params_fromnerfstudio(nerfstudio_dir, idx) #self.gs_params
        
        if self.load_pose_src == 'colmap':
            meta, train_imgs_path, test_imgs_path = self.load_images_cameras_fromcolmap(colmap_dir)
        if self.load_pose_src == 'nerfstudio':
            meta, train_imgs_path, test_imgs_path = self.load_images_cameras_fromnerfstudio(nerfstudio_dir, colmap_dir)
            
            
        meta['train_camera_to_worlds'][:,:3,-1] = scaler.transform(meta['train_camera_to_worlds'][:,:3,-1])
        meta['test_camera_to_worlds'][:,:3,-1] = scaler.transform(meta['test_camera_to_worlds'][:,:3,-1])

        outputs = {'gs_params': gs_params, 'meta': meta, 'idx': idx, 'indices':random_indices,
                'scene_name': nerfstudio_dir.split('/')[-2], #The basename is splatfacto
                'train_imgs_path': train_imgs_path,
                'test_imgs_path': test_imgs_path}
        return outputs

    def get_scene_from_cache(self):
        # 1. If the cache is not full, we just append the new data
        if len(self.cache_scenes) < self.cache_num_scenes:
            idx = self.remaining_scenes.pop(0)
            #print(self.remaining_scenes)
            if self.train_or_test=='train' and len(self.remaining_scenes)==0:
                self.refresh_remaining_training()
            new_scene = self.load_scene(idx)
            if self.cache_steps!=1: #cache_steps=-1: cache forever, cache_steps>1: cache for more than one step
                self.cache_scenes.append([new_scene,1]) # Set the counter to 1, otherwise keep the cache empty
            return new_scene
        else:
            # 2. If the cache is full, 
            scene_i = random.randint(0, len(self.cache_scenes)-1) # We need not to worry about the test set, since cache_num_scenes=1
            scene  = self.cache_scenes[scene_i][0]
            self.cache_scenes[scene_i][1] += 1
            if self.cache_scenes[scene_i][1] == self.cache_steps: #cache_steps=-1 means we cache forever
                self.cache_scenes.pop(scene_i)
                self.get_scene_from_cache()
            return scene

    def __iter__(self):
        if self.train_or_test == 'train':
            self.refresh_remaining_training()
        if len(self.remaining_scenes) < self.cache_num_scenes:
            print(f'Warning: The number of scenes is less than the cache_num_scenes, {len(self.remaining_scenes)} < {self.cache_num_scenes}')
            self.cache_num_scenes = len(self.remaining_scenes)
            print(f'cache_num_scenes is set to {self.cache_num_scenes}')
        while len(self.remaining_scenes) > 0:
            scene = self.get_scene_from_cache()
            gs_params, meta, scene_name = scene['gs_params'], scene['meta'], scene['scene_name']
            mask_indices = scene['indices']
            train_imgs_path, test_imgs_path = scene['train_imgs_path'], scene['test_imgs_path']
            
            train_imgs_name = [os.path.basename(path) for path in train_imgs_path]
            test_imgs_name = [os.path.basename(path) for path in test_imgs_path]
            #print(test_imgs_name)
            total_train_num, total_test_num = len(meta['train_camera_to_worlds']), len(meta['test_camera_to_worlds'])
            cameras = {}
            total_test_num = len(test_imgs_path)
            self.total_test_num = total_test_num
            if self.train_or_test == 'train':
                sample_test = np.random.rand(self.image_per_scene) < self.sample_ratio_test
                sample_test_num = 0 #min(np.sum(sample_test), total_test_num) #previously here it's max
                sample_train_num = self.image_per_scene - sample_test_num
                sample_train_num = min(sample_train_num, total_train_num)
                images, images_names = [], []
                cameras['camera_to_worlds'] = []
                #decide background_color
                # train_cam_ids = np.random.permutation(total_train_num)[:sample_train_num]
                
                if self.background_color == 'random':
                    background = torch.rand(3)
                else:
                    background = torch.tensor(self.background_color)/255.
                if sample_train_num > 0: #TODO not enough training or test views

                    rng = np.random.default_rng()  
                    train_cam_ids = rng.permutation(total_train_num)[:sample_train_num]

                    # train_cam_ids = np.random.permutation(total_train_num)[:sample_train_num]

                    #print(train_cam_ids)
                    # print(len(train_imgs_path))

                    images.extend([self.read_image(train_imgs_path[i], background=background) for i in train_cam_ids])
                    images_names.extend([train_imgs_name[i] for i in train_cam_ids])
                    cameras['camera_to_worlds'].append(meta['train_camera_to_worlds'][train_cam_ids])
                    
                if sample_test_num > 0:
                    rng = np.random.default_rng()  
                    test_cam_ids = rng.permutation(total_test_num)[:sample_test_num]
                    # test_cam_ids = np.random.permutation(total_test_num)[:sample_test_num]
                    images.extend([self.read_image(test_imgs_path[i], background=background) for i in test_cam_ids])
                    images_names.extend([test_imgs_name[i] for i in test_cam_ids])
                    cameras['camera_to_worlds'].append(meta['test_camera_to_worlds'][test_cam_ids])

                    print(meta['test_camera_to_worlds'][test_cam_ids])
                cameras['camera_to_worlds'] = torch.concatenate(cameras['camera_to_worlds'], axis=0)
                
                

                for key in ['fx','fy','cx','cy','width','height']:
                    cameras[key] = meta[key]
                # actual_image = images[0]  # Get the actual loaded image
                # actual_height, actual_width = actual_image.shape[:2]

                # scale_x = actual_width / meta['width'].item()
                # scale_y = actual_height / meta['height'].item()

                # cameras['fx'] = meta['fx'] * scale_x
                # cameras['fy'] = meta['fy'] * scale_y
                # cameras['cx'] = meta['cx'] * scale_x
                # cameras['cy'] = meta['cy'] * scale_y
                # cameras['width'] = torch.tensor(actual_width, dtype=torch.float32)
                # cameras['height'] = torch.tensor(actual_height, dtype=torch.float32)

                cameras['background_color'] = background
                output_dict = {'gs_params': gs_params, 'images': images, 'cameras': cameras, 
                            'scene_idx': scene['idx'],
                            'scene_name': scene_name} #..../XX/splatfacto
                output_dict['images_name'] = images_names
                output_dict['indices']=mask_indices
                # print(f'prep: {images_names}, {actual_height}, {actual_width}')
                yield output_dict

            elif self.train_or_test == 'test':
                
                assert self.background_color!='random', 'For test set, background_color cannot be random'
                background = torch.tensor(self.background_color)/255.
                #test_cam_ids = np.arange(total_test_num) #We take all the test images
                # print(total_test_num)
                for test_id in range(total_test_num):
                    
                    test_cam_ids = [test_id]
                    images = [self.read_image(test_imgs_path[i], background=background) for i in test_cam_ids]

                    images_names = [test_imgs_name[i] for i in test_cam_ids]
                    cameras['camera_to_worlds'] = meta['test_camera_to_worlds'][test_cam_ids] 
                    
                    for key in ['fx','fy','cx','cy','width','height']:
                        cameras[key] = meta[key]

                    # actual_image = images[0]  # Get the actual loaded image
                    # actual_height, actual_width = actual_image.shape[:2]

                    # Adjust intrinsics proportionally
                    # scale_x = actual_width / meta['width'].item()
                    # scale_y = actual_height / meta['height'].item()

                    # cameras['fx'] = meta['fx'] * scale_x
                    # cameras['fy'] = meta['fy'] * scale_y
                    # cameras['cx'] = meta['cx'] * scale_x
                    # cameras['cy'] = meta['cy'] * scale_y
                    # cameras['width'] = torch.tensor(actual_width, dtype=torch.float32)
                    # cameras['height'] = torch.tensor(actual_height, dtype=torch.float32)
                    
                    # print(f"Adjusted intrinsics: fx={cameras['fx']:.2f}, fy={cameras['fy']:.2f}, cx={cameras['cx']:.2f}, cy={cameras['cy']:.2f}")

                    cameras['background_color'] = background
                    output_dict = {'gs_params': gs_params, 'images': images, 'cameras': cameras, 
                                'scene_idx': scene['idx'],
                                'scene_name': scene_name} #..../XX/splatfacto
                    output_dict['images_name'] = images_names
                    output_dict['indices']=mask_indices
                    output_dict['total_test_num']=total_test_num
                    output_dict['remain_scenes'] = len(self.remaining_scenes)

                    yield output_dict

            else:
                raise ValueError
            
            
            
            
