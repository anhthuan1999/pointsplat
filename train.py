import torch, os # type: ignore
import torch.nn as nn # type: ignore
from tqdm import tqdm # type: ignore
import numpy as np # type: ignore
import sys
from torch.optim import SGD # type: ignore
import wandb, json # type: ignore
import argparse
import cv2 # type: ignore
from models.feature_predictor import FeaturePredictor

from collections import OrderedDict
import random
import gin # type: ignore
from absl import app, flags # type: ignore
from dataset.Loader import build_trainloader, build_testloader
from utils import gpu_utils, gs_utils, loss_utils
from utils.optimizers import build_optimizer, build_scheduler
from utils.metrics import MetricComputer
from utils.log_utils import ProcessSafeLogger
from utils.metrics import psnr
from torch.nn.parallel import DistributedDataParallel as DDP # type: ignore
import torch.distributed as dist # type: ignore
from pytorch_msssim import SSIM # type: ignore

APPLY_LPIPS = 0
REMOVE_LPIPS = 15000

flags.DEFINE_string('output_dir', 'output', 'Output directory')
flags.DEFINE_string('eval_subdir', 'eval_final', 'Eval subdirectory')
flags.DEFINE_string('wandb_dir', './wandb', 'Wandbs Output directory')
flags.DEFINE_boolean('only_eval', False, 'eval or train')
flags.DEFINE_boolean('compare_with_input', False, 'Compare with input') #for evaluation
flags.DEFINE_multi_string(
  'gin_file', None, 'List of paths to the config files.')
flags.DEFINE_multi_string(
  'gin_param', '', 'Newline separated list of Gin parameter bindings.')

FLAGS = flags.FLAGS


@gin.configurable
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_grid(imgs, nrows=3, ncols=3): #ncols=3, nrow=3
    img_h, img_w = imgs[0].shape[:2]
    if imgs[0].ndim == 3:
        grid = np.zeros((img_h*nrows, img_w*ncols, 3), dtype=np.uint8)
    elif imgs[0].ndim == 2:
        grid = np.zeros((img_h*nrows, img_w*ncols), dtype=np.uint8)
    for i in range(nrows):
        for j in range(ncols):
            if i*ncols+j >= len(imgs):
                break
            grid[i*img_h:(i+1)*img_h, j*img_w:(j+1)*img_w] = imgs[i*ncols+j]
    return grid


# NUM_TEST = 20
# Home_005_1: 71
# Home_006_1: 207
# Home_008_1: 81
# Office_001_1: 115
# Room 0,1, Office 3: 8
# Office 4: 7
# ebff4de90b: 20

@gin.configurable
def evaluation(model, test_loader, output_dir, output_gt, compare_with_pseudo, 
              compare_with_input=False,
              save_as_single=False,
              evaluate_input=False,output_pred=True):
    # try:
    if True:
      model.eval()
      metric_computer = MetricComputer()
      if compare_with_input:
        metric_computer_input = MetricComputer()
      os.makedirs(output_dir, exist_ok=True)
      with torch.no_grad():
        cnt = 0
        num_images, num_scenes = 0, 0
        pseudo_loss = {}
        #print(test_loader.total_test_num)
        test_iterator = iter(test_loader)
        #for test_batch in tqdm(test_loader): #A scene at a time
        test_idx=0
        total_test_num = 0
        remain_scenes=-1
        prev_scene = ''
        while True:
          
          if test_idx == total_test_num and test_idx!=0 and remain_scenes==0:
            break

          try:
            test_batch = next(test_iterator)
          except StopIteration:
            test_batch = next(test_iterator)

          #if test_idx == 0:
          if evaluate_input:
            output_pred = True

          if remain_scenes!=test_batch[0]['remain_scenes']:
            total_test_num += test_batch[0]['total_test_num']
            remain_scenes =   test_batch[0]['remain_scenes']
            
            scene_name =   [test_batch[0]['scene_name']]

            test_batch_gs = [test_batch[0]['gs_params']]#.copy()
            test_batch_gs = gpu_utils.move_to_device(test_batch_gs,model.device)

            #print(total_test_num)
          
          
            #------------------------------------------------
            # if test_idx==0:
            #   scene_gs = test_batch[0]['gs_params'].copy()
            #   scene_gs = gpu_utils.move_to_device(scene_gs,model.device)
            
            # test_batch_gs_split={}
            # if test_batch[0]['indices'] != None:
            #   for data in test_batch:
            #     mask_indices = data['indices']
            #     for key in scene_gs.keys():
            #       tensor = scene_gs[key].clone()
            #       test_batch_gs_split[key] = tensor[mask_indices]
            #   test_batch_gs_split = [test_batch_gs_split]
            # else:
            #   test_batch_gs_split = [data['gs_params'] for data in test_batch]
            # del scene_gs
            # if prev_scene != test_batch[0]['scene_name']:

            # test_batch_gs_split = [data['gs_params'] for data in test_batch]
            
            #------------------------------------------------


            # test_batch_gs = gpu_utils.move_to_device(test_batch_gs_split,model.device)
            #print(test_batch[0]['cameras'].shape)
            
            if evaluate_input == True:
              forward_kwargs = {'batch_normalized_gs': test_batch_gs, 'batch_scene_idx': -9}
            else:
              forward_kwargs = {'batch_normalized_gs': test_batch_gs, 'batch_scene_idx': -1}
            out_test_batch_gs = model(**forward_kwargs)
            # #-------------------------------------------------------------------
            # if test_batch[0]['indices'] != None:
            #   out_test_batch_gs_split={}
            #   for data in test_batch:
            #     mask_indices = data['indices']
            #     for key in scene_gs.keys():
            #       tensor = scene_gs[key].clone()
            #       tensor[mask_indices] = out_test_batch_gs[0][key]
            #       out_test_batch_gs_split[key] = tensor
            #   #print(out_test_batch_gs_split['means'].shape)
            #   out_test_batch_gs = [out_test_batch_gs_split]
            #-------------------------------------------------------------------
            # prev_scene = test_batch_name[0]
            # print(prev_scene)

          test_batch_cameras = gpu_utils.move_to_device([data['cameras'] for data in test_batch],model.device)
          test_batch_images = gpu_utils.move_to_device([data['images'] for data in test_batch],model.device)
          test_batch_idx = [data['scene_idx'] for data in test_batch]
          # test_batch_name = [data['scene_name'] for data in test_batch]
          test_batch_imgname = [data['images_name'] for data in test_batch]

          for iii, (out_gs, in_gs, cameras, gt_imgs, scene_idx) in enumerate(zip(out_test_batch_gs, test_batch_gs, test_batch_cameras, test_batch_images, test_batch_idx)):
            #print(scene_idx)
            scene_idx = test_idx
            if evaluate_input:
              pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(in_gs, cameras, gt_imgs)
            else:
              pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(out_gs, cameras, gt_imgs) # List of torch.tensor([H,W,3])

            pred_imgs = torch.stack(pred_imgs, dim=0) #torch.tensor([N,H,W,3])
            gt_imgs = torch.stack(gt_imgs, dim=0) #torch.tensor([N,H,W,3])
            # print('Hi')
            # print(gt_imgs.shape)
            if gt_imgs.shape[-1] == 4:
              # only for real images
              masks = gt_imgs[...,3].unsqueeze(-1)
              pred_imgs = pred_imgs*masks
              gt_imgs = (gt_imgs[...,:3]*255).to(torch.uint8)
              pred_imgs = (pred_imgs*255).to(torch.uint8)
            else:
              masks = None
              gt_imgs = (gt_imgs*255).to(torch.uint8)
              pred_imgs = (pred_imgs*255).to(torch.uint8)

            imgs = [im.cpu().numpy().astype(np.uint8) for im in pred_imgs]
            if output_pred:
              grid = make_grid(imgs, nrows=1, ncols=1)
              grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
              cv2.imwrite(os.path.join(output_dir, f'scene_{scene_name[0]}_{scene_idx}_pred.png'), grid)
            
            if output_gt:
              gt_imgs_ = [im.cpu().numpy().astype(np.uint8) for im in gt_imgs]
              grid = make_grid(gt_imgs_, nrows=1, ncols=1)
              grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
              cv2.imwrite(os.path.join(output_dir, f'scene_{scene_name[0]}_{scene_idx}_gt.png'), grid)

            metric_computer.update(pred_imgs, gt_imgs, name=f'{scene_name[0]}_{scene_idx}')
            if compare_with_input:
              input_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(in_gs, cameras, gt_imgs)
              input_imgs = torch.stack(input_imgs, dim=0)
              if masks is not None:
                input_imgs = input_imgs*masks
                input_imgs = (input_imgs*255).to(torch.uint8)
              else:
                input_imgs = (input_imgs*255).to(torch.uint8)

              #------
              # H = min(input_imgs.shape[1], gt_imgs.shape[1])
              # W = min(input_imgs.shape[2], gt_imgs.shape[2])

              # input_imgs = input_imgs[:,:H, :W, :]
              # gt_imgs    = gt_imgs[:,:H, :W, :]
              #-----

              metric_computer_input.update(input_imgs, gt_imgs, name=f'{scene_name[0]}_{scene_idx}')
              output_dir_thisscene = os.path.join(output_dir, f'compare/{scene_name[iii]}')
              os.makedirs(output_dir_thisscene, exist_ok=True)

              #save ([gt, input, pred])
              for ii, (gt_img, input_img, pred_img) in enumerate(zip(gt_imgs, input_imgs, pred_imgs)):
                gt_img = gt_img.cpu().numpy().astype(np.uint8)
                input_img = input_img.cpu().numpy().astype(np.uint8)
                pred_img = pred_img.cpu().numpy().astype(np.uint8)
                cmp_img = np.concatenate([gt_img, input_img, pred_img], axis=1) #H,W*3
                cv2.imwrite(os.path.join(output_dir_thisscene, test_batch_imgname[iii][ii]), cmp_img[:,:,::-1])

            if save_as_single:
              output_dir_thisscene_single = os.path.join(output_dir, f'pred/{scene_name[iii]}')
              os.makedirs(output_dir_thisscene_single, exist_ok=True)
              for ii,pred_img in enumerate(pred_imgs):
                pred_img = pred_img.cpu().numpy().astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir_thisscene_single, test_batch_imgname[iii][ii]), pred_img[:,:,::-1])
            
            cnt += 1
            num_images += len(pred_imgs)
            # print(metric_computer)
          test_idx+=1
        metrics = metric_computer.sum() # We need to sum the metrics
        metric_computer.write_to_file(os.path.join(output_dir, f'metrics.rank{dist.get_rank()}.json'))
        if compare_with_input:
          metric_computer_input.write_to_file(os.path.join(output_dir, f'metrics_input.rank{dist.get_rank()}.json'))
      model.train()

      num_images = torch.tensor([num_images]).to(model.device)
      num_scenes = torch.tensor([num_scenes]).to(model.device)
      torch.distributed.reduce(num_images, dst=0) #Sum
      torch.distributed.reduce(num_scenes, dst=0) #Sum
      for key in metrics:
        torch.distributed.reduce(metrics[key], dst=0) #Sum
        if dist.get_rank() == 0:
          metrics[key] = (metrics[key]/num_images).item()
      if compare_with_pseudo:
        for key in pseudo_loss:
          torch.distributed.reduce(pseudo_loss[key], dst=0)
          if dist.get_rank() == 0:
            metrics[f'pseudo_loss_{key}'] = (pseudo_loss[key]/num_scenes).item()
          
      if compare_with_input:
        metrics_input = metric_computer_input.sum()
        for key in metrics_input:
          torch.distributed.reduce(metrics_input[key], dst=0)
          if dist.get_rank() == 0:
            metrics_input[key] = (metrics_input[key]/num_images).item()
      else:
        metrics_input = {}
    
      return metrics, metrics_input
    # except:
    #   print('Error')
    #   return {},{}






@gin.configurable
def training(
    model, optimizer_, scheduler_, train_loader, 
    output_dir,
    total_steps: gin.REQUIRED,
    pretrain_steps: gin.REQUIRED,
    eval_interval: gin.REQUIRED,
    log_interval: gin.REQUIRED,
    save_interval: gin.REQUIRED,
    log_image_interval: gin.REQUIRED,
    grad_clip_norm: gin.REQUIRED,
    image_l1_loss_weight=1.0,
    lpips_loss_weight=0,
    resume_from_step=0,
    enable_amp=False,
    empty_cache_fre=-1
):   

    if dist.get_rank() == 0:
      logger = ProcessSafeLogger(os.path.join(output_dir, 'train.log')).get_logger()
    if enable_amp:
      scaler = torch.cuda.amp.GradScaler()
      torch.autograd.set_detect_anomaly(False)
    else:
      torch.autograd.set_detect_anomaly(False)

    if lpips_loss_weight > 0:
      lpips_loss_func = loss_utils.lpips_loss_fn()
    train_iterator = iter(train_loader)
    accumulate_step = gin.query_parameter('build_trainloader.accumulate_step')
    if dist.get_rank() == 0:
      logger.info(f'Accumulate step: {accumulate_step}')
    for step in tqdm(range(resume_from_step*accumulate_step, total_steps*accumulate_step), disable=dist.get_rank()!=0):

      step_consider_accum = step//accumulate_step
      
      try:
        batch = next(train_iterator)
      except StopIteration:
        batch = next(train_iterator)
      #print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
      #print(batch[0]['gs_params'])
      if step_consider_accum == 0:
        
        print('----------------------Num train------------------')
        print(batch[0]['gs_params']['means'].shape[0])

      scene_gs = batch[0]['gs_params'].copy()
      scene_gs = gpu_utils.move_to_device(scene_gs,model.device)

      #------------------------------------------------
      
      if batch[0]['indices']!=None:
        batch_gs_split={}
        for data in batch:
          mask_indices = data['indices']
          for key in scene_gs.keys():
            tensor = scene_gs[key].clone()
            batch_gs_split[key] = tensor[mask_indices]
        #print(batch_gs_split['means'].shape)
        batch_gs_split = [batch_gs_split]
      else:
        batch_gs_split = [data['gs_params'] for data in batch]
        #print(batch_gs_split[0['gs_params']['means'].shape])
        #del scene_gs
      
      #------------------------------------------------
      #batch_gs_split = split_dict(batch, process_type='in', scene_gs=scene_gs)
      
      batch_gs = gpu_utils.move_to_device(batch_gs_split,model.device)
      batch_cameras = gpu_utils.move_to_device([data['cameras'] for data in batch],model.device)
      batch_images = gpu_utils.move_to_device([data['images'] for data in batch],model.device)
      batch_scene_idx = [data['scene_idx'] for data in batch]
      
      #print([data['images_name'] for data in batch])
      forward_kwargs = {'batch_normalized_gs': batch_gs, 'batch_scene_idx': step}

      with torch.cuda.amp.autocast(enabled=enable_amp):
          out_batch_gs = model(**forward_kwargs)
  
      #out_batch_gs = split_dict(batch, process_type='in', out_batch_gs=out_batch_gs, scene_gs=scene_gs)
      #-------------------------------------------------------------------
      if batch[0]['indices']!=None:
        out_batch_gs_split={}
        for data in batch:
          mask_indices = data['indices']
          for key in scene_gs.keys():
            tensor = scene_gs[key].clone()
            tensor[mask_indices] = out_batch_gs[0][key]
            out_batch_gs_split[key] = tensor
        out_batch_gs = [out_batch_gs_split]
      #-------------------------------------------------------------------

      loss_dict = {}
      metric_dict = {}
      if step_consider_accum < pretrain_steps:
        loss = 0
        for ii, (out_gs, in_gs) in enumerate(zip(out_batch_gs, batch_gs)):
          #print('44444444444444444444444444444444444444444444444444444')
          with torch.no_grad():
            pseudo_target = gs_utils.create_pseudo_target(
              sh_degree=model.module.sh_degree, 
              N=in_gs['means'].shape[0],
              input_gs=in_gs,)

          for key in pseudo_target:
            target = pseudo_target[key].to(model.device)
            pred = out_gs[key]
            value = (pred - target).abs().mean()
            if key == 'features_rest':
              if model.module.sh_degree>0:
                loss += value
            else:
              loss += value
            metric_dict['pretrain/'+key] = value
        loss = loss/len(out_batch_gs)
        loss_dict['pretrain_loss'] = loss/len(out_batch_gs)
        optimizer, scheduler = optimizer_['pretrain'], scheduler_['pretrain']
      else:
          #print('4444444444444444444444444444444444444444444444444444412312312')
          metric_dict['train_psnr'] = 0
          loss_dict['image_l1'] = 0
          

          if lpips_loss_weight > 0 and step > APPLY_LPIPS and step < REMOVE_LPIPS:
            loss_dict['lpips'] = 0

          num_images = 0

          ssim = SSIM(data_range=1.0, size_average=True, channel=3)

          for out_gs, cameras, images, in_gs in zip(out_batch_gs, batch_cameras, batch_images, batch_gs):

            pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(out_gs, cameras, images) #a List
            
            

            for pred_img, gt_img in zip(pred_imgs, images):
                #loss_dict['image_l1'] +=  (pred_img - gt_img).abs().mean()#/len(pred_imgs)
                #simloss = 1 - ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])

                # if pred_img.shape[0] < gt_img.shape[0]:
                #   gt_img = gt_img[:pred_img.shape[0], :, :]
                # elif pred_img.shape[0] > gt_img.shape[0]:
                #   pred_img = pred_img[:gt_img.shape[0], :, :]

                # if pred_img.shape[1] < gt_img.shape[1]:
                #   gt_img = gt_img[:,:pred_img.shape[1], :]
                # elif pred_img.shape[1] > gt_img.shape[1]:
                #   pred_img = pred_img[:,:gt_img.shape[1], :]


                # pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_LINEAR)

                loss_dict['image_l1'] += (pred_img - gt_img).abs().mean()
                #loss_dict['image_ssim'] += 0.1 * simloss

                if lpips_loss_weight > 0 and step > APPLY_LPIPS and step < REMOVE_LPIPS:
                  loss_dict['lpips'] += lpips_loss_func(pred_img.unsqueeze(0), gt_img.unsqueeze(0)).mean()
                
                metric_dict['train_psnr'] += (psnr(pred_img.unsqueeze(0), gt_img.unsqueeze(0)).mean())#/len(pred_imgs)

                

                num_images += 1
            #print(num_images)
          
          
          loss_dict['image_l1'] = loss_dict['image_l1']/num_images/len(out_batch_gs)*image_l1_loss_weight
          
          if lpips_loss_weight > 0 and step > APPLY_LPIPS and step < REMOVE_LPIPS:
            loss_dict['lpips'] = loss_dict['lpips']/num_images/len(out_batch_gs)*lpips_loss_weight

          metric_dict['train_psnr'] = metric_dict['train_psnr']/num_images/len(out_batch_gs)

          optimizer, scheduler = optimizer_['train2D'], scheduler_['train2D']
      #print(loss_dict)
      total_loss = sum(loss_dict.values())/accumulate_step
      #print('--------------------------------------')
      #print(total_loss)
      

      if enable_amp:
        scaler.scale(total_loss).backward()
      else:
        total_loss.backward()
      if (step+1) % accumulate_step == 0:
        if grad_clip_norm > 0:
          if enable_amp:
            scaler.unscale_(optimizer)
          torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        if enable_amp:
          scaler.step(optimizer)
          scaler.update()
        else:
          optimizer.step()
        optimizer.zero_grad()
        scheduler.step()


      # pred_imgs_vis = pred_imgs.copy()
      # pred_imgs_vis = [(im*255).detach().cpu().numpy().astype(np.uint8) for im in pred_imgs_vis]
      # grid = make_grid(pred_imgs_vis, nrows=1, ncols=1)
      # grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
      # cv2.imwrite(os.path.join(output_dir, f'train_vis/{step_consider_accum:08d}_pred-rank{dist.get_rank()}.png'), grid)

      # gt_imgs_vis = images.copy()
      # gt_imgs_vis = [(im*255).cpu().numpy().astype(np.uint8) for im in gt_imgs_vis]
      # grid = make_grid(gt_imgs_vis, nrows=1, ncols=1)
      # grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
      # cv2.imwrite(os.path.join(output_dir, f'train_vis/{step_consider_accum:08d}_gt-rank{dist.get_rank()}.png'), grid)


      if empty_cache_fre > 0 and (step+1) % empty_cache_fre == 0:
        torch.cuda.empty_cache()

      if (step_consider_accum % log_interval == 0) and step%accumulate_step==0:
        for key, value in list(loss_dict.items()) + list(metric_dict.items()):
            torch.distributed.reduce(value, dst=0)
            if dist.get_rank() == 0:
              value = (value/torch.cuda.device_count()).item()
              wandb.log({key: value}, step=step_consider_accum)
              #if step_consider_accum % (log_interval*10)==0:
              if step_consider_accum % (log_interval)==0:
                logger.info(f'Training-Step {step_consider_accum}: {key}: {value:.3f}')
              wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=step_consider_accum)

      if step_consider_accum % log_image_interval == 0 and step%accumulate_step==0:
        os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
        with torch.no_grad():
          imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(
            gpu_utils.move_to_device(out_batch_gs[0], device=model.device), batch_cameras[0], batch_images[0])
          imgs = [(im*255).cpu().numpy().astype(np.uint8) for im in imgs]
        grid = make_grid(imgs, nrows=1, ncols=1)
        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f'train/{step_consider_accum:08d}_pred-rank{dist.get_rank()}.png'), grid)

      #----------------------------------------------
      #if ((step%accumulate_step==0) and ((step_consider_accum % eval_interval == 0) or (step_consider_accum+1)==pretrain_steps)):
      if (step%accumulate_step==0) and ((step_consider_accum) % save_interval == 0 or (step_consider_accum)==total_steps-1): 
        
        model.eval()
        for test_dataset, test_loader in build_testloader().items():
            metrics, metrics_input = evaluation(model, test_loader = test_loader,
                                output_dir=output_dir+f'/eval/{test_dataset}/{step_consider_accum}', output_gt=(step_consider_accum==0), compare_with_pseudo=step_consider_accum<pretrain_steps,
                                evaluate_input=(step_consider_accum==0)) #when step==0, we evaluate the input
            if dist.get_rank() == 0:
                wandb.log({f'metrics_testscenes/{test_dataset}/{k}_testviews':v for k,v in metrics.items()}, step=step_consider_accum)
                metric_str = ' '.join([f'{k}: {v:.4f}' for k,v in metrics.items()])
                logger.info(f'Test {test_dataset} Step {step_consider_accum}: {metric_str}')
            dist.barrier()
      # --------------------------------------------------
      #if (step%accumulate_step==0) and ((step_consider_accum+1) % save_interval == 0 or (step_consider_accum+1)==pretrain_steps): 
      #if (step%accumulate_step==0) and ((step_consider_accum) % save_interval == 0 or (step_consider_accum)==pretrain_steps): 
      if (step%accumulate_step==0) and ((step_consider_accum) % save_interval == 0 or (step_consider_accum)==total_steps-1): 
        if dist.get_rank()==0:
            os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(output_dir, f'checkpoints/model_{step_consider_accum:08d}.pth'))
            logger.info(f'Save model at step {step_consider_accum}')
        dist.barrier()
      model.train()
        
      if step==resume_from_step and dist.get_rank() == 0:
        with open(os.path.join(FLAGS.output_dir, 'config.gin'),'w') as f:
            f.writelines(gin.operative_config_str())
      
    return step


def main(argv):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()
    gin.bind_parameter('training.output_dir', FLAGS.output_dir)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    set_seed()


    # 1. Dataloading
    train_loader = build_trainloader()
    # 2. Build Model
    model = FeaturePredictor()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_params}')

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if model.resume_ckpt is not None:
      model.load_state_dict(torch.load(model.resume_ckpt,map_location='cpu'))
      print(f'Load model from {model.resume_ckpt}')

    if FLAGS.only_eval:
      assert model.resume_ckpt is not None, 'Need to specify the model checkpoint for evaluation'

    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    
    if FLAGS.only_eval == False:
      model.train()
      # 3. Optimizer
      optimizer, scheduler = {}, {}
      with gin.config_scope('pretrain'):
        optimizer['pretrain'] = build_optimizer(model.module)
        scheduler['pretrain'] = build_scheduler(optimizer['pretrain'])
      with gin.config_scope('train2D'):
        optimizer['train2D'] = build_optimizer(model.module)
        scheduler['train2D'] = build_scheduler(optimizer['train2D'])

      if rank==0:
        wandb_run = wandb.init(project='3dgs_multiple-scenes', dir=FLAGS.wandb_dir) #resume=?
        if FLAGS.output_dir[-1]=='/':
          FLAGS.output_dir = FLAGS.output_dir[:-1]
        wandb.run.name = '/'.join(FLAGS.output_dir.split('/')[-2:])
      final_step = training(model, optimizer, scheduler, train_loader, output_dir=FLAGS.output_dir, resume_from_step=0)

    model.eval()
    for test_dataset, test_loader in build_testloader().items():
        metrics, metrics_input = evaluation(model, test_loader = test_loader, 
                            output_dir=FLAGS.output_dir+f'/{FLAGS.eval_subdir}/{test_dataset}', 
                            compare_with_input=FLAGS.compare_with_input,
                            save_as_single=True,
                            output_gt=True, compare_with_pseudo=False,output_pred=True)
        if dist.get_rank() == 0:
            logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, FLAGS.eval_subdir, 'eval.log')).get_logger()
            metric_str = ' '.join([f'{k}: {v:.4f}' for k,v in metrics.items()])
            logger.info(f'Test-{test_dataset}: {metric_str}')
            if FLAGS.compare_with_input:
              metric_str = ' '.join([f'{k}: {v:.4f}' for k,v in metrics_input.items()])
              logger.info(f'Input 3DGS: Test-{test_dataset}: {metric_str}')
        dist.barrier()
    
    dist.destroy_process_group()

if __name__=="__main__":
  #torch.multiprocessing.set_start_method('spawn')
  app.run(main)
