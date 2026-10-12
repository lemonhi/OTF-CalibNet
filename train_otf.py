"""Single-frame supervised calibration training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler
from dataset import BaseKITTIDataset, KITTI_perturb
from OTF_CalibNet import OTF_CalibNet
from otf_preprocessing import add_preprocessing_options, prepare_inputs, PREPROCESSING_VERSION
from calibration_objective import CalibrationObjective, correction_matrix, IMPLEMENTATION_VERSION
from depth_projection import project_depth


class SequenceBatchSampler(Sampler):
    """Keep image sizes/intrinsics homogeneous without discarding frames."""
    def __init__(self, lengths, batch_size, shuffle):
        self.lengths, self.batch_size, self.shuffle = lengths, batch_size, shuffle
    def __len__(self):
        return sum((n+self.batch_size-1)//self.batch_size for n in self.lengths)
    def __iter__(self):
        batches, offset = [], 0
        for n in self.lengths:
            ids = torch.randperm(n).tolist() if self.shuffle else list(range(n))
            batches.extend([[offset+i for i in ids[j:j+self.batch_size]]
                            for j in range(0,n,self.batch_size)])
            offset += n
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches)).tolist()]
        yield from batches


def options():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yml')
    p.add_argument('--dataset_path', default='./data')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--epoch', type=int, default=50)
    p.add_argument('--lr0', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--pcd_sample', type=int, default=4096)
    p.add_argument('--skip_frame', type=int, default=1)
    p.add_argument('--resize_ratio', type=float, nargs=2, default=[1.,1.])
    p.add_argument('--max_deg', type=float, default=10.)
    p.add_argument('--max_tran', type=float, default=.25)
    p.add_argument('--alpha', type=float, default=.3)
    p.add_argument('--beta', type=float, default=.3)
    p.add_argument('--gama', type=float, default=.4)
    p.add_argument('--inner_iter', type=int, choices=[1], default=1)
    p.add_argument('--backbone_checkpoint', default=None)
    p.add_argument('--from_scratch', action='store_true', help='Diagnostic training without RGB pretraining')
    p.add_argument('--resume', default=None)
    p.add_argument('--checkpoint_dir', default='checkpoint/single_frame')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--allow_count_mismatch', action='store_true', help='Diagnostic use of a different frame subset')
    add_preprocessing_options(p)
    return p.parse_args()


def make_loader(args, cfg, split, shuffle):
    settings = cfg['dataset']
    base = BaseKITTIDataset(args.dataset_path, args.batch_size,
        [str(s).zfill(2) for s in settings[split]], settings['cam_id'],
        frame_manifest=settings.get('frame_manifest'), skip_frame=args.skip_frame,
        voxel_size=settings['voxel_size'], pcd_sample_num=args.pcd_sample,
        resize_ratio=args.resize_ratio, extend_ratio=settings['extend_ratio'])
    expected = settings['expected_pairs'][split]
    if len(base) != expected and not args.allow_count_mismatch:
        raise ValueError(f'{split}: {len(base)} loaded pairs, expected {expected}. Supply the exact frame manifest; '
                         '--allow_count_mismatch is for diagnostic runs only.')
    dataset = KITTI_perturb(base,args.max_deg,args.max_tran,True,pooling_size=settings['pooling'])
    if not shuffle:
        # Fixed validation perturbations, independent of the training RNG stream.
        with torch.random.fork_rng():
            torch.manual_seed(args.seed+1)
            samples = [dataset.transform.generate_transform().reshape(6) for _ in range(len(base))]
        dataset.file = '<fixed validation perturbations>'
        dataset.perturb = torch.stack(samples)[None]
    sampler = SequenceBatchSampler(base.sep,args.batch_size,shuffle)
    return DataLoader(dataset,batch_sampler=sampler,num_workers=args.num_workers)


def run_epoch(model, loader, objective, optimizer, device, args, cfg):
    model.train(optimizer is not None)
    totals = torch.zeros(4,device=device)
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            rgb, depth = batch['img'].to(device), batch['uncalibed_depth_img'].to(device)
            rotation, translation = model(*prepare_inputs(rgb,depth,args))
            matrix = correction_matrix(rotation,translation)
            cloud = batch['uncalibed_pcd'].to(device)
            corrected = matrix[:,:3,:3] @ cloud + matrix[:,:3,3,None]
            corrected_depth = project_depth(corrected,batch['InTran'].to(device),rgb.shape[-2:],cfg['dataset']['pooling'])
            total, terms = objective(rotation,translation,batch['igt'].to(device),
                corrected_depth,batch['depth_img'].to(device),corrected,batch['pcd'].to(device))
            if not torch.isfinite(total): raise FloatingPointError('Non-finite training objective')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            n = rgb.shape[0]
            totals += torch.stack((total,*terms)).detach()*n
            count += n
    return (totals/count).cpu().tolist()


def main():
    args = options()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if not args.resume and not args.backbone_checkpoint and not args.from_scratch:
        raise ValueError('Provide --backbone_checkpoint for RGB pretraining; --from_scratch is diagnostic only.')
    with open(args.config) as f: cfg=yaml.safe_load(f)
    device=torch.device(args.device)
    model=OTF_CalibNet(backbone_pretrained=bool(args.backbone_checkpoint),
                      backbone_checkpoint=args.backbone_checkpoint).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr0,weight_decay=args.weight_decay)
    start,best=0,float('inf')
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        if saved.get('implementation_version')!=IMPLEMENTATION_VERSION:
            raise ValueError('Checkpoint architecture/objective version mismatch')
        for key in ('rgb_encoding','canny_low','canny_high','alpha','beta','gama','resize_ratio','max_deg','max_tran','pcd_sample'):
            if saved['args'].get(key)!=getattr(args,key): raise ValueError(f'Resume setting differs: {key}')
        if saved['config']!=cfg: raise ValueError('Resume dataset configuration differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        start=saved['epoch']+1;best=saved['best']
    objective=CalibrationObjective((args.alpha,args.beta,args.gama))
    train=make_loader(args,cfg,'train',True);val=make_loader(args,cfg,'val',False)
    root=Path(args.checkpoint_dir);root.mkdir(parents=True,exist_ok=True)
    args.preprocessing_version=PREPROCESSING_VERSION
    print(json.dumps({'parameters':sum(p.numel() for p in model.parameters()),'train_pairs':len(train.dataset),
                      'val_pairs':len(val.dataset),'arguments':vars(args)},indent=2))
    for epoch in range(start,args.epoch):
        tr=run_epoch(model,train,objective,optimizer,device,args,cfg)
        va=run_epoch(model,val,objective,None,device,args,cfg)
        improved=va[0]<best;best=min(best,va[0])
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,
                   args=vars(args),config=cfg,implementation_version=IMPLEMENTATION_VERSION)
        torch.save(state,root/'last.pth')
        if improved:torch.save(state,root/'best.pth')
        with (root/'training.jsonl').open('a') as f:
            f.write(json.dumps({'epoch':epoch,'train_total_grad_cd_direct':tr,'val_total_grad_cd_direct':va})+'\n')
        print(epoch,tr,va)

if __name__=='__main__':main()
