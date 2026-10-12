"""Single-frame accuracy and synchronized latency evaluation."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from dataset import BaseKITTIDataset, KITTI_perturb
from OTF_CalibNet import OTF_CalibNet
from calibration_objective import correction_matrix, IMPLEMENTATION_VERSION
from depth_projection import project_depth
from scipy.spatial.transform import Rotation
import utils
from otf_preprocessing import add_preprocessing_options, prepare_inputs, PREPROCESSING_VERSION


def options(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yml')
    p.add_argument('--dataset_path', default='../data')
    p.add_argument('--pretrained', default='./checkpoint/single_frame/best.pth')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch_size', type=int, choices=[1], default=1)
    p.add_argument('--inner_iter', type=int, choices=[1], default=1, help='Single forward pass for the paper protocol')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--skip_frame', type=int, default=1)
    p.add_argument('--pcd_sample', type=int, default=-1)
    p.add_argument('--max_deg', type=float, default=10)
    p.add_argument('--max_tran', type=float, default=0.25)
    p.add_argument('--mag_randomly', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--max_frames', type=int, default=0, help='0: all frames; >0: explicit limit')
    p.add_argument('--res_dir', default='res/single_frame')
    p.add_argument('--perturb_file', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--allow_count_mismatch', action='store_true', help='Diagnostic frame subset')
    add_preprocessing_options(p)
    return p.parse_args(argv)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def timed_call(fn, device):
    synchronize(device)  # drain previous GPU work BEFORE starting timer
    t0 = time.perf_counter()
    result = fn()
    synchronize(device)  # wait for all submitted work BEFORE stopping timer
    return result, (time.perf_counter() - t0) * 1000.0


def statistics(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        raise ValueError('No timed frames')
    return dict(count=int(a.size), mean_ms=float(a.mean()), std_ms=float(a.std()),
                median_ms=float(np.median(a)), p95_ms=float(np.percentile(a, 95)),
                min_ms=float(a.min()), max_ms=float(a.max()))


def pipeline(batch, model, device, args, cfg):
    # Start from prepared CPU inputs and project depth within the timed interval.
    rgb = batch['img'].to(device=device, dtype=torch.float32)
    pcd = batch['uncalibed_pcd'].to(device=device, dtype=torch.float32)
    intrinsic = batch['InTran'][0].to(device=device, dtype=torch.float32)
    depth = project_depth(pcd, intrinsic, rgb.shape[-2:], cfg['dataset']['pooling'])
    rgb_input, depth_input = prepare_inputs(rgb, depth, args)
    rotation, translation = model(rgb_input, depth_input)
    correction = torch.cat([rotation, translation], dim=1)
    # The device-resident six-vector is the endpoint; metrics and output I/O are excluded.
    return correction, rgb_input, depth_input, depth


@torch.inference_mode()
def evaluate(args, cfg, checkpoint, loader):
    device = torch.device(args.device)
    model = OTF_CalibNet(backbone_pretrained=False).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    records, errors = [], []
    first_shape = None
    for index, batch in enumerate(loader):
        if args.max_frames and index >= args.max_frames:
            break
        if index == 0:
            # Warm up the complete pipeline outside the timed intervals.
            for _ in range(args.warmup):
                pipeline(batch, model, device, args, cfg)
            synchronize(device)

        result, pipeline_ms = timed_call(lambda: pipeline(batch, model, device, args, cfg), device)
        correction, rgb_input, depth_input, projected_depth = result

        if index == 0:
            first_shape = list(batch['img'].shape)
            cached = batch['uncalibed_depth_img'].to(device)
            if not torch.allclose(cached, projected_depth, rtol=1e-4, atol=1e-5):
                raise ValueError('Dataset and runtime depth projection differ')

        # Replay the network separately using device-resident inputs, including concatenation.
        def network_only():
            rot, tsl = model(rgb_input, depth_input)
            return torch.cat([rot, tsl], dim=1)

        _, network_ms = timed_call(network_only, device)
        records.append([index, network_ms, pipeline_ms])

        transform = correction_matrix(correction[:, :3], correction[:, 3:])
        target = torch.linalg.inv(batch['igt'].to(device))
        pred_np, target_np = transform[0].cpu().numpy(), target[0].cpu().numpy()
        pred_angles = Rotation.from_matrix(pred_np[:3,:3]).as_euler('xyz')
        target_angles = Rotation.from_matrix(target_np[:3,:3]).as_euler('xyz')
        delta_angles = pred_angles-target_angles
        pred_center = -pred_np[:3,:3].T @ pred_np[:3,3]
        target_center = -target_np[:3,:3].T @ target_np[:3,3]
        errors.append(np.abs(np.concatenate((delta_angles,pred_center-target_center))))

        if (index + 1) % 50 == 0:
            print(f'Completed {index + 1} frames')

    if not records:
        raise ValueError('Empty evaluation dataset')

    arr = np.asarray(records)
    err = np.asarray(errors)
    root = Path(args.res_dir)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / 'errors.npy', err)
    np.savetxt(root / 'latency.csv', arr, delimiter=',',
               header='frame,network_ms,pipeline_e2e_ms', comments='')
    report = dict(
        network=statistics(arr[:, 1]),
        pipeline_e2e=statistics(arr[:, 2]),
        rotation_mae_deg_xyz=np.degrees(err[:, :3]).mean(axis=0).tolist(),
        translation_mae_m_xyz=err[:, 3:].mean(axis=0).tolist(),
        ARE_deg=float(np.degrees(err[:,:3]).mean()),
        ATE_cm=float(100*err[:,3:].mean()),
        rotation_l2_mean_rad=float(np.linalg.norm(err[:,:3],axis=1).mean()),
        translation_l2_mean_m=float(np.linalg.norm(err[:,3:],axis=1).mean()),
        metadata=dict(
            device=str(device),
            gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            opencv_version=__import__('cv2').__version__,
            precision='FP32, no autocast',
            warmup=args.warmup,
            input_shape=first_shape,
            inner_iter=1,
            batch_size=1,
            preprocessing_version=PREPROCESSING_VERSION,
            implementation_version=IMPLEMENTATION_VERSION,
            sequences=cfg['dataset']['test'],
            arguments=vars(args),
            dataset_config=cfg['dataset'],
            resize_ratio=checkpoint['args']['resize_ratio'],
            network_scope='Prepared device-resident Canny/Laplacian tensors to device-resident six-vector; includes concatenation; synchronized wall time.',
            pipeline_scope='Dataset-prepared CPU RGB/perturbed point cloud to device-resident six-vector; includes transfers, initial depth reprojection/pooling, CPU OpenCV Canny and transfers, Laplacian normalization and model.',
            exclusions='Dataset disk IO, decoding, resizing, sampling, synthetic perturbation generation, sensor acquisition/synchronization, output transfer to CPU, correction-matrix conversion, evaluation and visualization. Not raw-sensor-to-result latency.',
            timing='perf_counter with CUDA synchronization before/after each interval; network replay measured separately from pipeline; CPU has no CUDA sync.')
    )
    (root / 'latency_summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['network', 'pipeline_e2e']}, indent=2))
    print(f'Results saved to {root}')


def validate_evaluation_settings(checkpoint, args, cfg):
    """Keep input construction consistent while permitting new test perturbations."""
    if checkpoint.get('implementation_version') != IMPLEMENTATION_VERSION:
        raise ValueError('Checkpoint architecture/objective version mismatch')
    saved = checkpoint['args']
    if saved.get('preprocessing_version') != PREPROCESSING_VERSION:
        raise ValueError('Checkpoint preprocessing version mismatch')
    for field in ('rgb_encoding', 'canny_low', 'canny_high'):
        if saved.get(field) != getattr(args, field):
            raise ValueError(f'{field} must match training checkpoint: {saved.get(field)!r}')
    trained = checkpoint['config']['dataset']
    for field in ('cam_id', 'voxel_size', 'pooling', 'extend_ratio'):
        if cfg['dataset'].get(field) != trained.get(field):
            raise ValueError(f'Dataset preprocessing must match training: {field}')


def main():
    args = options()

    if args.warmup < 0 or args.max_frames < 0:
        raise ValueError('warmup and max_frames must be nonnegative')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; pass --device cpu for a CPU-only diagnostic, not GPU timing')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    saved = checkpoint['args']

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    validate_evaluation_settings(checkpoint, args, cfg)

    split = [str(s).zfill(2) for s in cfg['dataset']['test']]
    if not split:
        raise ValueError('No test sequences in config')

    # Use the saved image scale and validated dataset preprocessing settings.
    base = BaseKITTIDataset(
        args.dataset_path, 1, split, cfg['dataset']['cam_id'],
        frame_manifest=cfg['dataset'].get('frame_manifest'), skip_frame=args.skip_frame,
        voxel_size=cfg['dataset']['voxel_size'],
        pcd_sample_num=args.pcd_sample,
        resize_ratio=saved['resize_ratio'],
        extend_ratio=cfg['dataset']['extend_ratio'])
    expected = cfg['dataset']['expected_pairs']['test']
    if len(base) != expected and not args.allow_count_mismatch:
        raise ValueError(f'Loaded {len(base)} pairs, expected {expected}; supply the exact frame manifest')
    if not len(base):
        raise ValueError('Empty dataset')

    Path(args.res_dir).mkdir(parents=True, exist_ok=True)

    perturb_path = Path(args.perturb_file) if args.perturb_file else \
        Path(args.res_dir) / ('perturb_' + '-'.join(split)
                              + f'_{args.max_deg:g}deg_{args.max_tran:g}m_seed{args.seed}_n{len(base)}.csv')

    if not perturb_path.exists():
        perturb_path.parent.mkdir(parents=True, exist_ok=True)
        generator = utils.transform.UniformTransformSE3(args.max_deg, args.max_tran, args.mag_randomly)
        perturbations = np.stack(
            [generator.generate_transform().cpu().numpy().reshape(6) for _ in range(len(base))])
        np.savetxt(perturb_path, perturbations, delimiter=',')
    else:
        existing = np.loadtxt(perturb_path, delimiter=',', ndmin=2)
        if existing.shape != (len(base), 6):
            raise ValueError('Existing perturbation file has wrong shape; '
                             'provide a matching file rather than overwrite it')

    dataset = KITTI_perturb(base, args.max_deg, args.max_tran, args.mag_randomly,
                            pooling_size=cfg['dataset']['pooling'],
                            file=str(perturb_path))
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=args.pin_memory)
    evaluate(args, cfg, checkpoint, loader)


if __name__ == '__main__':
    main()
