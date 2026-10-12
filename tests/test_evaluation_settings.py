"""Checks for checkpoint inputs and single-frame perturbation files."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from calibration_objective import IMPLEMENTATION_VERSION
from otf_preprocessing import PREPROCESSING_VERSION
from dataset import KITTI_perturb
from test_otf import options, validate_evaluation_settings


class EvaluationSettingsTests(unittest.TestCase):
    def test_preprocessing_consistency(self):
        args = options([])
        cfg = {'dataset': dict(cam_id=3, voxel_size=.1, pooling=5,
                               extend_ratio=[2.5, 2.5], test=[20, 21])}
        checkpoint = dict(implementation_version=IMPLEMENTATION_VERSION,
            args=dict(preprocessing_version=PREPROCESSING_VERSION,
                      rgb_encoding=args.rgb_encoding, canny_low=args.canny_low,
                      canny_high=args.canny_high), config=copy.deepcopy(cfg))
        # Test disturbance settings may differ from training.
        args.max_deg = 1
        validate_evaluation_settings(checkpoint, args, cfg)
        cfg['dataset']['pooling'] = 3
        with self.assertRaisesRegex(ValueError, 'pooling'):
            validate_evaluation_settings(checkpoint, args, cfg)
        cfg['dataset']['pooling'] = 5
        args.canny_low = 50
        with self.assertRaisesRegex(ValueError, 'canny_low'):
            validate_evaluation_settings(checkpoint, args, cfg)

    def test_single_frame_perturbations(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'perturb.csv'
            np.savetxt(path, np.zeros((1, 6)), delimiter=',')
            dataset = KITTI_perturb([None], 10, .25, file=str(path))
            self.assertEqual(tuple(dataset.perturb.shape), (1, 1, 6))
            np.savetxt(path, np.full((1, 6), np.nan), delimiter=',')
            with self.assertRaisesRegex(ValueError, 'finite'):
                KITTI_perturb([None], 10, .25, file=str(path))


if __name__ == '__main__':
    unittest.main()
