# from pathlib import Path
# import argparse
# import random
# import numpy as np
# import matplotlib.cm as cm
# import torch
# import argparse
# import time
# from super_match_model import Matching
# from utils import read_image


# parser = argparse.ArgumentParser()
# parser.add_argument('--superglue_weightname', default='outdoor', help='indoor or outdoor')
# parser.add_argument('--save_npz', default=False, help='whether save to disk')
# args = parser.parse_args()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Running inference on device \"{}\"'.format(device))
# config = {
#     'superpoint': {
#         'nms_radius': 4,
#         'keypoint_threshold': 0.005,
#         'max_keypoints': -1,
#     },
#     'superglue': {
#         'weights': args.superglue_weightname,
#         'sinkhorn_iterations': 100,
#         'match_threshold': 0.2,
#     }
# }
# matching = Matching(config).eval().to(device)

# image0, inp0, scales0 = read_image(
#         input_dir / name0, device, opt.resize, rot0, opt.resize_float)
#     image1, inp1, scales1 = read_image(
#         input_dir / name1, device, opt.resize, rot1, opt.resize_float)
#     if image0 is None or image1 is None:
#         print('Problem reading image pair: {} {}'.format(
#             input_dir/name0, input_dir/name1))
# pred = matching({'image0': inp0, 'image1': inp1})
# pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
# kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
# matches, conf = pred['matches0'], pred['matching_scores0']


# # Write the matches to disk.
# out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
#                 'matches': matches, 'match_confidence': conf}