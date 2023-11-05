import os
import math
import cv2
import types
import json
import numpy as np
import torch
from tqdm import tqdm
import torch
import time
from superglue.superglue import SuperGlue


def error_auc(errors, thresholds=[3, 5, 10]):
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def get_bitmap(image_path, new_shape=None):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ori_shape = image.shape

    if new_shape:
        new_shape = np.array(new_shape)
        ori_shape = np.array(ori_shape)
        scale = max(new_shape / ori_shape)

        image = image[:int(new_shape[0] / scale), 
                      :int(new_shape[1] / scale)]
        image = cv2.resize(
            image, (new_shape[1], new_shape[0]),
            interpolation=cv2.INTER_AREA
        )
    # image = torch.from_numpy(image/255.).float()[None, None]

    return image, ori_shape


def adapt_homography_to_processing(H, new_shape, ori_shape0, ori_shape1):
    new_shape = np.array(new_shape)
    ori_shape0 = np.array(ori_shape0)
    ori_shape1 = np.array(ori_shape1)

    scale0 = max(new_shape / ori_shape0)
    up_scale = np.diag(np.array([1. / scale0, 1. / scale0, 1.]))

    scale1 = max(new_shape / ori_shape1)
    down_scale = np.diag(np.array([scale1, scale1, 1.]))

    H = down_scale @ H @ up_scale
    return H


def generate_sift(bitmap, extractor):
    gray = cv2.cvtColor(bitmap, cv2.COLOR_RGB2GRAY)
    kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in extractor.detect(gray)])
    return torch.from_numpy(kpts).float()


def generate_superpoint(bitmap, extractor):
    gray = cv2.cvtColor(bitmap, cv2.COLOR_RGB2GRAY)
    gray = torch.from_numpy(gray / 255.)[None].float()

    preds = extractor({'image': gray[None].to('cuda')})
 
    return {k: torch.stack(v) for k, v in preds.items()}


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def main(cfg):
    assert os.path.exists(cfg.path_data)
    scenes = sorted(os.listdir(cfg.path_data))
    
    print('Running inference on device \"{}\"'.format(cfg.device))
    start_time = time.time()
    # ORB
    orb = cv2.ORB_create()
    
    # Load Model and Weights (IF EXISTED)
    config = {
        'superglue': {
            'weights': cfg.superglue_weightname,
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
        }
    }
    matcher = SuperGlue(config['superglue']).eval().to(cfg.device)

    i_results = []
    v_results = []
    mean_of_inliers = 0
    for scene in tqdm(scenes[::-1], total=len(scenes)):
        scene_cls = scene.split('_')[0]
        path_scene = os.path.join(cfg.path_data, scene)

        path0 = os.path.join(path_scene, '1.ppm')
        im0, ori_shape0 = get_bitmap(path0, cfg.resize)

        # ORB feature1
        keypoints0 = orb.detect(im0)
        keypoints0, descriptors0 = orb.compute(im0, keypoints0, )
        # orb.detectAndCompute(im0)
        print()
        

        shape = im0.shape[:2]
        corners = np.array([
            [0,            0,            1],
            [shape[1] - 1, 0,            1],
            [0,            shape[0] - 1, 1],
            [shape[1] - 1, shape[0] - 1, 1]
        ])

        sum_of_inliers = 0
        for idx in range(2, 7):
            path1 = os.path.join(path_scene, f'{idx}.ppm')
            im1, ori_shape1 = get_bitmap(path1, cfg.resize)
            im1 = im1.to(cfg.device)
            
            inp_data_pair = {'image0': im0, 'image1': im1}
            pred = matcher(inp_data_pair)
            pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            
            # Keep the matching keypoints.
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]
            
            if len(mkpts0) < 4:
                # print('Match Failed, return error=999')
                failed_error = 999
                if scene_cls == 'i':
                    i_results.append(failed_error)
                else:
                    v_results.append(failed_error)
                continue

            H_pred, inliers = cv2.findHomography(
                mkpts0, mkpts1, method=cv2.RANSAC
            )
            if H_pred is None:
                # print('Find Homography Failed, return error=999')
                failed_error = 999
                if scene_cls == 'i':
                    i_results.append(failed_error)
                else:
                    v_results.append(failed_error)
                continue

            inliers = inliers.flatten().astype(bool)
            n_inliers = np.sum(inliers)
            sum_of_inliers += float(n_inliers)

            pred_corners = np.dot(corners, np.transpose(H_pred))
            pred_corners = pred_corners[:, :2] / pred_corners[:, 2:]

            H_real = np.loadtxt(os.path.join(path_scene, f'H_1_{idx}'))
            H_real = adapt_homography_to_processing(
                H_real, cfg.resize, ori_shape0, ori_shape1
            )

            # Real corners
            real_corners = np.dot(corners, np.transpose(H_real))
            real_corners = real_corners[:, :2] / real_corners[:, 2:]

            error = np.mean(
                np.linalg.norm(real_corners - pred_corners, axis=1)
            )
            # print(f'[{scene_cls}]: Inliers: {n_inliers} | Error: {error}')

            if scene_cls == 'i':
                i_results.append(error)
            else:
                v_results.append(error)

        mean_of_inliers += sum_of_inliers / 5.

    mean_of_inliers /= float(len(scenes))

    v_results = np.array(v_results).astype(np.float32)
    i_results = np.array(i_results).astype(np.float32)
    results = np.concatenate((i_results, v_results), axis=0)

    # Compute auc
    auc_of_homo_i = error_auc(i_results)
    auc_of_homo_v = error_auc(v_results)
    auc_of_homo = error_auc(results)

    dumps = {
        'inliers': mean_of_inliers,
        **{k: v for k, v in auc_of_homo.items()},
        **{f'i_{k}': v for k, v in auc_of_homo_i.items()},
        **{f'v_{k}': v for k, v in auc_of_homo_v.items()},
    }
    print(f'-- Homography results: \n{dumps}')

    with open(cfg.path_dump, 'w') as json_file:
        json.dump(dumps, json_file)


if __name__ == "__main__":
    cfg = types.SimpleNamespace()
    cfg.path_data = 'D:/WorkSpace/Datasets/Hpatches'
    log_path = 'log_dumps'
    cfg.path_dump = os.path.join(log_path, 'h_evaluate_orb_sg.json')
    cfg.resize = (480, 640)
    cfg.max_features = 2048
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.superglue_weightname = 'outdoor'

    main(cfg)