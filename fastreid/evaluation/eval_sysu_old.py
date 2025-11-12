import os
import logging
import torch
import numpy as np
from sklearn.preprocessing import normalize
from .rerank import re_ranking
from torch.nn import functional as F


def pairwise_distance(query_features, gallery_features):
    x = query_features
    y = gallery_features
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def get_gallery_names(perm, cams, ids, trial_id, num_shots=1):
    names = []
    for cam in cams:
        cam_perm = perm[cam - 1][0].squeeze()
        for i in ids:
            instance_id = cam_perm[i - 1][trial_id][:num_shots]
            names.extend(['SYSU_MM01_SCT_new/gallery/{:0>4d}_c{}_{:0>4d}'.format(i, cam, ins) for ins in instance_id.tolist()])
    return names


def get_unique(array):
    _, idx = np.unique(array, return_index=True)
    return array[np.sort(idx)]


def get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    gallery_unique_count = get_unique(gallery_ids).shape[0]
    match_counter = np.zeros((gallery_unique_count,))

    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]
        result_i[np.equal(cam_locations_result[probe_index], query_cam_ids[probe_index])] = -1

        # remove the -1 entries from the label result
        result_i = np.array([i for i in result_i if i != -1])

        # remove duplicated id in "stable" manner
        result_i_unique = get_unique(result_i)

        # match for probe i
        match_i = np.equal(result_i_unique, query_ids[probe_index])

        if np.sum(match_i) != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            match_counter += match_i

    rank = match_counter / valid_probe_sample_count
    cmc = np.cumsum(rank)
    return cmc


def get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0
    avg_precision_sum = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]
        result_i[cam_locations_result[probe_index, :] == query_cam_ids[probe_index]] = -1

        # remove the -1 entries from the label result
        result_i = np.array([i for i in result_i if i != -1])

        # match for probe i
        match_i = result_i == query_ids[probe_index]
        true_match_count = np.sum(match_i)

        if true_match_count != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]

            ap = np.mean(np.arange(1, true_match_count + 1) / (true_match_rank + 1))
            avg_precision_sum += ap

    mAP = avg_precision_sum / valid_probe_sample_count
    return mAP


def eval_sysu(query_feats, q_feats_mask, query_ids, query_cam_ids, gallery_feats, g_feats_mask, gallery_ids, gallery_cam_ids, gallery_img_paths,
              perm, mode='all', num_shots=1, num_trials=10, rerank=False, logger=None):
    assert mode in ['indoor', 'all']

    gallery_cams = [1, 2] if mode == 'indoor' else [1, 2, 4, 5]

    # cam2 and cam3 are in the same location
    query_cam_ids[np.equal(query_cam_ids, 3)] = 2
    query_feats = F.normalize(query_feats, dim=1)
    query_feats_mask = F.normalize(q_feats_mask, dim=1)

    gallery_indices = np.in1d(gallery_cam_ids, gallery_cams)

    gallery_feats = gallery_feats[gallery_indices]
    gallery_feats = F.normalize(gallery_feats, dim=1)
    gallery_feats_mask = g_feats_mask[gallery_indices]
    gallery_feats_mask = F.normalize(gallery_feats_mask, dim=1)

    gallery_cam_ids = gallery_cam_ids[gallery_indices]
    gallery_ids = gallery_ids[gallery_indices]
    gallery_img_paths = gallery_img_paths[gallery_indices]
    gallery_names = np.array(['/'.join(os.path.splitext(path)[0].split('/')[-3:]) for path in gallery_img_paths])

    gallery_id_set = np.unique(gallery_ids)

    mAP, r1, r5, r10, r20 = 0, 0, 0, 0, 0
    #mAP_m, r1_m, r5_m, r10_m, r20_m = 0, 0, 0, 0, 0
    #mAP_e, r1_e, r5_e, r10_e, r20_e = 0, 0, 0, 0, 0
    for t in range(num_trials):
        names = get_gallery_names(perm, gallery_cams, gallery_id_set, t, num_shots)
        flag = np.in1d(gallery_names, names)

        g_feat = gallery_feats[flag]
        g_feat_mask = gallery_feats_mask[flag]
        g_ids = gallery_ids[flag]
        g_cam_ids = gallery_cam_ids[flag]

        dist_mat = pairwise_distance(query_feats, g_feat)
        #dist_mat_mask = pairwise_distance(query_feats_mask, g_feat_mask)
        #dist_mat_ensemble = (dist_mat + dist_mat_mask)/2
        # dist_mat = -torch.mm(query_feats, g_feat.permute(1,0))

        sorted_indices = np.argsort(dist_mat, axis=1)
        #sorted_indices_mask = np.argsort(dist_mat_mask, axis=1)
        #sorted_indices_e = np.argsort(dist_mat_ensemble, axis=1)

        mAP += get_mAP(sorted_indices, query_ids, query_cam_ids, g_ids, g_cam_ids)
        cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, g_ids, g_cam_ids)

        #mAP_m += get_mAP(sorted_indices_mask, query_ids, query_cam_ids, g_ids, g_cam_ids)
        #cmc_m  = get_cmc(sorted_indices_mask, query_ids, query_cam_ids, g_ids, g_cam_ids)

        #mAP_e += get_mAP(sorted_indices_e, query_ids, query_cam_ids, g_ids, g_cam_ids)
        #cmc_e = get_cmc(sorted_indices_e, query_ids, query_cam_ids, g_ids, g_cam_ids)

        r1 += cmc[0]
        r5 += cmc[4]
        r10 += cmc[9]
        r20 += cmc[19]

        '''
        r1_m += cmc_m[0]
        r5_m += cmc_m[4]
        r10_m += cmc_m[9]
        r20_m += cmc_m[19]

        r1_e += cmc_e[0]
        r5_e += cmc_e[4]
        r10_e += cmc_e[9]
        r20_e += cmc_e[19]
        '''

    '''
    r1, r1_m, r1_e = r1 / num_trials * 100, r1_m / num_trials * 100, r1_e / num_trials * 100
    r5, r5_m, r5_e = r5 / num_trials * 100, r5_m / num_trials * 100, r5_e / num_trials * 100
    r10, r10_m, r10_e = r10 / num_trials * 100, r10_m / num_trials * 100, r10_e / num_trials * 100
    r20, r20_m, r20_e = r20 / num_trials * 100, r20_m / num_trials * 100, r20_e / num_trials * 100
    mAP, mAP_m, mAP_e = mAP / num_trials * 100, mAP_m / num_trials * 100, mAP_e / num_trials * 100
    '''
    r1 = r1 / num_trials * 100
    r5 = r5 / num_trials * 100
    r10 = r10 / num_trials * 100
    r20 = r20 / num_trials * 100
    mAP = mAP / num_trials * 100

    perf = '{}-{} num-shot:{} r1 precision = {:.2f} , r10 precision = {:.2f} , r20 precision = {:.2f}, mAP = {:.2f}'
    #perf_mask = '{}-{} num-shot:{} r1 precision = {:.2f} , r10 precision = {:.2f} , r20 precision = {:.2f}, mAP = {:.2f}'
    #perf_ensemble = '{}-{} num-shot:{} r1 precision = {:.2f} , r10 precision = {:.2f} , r20 precision = {:.2f}, mAP = {:.2f}'
    logger.info(perf.format('RGB+IR',mode, num_shots, r1, r10, r20, mAP))
    #logger.info(perf_mask.format('mask',mode,num_shots, r1_m, r10_m, r20_m, mAP_m))
    #logger.info(perf_ensemble.format('ensemble',mode,num_shots, r1_e, r10_e, r20_e, mAP_e))

    return mAP, r1, r5, r10, r20
