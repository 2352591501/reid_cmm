# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from fastreid.utils import comm
from fastreid.utils.compute_dist import build_dist
from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank import evaluate_rank
from .eval_regdb import eval_regdb
from .eval_sysu import eval_sysu
from .roc import evaluate_roc

logger = logging.getLogger(__name__)


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.features = []
        self.features_mask = []
        self.pids = []
        self.camids = []

    def reset(self):
        self.features = []
        self.features_mask = []
        self.pids = []
        self.camids = []
        self.fns = []

    def process(self, inputs, outputs, outputs_mask):
        self.pids.extend(inputs["targets"])
        self.camids.extend(inputs["camids"])
        self.features.append(outputs.cpu())
        self.fns.extend(inputs['img_paths'])
        self.features_mask.append(outputs_mask.cpu())

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            features_mask = comm.gather(self.features_mask)
            features_mask = sum(features_mask, [])

            pids = comm.gather(self.pids)
            pids = sum(pids, [])

            camids = comm.gather(self.camids)
            camids = sum(camids, [])

            # fmt: off
            if not comm.is_main_process(): return {}
            # fmt: on
        else:
            features = self.features
            features_mask = self.features_mask
            pids = self.pids
            camids = self.camids

        features = torch.cat(features, dim=0)
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(pids[:self._num_query])
        query_camids = np.asarray(camids[:self._num_query])

        features_mask = torch.cat(features_mask, dim=0)
        query_features_mask = features_mask[:self._num_query]

        g_img_path = np.array(self.fns[self._num_query:])
        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(pids[self._num_query:])
        gallery_camids = np.asarray(camids[self._num_query:])

        gallery_features_mask = features_mask[self._num_query:]

        if 'SYSU_MM01' in  self.cfg.DATASETS.TESTS:
        
            import scipy.io as sio
            import os
            q_feats, q_feats_mask, q_ids, q_cams = query_features, query_features_mask, query_pids, query_camids
            g_feats, g_feats_mask, g_ids, g_cams = gallery_features, gallery_features_mask, gallery_pids, gallery_camids
            perm = sio.loadmat(os.path.join('/home/amax/zn/fast-reid-orig/datasets/SYSU-MM01/exp', 'rand_perm_cam.mat'))[
                'rand_perm_cam']
            eval_sysu(q_feats, q_feats_mask, q_ids, q_cams, g_feats, g_feats_mask, g_ids, g_cams,  g_img_path, perm, mode='all', num_shots=1,
                        logger=logger)
            #eval_sysu(q_feats, q_feats_mask, q_ids, q_cams, g_feats, g_feats_mask, g_ids, g_cams,  g_img_path, perm, mode='all', num_shots=10,
                        #logger=logger)
            eval_sysu(q_feats, q_feats_mask, q_ids, q_cams, g_feats, g_feats_mask, g_ids, g_cams,  g_img_path, perm, mode='indoor', num_shots=1,
                        logger=logger)
            #eval_sysu(q_feats, q_feats_mask, q_ids, q_cams, g_feats, g_feats_mask, g_ids, g_cams,  g_img_path, perm, mode='indoor',
                       #   num_shots=10,logger=logger)
        else:
    
            if len(query_features) > 2000:
                logger.info('infrared to visible')
                q_feats, q_feats_mask, q_ids, q_cams = query_features, query_features_mask, query_pids, query_camids
                g_feats, g_feats_mask, g_ids, g_cams = gallery_features, gallery_features_mask, gallery_pids, gallery_camids
                eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, logger=logger)
                logger.info('visible to infrared')
                eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, logger=logger )
            else:
                logger.info('RGB to TI')
                dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, gallery_features)
                cmc, map = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids, use_metric_cuhk03=True)
                self.print_reg(cmc, map)
                dist = self.cal_dist(self.cfg.TEST.METRIC, gallery_features, query_features)
                logger.info('TI to RGB')
                cmc, map = evaluate_rank(dist, gallery_pids, query_pids, gallery_camids, query_camids, use_metric_cuhk03=True)
                self.print_reg(cmc, map)
        
        
        '''
        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)


        dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA

            if self.cfg.TEST.METRIC == "cosine":
                query_features = F.normalize(query_features, dim=1)
                gallery_features = F.normalize(gallery_features, dim=1)

            rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100
        self._results['mAP'] = mAP * 100
        self._results['mINP'] = mINP * 100
        self._results["metric"] = (mAP + cmc[0]) / 2 * 100

        if self.cfg.TEST.ROC_ENABLED:
            scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]
    
        return copy.deepcopy(self._results)
        '''
