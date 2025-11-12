# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import datetime
import itertools
import logging
import os
import tempfile
import time
from collections import Counter

import torch
from apex.parallel import DistributedDataParallel
from torch import nn

from fastreid.evaluation.testing import flatten_results_dict
from fastreid.solver import optim
from fastreid.utils import comm
from fastreid.utils.checkpoint import PeriodicCheckpointer as _PeriodicCheckpointer
from fastreid.utils.events import EventStorage, EventWriter, get_event_storage
from fastreid.utils.file_io import PathManager
from fastreid.utils.precision_bn import update_bn_stats, get_bn_modules
from fastreid.utils.timer import Timer
from .train_loop import HookBase
from fastreid.utils.serialization import save_checkpoint



__all__ = [
    "CallbackHook",
    "IterationTimer",
    "PeriodicWriter",
    "PeriodicCheckpointer",
    "LRScheduler",
    "AutogradProfiler",
    "EvalHook",
    "PreciseBN",
    "LayerFreeze",
    'ForwardHook',
    'Visu_Hook',
    'Forward_hook',
    'Parsing_hook',
    'Save_Checkpoint'
]

"""
Implement some common hooks.
"""


class CallbackHook(HookBase):
    """
    Create a hook using callback functions provided by the user.
    """

    def __init__(self, *, before_train=None, after_train=None, before_epoch=None, after_epoch=None,
                 before_step=None, after_step=None):
        """
        Each argument is a function that takes one argument: the trainer.
        """
        self._before_train = before_train
        self._before_epoch = before_epoch
        self._before_step = before_step
        self._after_step = after_step
        self._after_epoch = after_epoch
        self._after_train = after_train

    def before_train(self):
        if self._before_train:
            self._before_train(self.trainer)

    def after_train(self):
        if self._after_train:
            self._after_train(self.trainer)
        # The functions may be closures that hold reference to the trainer
        # Therefore, delete them to avoid circular reference.
        del self._before_train, self._after_train
        del self._before_step, self._after_step

    def before_epoch(self):
        if self._before_epoch:
            self._before_epoch(self.trainer)

    def after_epoch(self):
        if self._after_epoch:
            self._after_epoch(self.trainer)

    def before_step(self):
        if self._before_step:
            self._before_step(self.trainer)

    def after_step(self):
        if self._after_step:
            self._after_step(self.trainer)


class IterationTimer(HookBase):
    """
    Track the time spent for each iteration (each run_step call in the trainer).
    Print a summary in the end of training.
    This hook uses the time between the call to its :meth:`before_step`
    and :meth:`after_step` methods.
    Under the convention that :meth:`before_step` of all hooks should only
    take negligible amount of time, the :class:`IterationTimer` hook should be
    placed at the beginning of the list of hooks to obtain accurate timing.
    """

    def __init__(self, warmup_iter=3):
        """
        Args:
            warmup_iter (int): the number of iterations at the beginning to exclude
                from timing.
        """
        self._warmup_iter = warmup_iter
        self._step_timer = Timer()

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer = Timer()
        self._total_timer.pause()

    def after_train(self):
        logger = logging.getLogger(__name__)
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer.iter + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self):
        # +1 because we're in after_step
        iter_done = self.trainer.iter - self.trainer.start_iter + 1
        if iter_done >= self._warmup_iter:
            sec = self._step_timer.seconds()
            self.trainer.storage.put_scalars(time=sec)
        else:
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        self._total_timer.pause()


class PeriodicWriter(HookBase):
    """
    Write events to EventStorage periodically.
    It is executed every ``period`` iterations and after the last iteration.
    """

    def __init__(self, writers, period=20):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
            period (int):
        """
        self._writers = writers
        for w in writers:
            assert isinstance(w, EventWriter), w
        self._period = period

    def after_step(self):
        if (self.trainer.iter + 1) % self._period == 0 or (
                self.trainer.iter == self.trainer.max_iter - 1
        ):
            for writer in self._writers:
                writer.write()

    def after_epoch(self):
        for writer in self._writers:
            writer.write()

    def after_train(self):
        for writer in self._writers:
            writer.close()


class PeriodicCheckpointer(_PeriodicCheckpointer, HookBase):
    """
    Same as :class:`fastreid.utils.checkpoint.PeriodicCheckpointer`, but as a hook.
    Note that when used as a hook,
    it is unable to save additional data other than what's defined
    by the given `checkpointer`.
    It is executed every ``period`` iterations and after the last iteration.
    """

    def before_train(self):
        self.max_epoch = self.trainer.max_epoch
        if len(self.trainer.cfg.DATASETS.TESTS) == 1:
            self.metric_name = "metric"
        else:
            self.metric_name = self.trainer.cfg.DATASETS.TESTS[0] + "/metric"

    def after_epoch(self):
        # No way to use **kwargs
        storage = get_event_storage()
        metric_dict = dict(
            metric=storage.latest()[self.metric_name][0] if self.metric_name in storage.latest() else -1
        )
        self.step(self.trainer.epoch, **metric_dict)


class LRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer, scheduler):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim._LRScheduler)
        """
        self._optimizer = optimizer
        self._scheduler = scheduler

        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    self._best_param_group_id = i
                    break
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    self._best_param_group_id = i
                    break

    def after_step(self):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)

        next_iter = self.trainer.iter + 1
        if next_iter <= self.trainer.warmup_iters:
            self._scheduler["warmup_sched"].step()

    def after_epoch(self):
        next_iter = self.trainer.iter + 1
        next_epoch = self.trainer.epoch + 1
        if next_iter > self.trainer.warmup_iters and next_epoch >= self.trainer.delay_epochs:
            self._scheduler["lr_sched"].step()


class AutogradProfiler(HookBase):
    """
    A hook which runs `torch.autograd.profiler.profile`.
    Examples:
    .. code-block:: python
        hooks.AutogradProfiler(
             lambda trainer: trainer.iter > 10 and trainer.iter < 20, self.cfg.OUTPUT_DIR
        )
    The above example will run the profiler for iteration 10~20 and dump
    results to ``OUTPUT_DIR``. We did not profile the first few iterations
    because they are typically slower than the rest.
    The result files can be loaded in the ``chrome://tracing`` page in chrome browser.
    Note:
        When used together with NCCL on older version of GPUs,
        autograd profiler may cause deadlock because it unnecessarily allocates
        memory on every device it sees. The memory management calls, if
        interleaved with NCCL calls, lead to deadlock on GPUs that do not
        support `cudaLaunchCooperativeKernelMultiDevice`.
    """

    def __init__(self, enable_predicate, output_dir, *, use_cuda=True):
        """
        Args:
            enable_predicate (callable[trainer -> bool]): a function which takes a trainer,
                and returns whether to enable the profiler.
                It will be called once every step, and can be used to select which steps to profile.
            output_dir (str): the output directory to dump tracing files.
            use_cuda (bool): same as in `torch.autograd.profiler.profile`.
        """
        self._enable_predicate = enable_predicate
        self._use_cuda = use_cuda
        self._output_dir = output_dir

    def before_step(self):
        if self._enable_predicate(self.trainer):
            self._profiler = torch.autograd.profiler.profile(use_cuda=self._use_cuda)
            self._profiler.__enter__()
        else:
            self._profiler = None

    def after_step(self):
        if self._profiler is None:
            return
        self._profiler.__exit__(None, None, None)
        out_file = os.path.join(
            self._output_dir, "profiler-trace-iter{}.json".format(self.trainer.iter)
        )
        if "://" not in out_file:
            self._profiler.export_chrome_trace(out_file)
        else:
            # Support non-posix filesystems
            with tempfile.TemporaryDirectory(prefix="fastreid_profiler") as d:
                tmp_file = os.path.join(d, "tmp.json")
                self._profiler.export_chrome_trace(tmp_file)
                with open(tmp_file) as f:
                    content = f.read()
            with PathManager.open(out_file, "w") as f:
                f.write(content)


class EvalHook(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.
    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_period, eval_function):
        """
        Args:
            eval_period (int): the period to run `eval_function`.
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.
        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self._period = eval_period
        self._func = eval_function

    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    )
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Remove extra memory cache of main process due to evaluation
        torch.cuda.empty_cache()

    def after_epoch(self):
        next_epoch = self.trainer.epoch + 1
        is_final = next_epoch == self.trainer.max_epoch
        if next_epoch >= 200:
            self._period = 10
        if is_final or (self._period > 0 and next_epoch % self._period == 0):
            self._do_eval()
        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()

    def after_train(self):
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._func

class Save_Checkpoint(HookBase):
    def __init__(self, model, period, cfg):
        self.model = model
        self.period = period
        self.path = cfg.OUTPUT_DIR

    def after_epoch(self):
        if (self.trainer.epoch + 1) % self.period == 0:
            save_checkpoint({
                'state_dict': self.model.state_dict()}, True,
                fpath=osp.join(self.path, 'paring_net_'+str(self.trainer.epoch + 1)+'.pth'))



class PreciseBN(HookBase):
    """
    The standard implementation of BatchNorm uses EMA in inference, which is
    sometimes suboptimal.
    This class computes the true average of statistics rather than the moving average,
    and put true averages to every BN layer in the given model.
    It is executed after the last iteration.
    """

    def __init__(self, model, data_loader, num_iter):
        """
        Args:
            model (nn.Module): a module whose all BN layers in training mode will be
                updated by precise BN.
                Note that user is responsible for ensuring the BN layers to be
                updated are in training mode when this hook is triggered.
            data_loader (iterable): it will produce data to be run by `model(data)`.
            num_iter (int): number of iterations used to compute the precise
                statistics.
        """
        self._logger = logging.getLogger(__name__)
        if len(get_bn_modules(model)) == 0:
            self._logger.info(
                "PreciseBN is disabled because model does not contain BN layers in training mode."
            )
            self._disabled = True
            return

        self._model = model
        self._data_loader = data_loader
        self._num_iter = num_iter
        self._disabled = False

        self._data_iter = None

    def before_train(self):

        self.update_stats()

    def update_stats(self):
        """
        Update the model with precise statistics. Users can manually call this method.
        """
        if self._disabled:
            return

        if self._data_iter is None:
            self._data_iter = iter(self._data_loader)

        def data_loader():
            for num_iter in itertools.count(1):
                if num_iter % 100 == 0:
                    self._logger.info(
                        "Running precise-BN ... {}/{} iterations.".format(num_iter, self._num_iter)
                    )
                # This way we can reuse the same iterator
                yield next(self._data_iter)

        with EventStorage():  # capture events in a new storage to discard them
            self._logger.info(
                "Running precise-BN for {} iterations...  ".format(self._num_iter)
                + "Note that this could produce different statistics every time."
            )
            update_bn_stats(self._model, data_loader(), self._num_iter)


class LayerFreeze(HookBase):
    def __init__(self, model, freeze_layers, freeze_iters, fc_freeze_iters):
        self._logger = logging.getLogger(__name__)

        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.model = model

        self.freeze_layers = freeze_layers
        self.freeze_iters = freeze_iters
        self.fc_freeze_iters = fc_freeze_iters

        self.is_frozen = False
        self.fc_frozen = False

    def before_step(self):
        # Freeze specific layers
        if self.trainer.iter < self.freeze_iters and not self.is_frozen:
            self.freeze_specific_layer()

        # Recover original layers status
        if self.trainer.iter >= self.freeze_iters and self.is_frozen:
            self.open_all_layer()

        if self.trainer.max_iter - self.trainer.iter <= self.fc_freeze_iters \
                and not self.fc_frozen:
            self.freeze_classifier()

    def freeze_classifier(self):
        for p in self.model.heads.classifier.parameters():
            p.requires_grad_(False)

        self.fc_frozen = True
        self._logger.info("Freeze classifier training for "
                          "last {} iterations".format(self.fc_freeze_iters))

    def freeze_specific_layer(self):
        for layer in self.freeze_layers:
            if not hasattr(self.model, layer):
                self._logger.info(f'{layer} is not an attribute of the model, will skip this layer')

        for name, module in self.model.named_children():
            if name in self.freeze_layers:
                # Change BN in freeze layers to eval mode
                module.eval()
                for p in module.parameters():
                    p.requires_grad_(False)

        self.is_frozen = True
        freeze_layers = ", ".join(self.freeze_layers)
        self._logger.info(f'Freeze layer group "{freeze_layers}" training for {self.freeze_iters:d} iterations')

    def open_all_layer(self):
        for name, module in self.model.named_children():
            if name in self.freeze_layers:
                module.train()
                for p in module.parameters():
                    p.requires_grad_(True)

        self.is_frozen = False

        freeze_layers = ", ".join(self.freeze_layers)
        self._logger.info(f'Open layer group "{freeze_layers}" training')


class SWA(HookBase):
    def __init__(self, swa_start: int, swa_freq: int, swa_lr_factor: float, eta_min: float, lr_sched=False, ):
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr_factor = swa_lr_factor
        self.eta_min = eta_min
        self.lr_sched = lr_sched

    def before_step(self):
        is_swa = self.trainer.iter == self.swa_start
        if is_swa:
            # Wrapper optimizer with SWA
            self.trainer.optimizer = optim.SWA(self.trainer.optimizer, self.swa_freq, self.swa_lr_factor)
            self.trainer.optimizer.reset_lr_to_swa()

            if self.lr_sched:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer=self.trainer.optimizer,
                    T_0=self.swa_freq,
                    eta_min=self.eta_min,
                )

    def after_step(self):
        next_iter = self.trainer.iter + 1

        # Use Cyclic learning rate scheduler
        if next_iter > self.swa_start and self.lr_sched:
            self.scheduler.step()

        is_final = next_iter == self.trainer.max_iter
        if is_final:
            self.trainer.optimizer.swap_swa_param()

class ForwardHook(HookBase):

    def __init__(self, model, dataloader):

        self.model = model
        self.dataloader = dataloader

    def after_epoch(self):
        with torch.no_grad():
            for i, data in enumerate(self.dataloader):
                if i <= 50:
                    _ = self.model(data,only_farward=True)
                else:break

import torchvision.transforms as T
from fastreid.data.transforms import ToTensor
import glob
import matplotlib.pyplot as plt
from fastreid.data.common import CommDataset
import collections
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import os
import random

class Visu_Hook(HookBase):

    def __init__(self, model, name):
        self.model = model
        self.name = name

    def after_train(self):
        #breakpoint()
        save_path_test_tsne = os.path.join(self.name, 'distribution_test_tsne.pdf')
        save_path_test_mds = os.path.join(self.name, 'distribution_test_mds.pdf')
        save_path_train_tsne = os.path.join(self.name, 'distribution_train_tsne.pdf')
        save_path_train_mds = os.path.join(self.name, 'distribution_train_mds.pdf')

        res = []
        res.append(T.Resize((256, 128), interpolation=3))
        res.append(ToTensor())
        trans = T.Compose(res)

        data_IR = []
        data_RGB = []
        #test_IR_path = '/home/amax/zn/fast-reid-orig/datasets/SYSU_MM01_SCT_new//query'
        test_IR_path = '/home/amax/zn/fast-reid-orig/datasets/RegDB/Infrad'
        img_IR_paths = glob.glob(os.path.join(test_IR_path, '*.bmp'))
        #test_RGB_path = '/home/amax/zn/fast-reid-orig/datasets/SYSU_MM01_SCT_new/gallery'
        test_RGB_path = '/home/amax/zn/fast-reid-orig/datasets/RegDB/RGB'
        img_RGB_paths = glob.glob(os.path.join(test_RGB_path, '*.bmp'))

        for index, img_path in enumerate(img_IR_paths):
            basename = os.path.basename(img_path)
            pid = int(basename.split('_')[0])
            cid = int(basename.split('_')[1][1])
            data_IR.append([img_path, pid, cid, 0, None])

        for index, img_path in enumerate(img_RGB_paths):
            basename = os.path.basename(img_path)
            pid = int(basename.split('_')[0])
            cid = int(basename.split('_')[1][1])
            data_RGB.append([img_path, pid, cid, 0, None])

        colors = ['r', 'dodgerblue', 'limegreen', 'pink', 'darkorange', 'blue']

        visu_data_IR = CommDataset(data_IR, transform=trans, transform_va=trans, relabel=False)
        visu_loader_IR = torch.utils.data.DataLoader(visu_data_IR, batch_size=128, shuffle=False, drop_last=False)

        visu_data_RGB = CommDataset(data_RGB, transform=trans, transform_va=trans, relabel=False)
        visu_loader_RGB = torch.utils.data.DataLoader(visu_data_RGB, batch_size=128, shuffle=False, drop_last=False)

        visu_data_ir = collections.defaultdict(list)
        visu_data_rgb = collections.defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for i, batched_inputs in enumerate(visu_loader_IR):
                features,_ = self.model(batched_inputs,visu=True)
                targets = batched_inputs["targets"]
                camids = batched_inputs['camids']
                for j, (feature, pid, cid) in enumerate(zip(features, targets, camids)):
                    visu_data_ir['features'].append(feature)
                    visu_data_ir['pid'].append(pid)
                    visu_data_ir['cid'].append(cid)

            for i, batched_inputs in enumerate(visu_loader_RGB):
                features,_ = self.model(batched_inputs,visu=True)
                targets = batched_inputs["targets"]
                camids = batched_inputs['camids']
                for j, (feature, pid, cid) in enumerate(
                        zip(features, targets, camids)):
                    visu_data_rgb['features'].append(feature)
                    visu_data_rgb['pid'].append(pid)
                    visu_data_rgb['cid'].append(cid)

        features_ir = visu_data_ir['features']
        features_ir = torch.stack(features_ir)

        features_rgb = visu_data_rgb['features']
        features_rgb = torch.stack(features_rgb)

        feature_conca = torch.cat((features_ir, features_rgb), dim=0).cpu()

        y_ir = torch.tensor(visu_data_ir.pop('cid'))

        y_rgb = torch.tensor(visu_data_rgb.pop('cid'))

        y_conca = torch.cat((y_ir, y_rgb), dim=0)
        #y_conca = LabelEncoder().fit(y_conca).transform(y_conca)
        uniq_cid = torch.unique(y_conca)

        X = StandardScaler().fit(feature_conca).transform(feature_conca)

        tsne = TSNE(n_components=2)
        X = tsne.fit_transform(X)

        #mds = MDS(n_components=2)
       # X_mds = mds.fit_transform(X)

        colors = ['r', 'dodgerblue', 'limegreen', 'pink', 'darkorange', 'blue']
        plt.figure()  # (figsize=(8, 8))
        plt.axis('off')
        for cid,c in zip(uniq_cid, colors):
            index = torch.where(cid == y_conca)[0]
            plt.scatter(X[index, 0], X[index, 1], color=c, s=2, linewidths=0.3)
        plt.show()
        plt.savefig(save_path_test_tsne)
        plt.close()

        '''
        cid2data = collections.defaultdict(list)
        train_data = []
        pid2item = collections.defaultdict(list)
        train_path = '/home/wenhang/data/SYSU_MM01_SCT_new'

        for root, dirs, _ in os.walk(train_path):
            for dir in dirs:
                if dir == 'Infrad' or 'RGB':
                    img_IR_paths = glob.glob(os.path.join(root, dir, '*.jpg'))
                    for img_path in img_IR_paths:
                        basename = os.path.basename(img_path)
                        pid = int(basename.split('_')[0])
                        cid = int(basename.split('_')[1][1])
                        cid2data[cid].append([img_path, pid, cid, 0, None])
                        pid2item[pid].append([img_path, pid, cid, 0, None])

        seleceted_pid = random.sample(pid2item.keys(),50)
        for pid,items in pid2item.items():
            if pid in seleceted_pid:
                train_data.extend(items)

        train_data = CommDataset(train_data, transform=trans, transform_va=trans, relabel=False)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, drop_last=False)

        visu_data_train = collections.defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for i, batched_inputs in enumerate(train_loader):
                features, _ = self.model(batched_inputs)
                targets = batched_inputs["targets"]
                camids = batched_inputs['camids']
                for j, (feature, pid, cid) in enumerate(zip(features, targets, camids)):
                    visu_data_train['features'].append(feature)
                    visu_data_train['pid'].append(pid)
                    visu_data_train['cid'].append(cid)

        features_train = visu_data_train['features']
        features_train = torch.stack(features_train).cpu()

        cids_train = torch.tensor(visu_data_train.pop('cid'))

        uniq_cid = torch.unique(cids_train)

        X = StandardScaler().fit(features_train).transform(features_train)

        tsne = TSNE(n_components=2)
        X = tsne.fit_transform(X)

        plt.figure()  # (figsize=(8, 8))
        plt.axis('off')
        for cid, c in zip(uniq_cid, colors):
            index = torch.where(cid == cids_train)[0]
            plt.scatter(X[index, 0], X[index, 1], color=c, s=2, linewidths=0.3)
        plt.show()
        plt.savefig(save_path_train_tsne)
        plt.close()
        '''

        print("visual done!!!")
class BN_hook(HookBase):
    def __init__(self, model, name):
        self.model = model
        self.name = name
    def before_train(self):
        plt.figure()

    def after_epoch(self):
        save_path = os.path.join(self.name, 'BN.pdf')
        plt.show()
        plt.savefig(save_path)

class Forward_hook(HookBase):
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def before_train(self):
        self.model.eval()
        with torch.no_grad():
            for _ in range(100):
                data = next(iter(self.dataloader))
                self.model(data)
        self.model.train()

from fastreid.utils.parsing_utils import *
import torchvision.transforms as transforms
from torch.utils import data
import cv2
from fastreid.utils.transform import get_affine_transform
from torch.utils.data import DataLoader
import tqdm
from fastreid.data.data_utils import read_image


class SimpleFolderDataset(data.Dataset):
    def __init__(self, root, input_size=[512, 512], transform=None):
        self.root = root
        self.input_size = input_size
        self.transform = transform
        self.aspect_ratio = input_size[1] * 1.0 / input_size[0]
        self.input_size = np.asarray(input_size)

        self.file_list = os.listdir(self.root)

    def __len__(self):
        return len(self.file_list)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def __getitem__(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.root, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, _ = img.shape

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        #input = read_image(img_path)
        input = self.transform(input)
        meta = {
            'name': img_name,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return input, meta

class Parsing_hook(HookBase):
    def __init__(self, parsing_net, finetune_net):
        self.parsing_net = parsing_net
        self.finetune_net = finetune_net

    def after_epoch(self):
        transform = transforms.Compose([
            #T.Resize([256,128], interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])
        dataset = SimpleFolderDataset(root='/home/amax/zn/fast-reid-orig/visu/test_data'
                                      , input_size=[256,128], transform=transform)
        self.finetune_net.eval()
        dataloader = DataLoader(dataset, batch_size=len(dataset))
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                image, meta = batch
                img_name = meta['name']
                x2, x3, x4, x5 = self.parsing_net(image.cuda())
                parsing_result = self.finetune_net(x2.detach(), x3.detach(), x4.detach(), x5.detach())
                scale_pred = F.interpolate(input=parsing_result, size=(256, 128), mode='bilinear', align_corners=True)
                scale_pred_img = parsing2img(scale_pred)
                scale_pred_mask = parsing2mask(scale_pred)

                visualize_batch(scale_pred_img, img_name,
                                '/home/amax/zn/fast-reid-orig/visu/output_img')
                visualize_batch(scale_pred_mask, img_name,
                                '/home/amax/zn/fast-reid-orig/visu/output_mask')
        self.finetune_net.train()




