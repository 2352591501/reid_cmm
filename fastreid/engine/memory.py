import torch
import collections
import torch.nn.functional as F
from fastreid.utils.my_tools import build_G, generate
import collections
from fastreid.utils.my_tools import preprocess_image, multi_nll_loss, find_positive


class UpdateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, indices, camids_batch, features, camids, momentum, cid2index):
        loss_intra = 0
        loss_inter = 0
        ctx.features = features
        ctx.momentum = momentum
        ctx.camids = camids
        mask = camids_batch.view(64, 1).eq(camids)
        for i, (input, pid, cid) in enumerate(zip(inputs, indices, camids_batch)):
            index = cid2index[int(cid)]
            index.sort()
            target = torch.tensor(index.index(pid)).unsqueeze(0).cuda()
            input = input.unsqueeze(0)
            mask_temp = mask[i]
            intra_class_features = features[mask_temp]
            output = input.mm(intra_class_features.t())
            output /= 0.07
            logit = F.log_softmax(output)
            loss_intra += F.nll_loss(logit, target)

        ctx.save_for_backward(inputs, indices)
        return loss_intra / 64  # inner-product similarity

    @staticmethod
    def backward(ctx, grad_outputs):
        # grad_outputs:similarity matrix
        inputs, indices = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)  # BP

        # update memory bank on CPU
        # todo  change manner of memory update
        feature_dict = collections.defaultdict(list)
        for x, y in zip(inputs, indices):
            feature_dict[int(y)].append(x)

        for y, x in feature_dict.items():
            x = torch.stack(x)
            x = torch.mean(x, dim=0)
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1 - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()  # L2-normalization
        '''
        for x, y in zip(inputs, indices):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1 - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()  # L2-normalization
        '''
        return grad_inputs, None, None, None


class Memory(torch.nn.Module):
    # before train:construct the memory bank of proxy level features
    def __init__(self, num_feature_dims, num_samples, temp=0.07, momentum=0.2, memory_init=[]):
        super(Memory, self).__init__()
        self.num_features = num_feature_dims
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp

        self.model = memory_init['model']
        self.dataloader = memory_init['dataloader']

        self.cam2index = collections.defaultdict(list)

        self.record_posi_list = []

        # memory bank storage structure
        self.register_buffer('features', torch.zeros((self.num_samples, self.num_features)))
        self.register_buffer('labels', torch.zeros(num_samples).long())
        self.register_buffer('camids', torch.zeros(num_samples).long())
        self.register_buffer('add_labels', torch.zeros(num_samples).long())
        self.add_labels.fill_(5000)

    def before_train(self):
        self.init_memory_bank()

    def _update_memory(self, inputs, indices, camids_batch, features, camids, momentum, cid2index):
        return UpdateFunction.apply(inputs, indices, camids_batch, features, camids,
                                    torch.Tensor([momentum]).to(inputs.device), cid2index)

    def get_true2label(self):
        class2cid = {}
        cid_all = {}
        cid_sorted = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.dataloader):
                if i <= 500:
                    ture_labels = data['targets']
                    cids = data['camids']
                    for pid, cid in zip(ture_labels, cids):
                        cid_all[int(pid)] = int(cid)
                else:
                    break
            for idx in sorted(cid_all.keys()):
                cid_sorted.append(cid_all[idx])

            for i, data in enumerate(self.dataloader):
                if i <= 500:
                    ture_labels = data['targets']
                    cids = data['camids']
                    for pid, cid in zip(ture_labels, cids):
                        class2cid[int(pid)] = (cid != torch.tensor(cid_sorted)).long()
                else:
                    break
        self.model.train()
        return class2cid, cid_all

    def init_memory_bank(self):
        '''
            Initialize memory bank with tracklet centroids.
            return Initialized memory bank.
        '''
        feature_dict = collections.defaultdict(list)
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.dataloader):
                if (i + 1) % 50 == 0:
                    print('initial memory process{}/{}:'.format((i + 1), len(self.dataloader)))
                feats = self.model(data)
                fnames = data['img_paths']
                pids = data['targets']
                camids = data['camids']
                for fname, feat, pid, camid in zip(fnames, feats, pids, camids):
                    feature_dict[int(pid)].append(feat.unsqueeze(0))
                    self.labels[int(pid)] = pid.cuda()
                    self.camids[int(pid)] = camid
                    if int(pid) not in self.cam2index[int(camid)]:
                        self.cam2index[int(camid)].append(int(pid))
                '''
                # generate fake imgs
                fake_imgs_list = generate(self.G, data)  # a list whose len is 6, in each including 64 imgs for one camera
                for cid, fake_imgs in enumerate(fake_imgs_list):
                    cid = cid + 1
                    fake_feats = self.model(fake_imgs, if_fake=True)
                    for fname, feat, pid in zip(fnames,fake_feats,pids):
                        feature_dict[int(pid) + cid * self.num_samples].append(feat.unsqueeze(0))
                        self.labels[int(pid) + cid * self.num_samples] = (pid + cid * self.num_samples).cuda()
                '''
        for pid, feature_list in feature_dict.items():
            features = torch.stack(feature_list)
            self.features[pid] = torch.mean(features, dim=0)

        self.model.train()
        for cid, indexs in self.cam2index.items():
            print('camera {} has {} identities'.format(cid, len(indexs)))

        print('>>> Memory bank is initiated with feature shape {}, label shape {}.'.format(self.features.shape,
                                                                                           self.labels.shape))

    def find_posi(self, true2label):
        add_labels = torch.zeros(self.num_samples).long()
        add_labels.fill_(5000)
        add_id_dict = find_positive(self.features[:self.num_samples], true2label)
        for i, added in add_id_dict.items():
            self.add_labels[i] = torch.tensor(added).cuda()
            add_labels[i] = torch.tensor(added).cuda()
        print(self.add_labels, add_labels)
        self.record_posi_list.append(add_labels)
        torch.save(self.add_labels, '/home/amax/fast-reid-orig/past_posi/add_labels')
        torch.save(obj=self.record_posi_list, f='/home/amax/fast-reid-orig/past_posi/recod_posi.pth')

    def forward(self, inputs, indices, batch_inputs, index2label):
        '''
        indices_origi = indices

        # update the memory bank with fake imgs
        fake_imgs_list = generate(self.G, batch_inputs)
        for cid, fake_imgs in enumerate(fake_imgs_list):
            cid = cid + 1
            fake_feats = self.model(fake_imgs, if_fake=True)
            inputs['features'] = torch.cat((inputs['features'], fake_feats['features']))
            indices_ = indices + cid * self.num_samples
            indices = torch.cat((indices,indices_))
        '''
        batch_inputs['camids'] = batch_inputs['camids'].cuda()
        loss = self._update_memory(inputs['features'], indices, batch_inputs['camids'], self.features,
                                   self.camids, self.momentum, self.cam2index)  # -> shape: (bs, num_samples)
        loss_dict = {}
        loss_dict['intra_camera'] = loss

        return loss_dict

        '''

        multi_targets_inter = []
        targets = self.labels[indices_origi].clone()

        for current in targets:
            posi = index2label[int(current)]
            temp = [int(current)]
            for i in posi:
                temp.append(int(i))
            multi_targets_inter.append(temp)

        if 1 == 2:#any(self.add_labels) is not False:
            for current in targets:
                added = int(self.add_labels[int(current)])
                assert int(current)!=int(added)
                if added != 5000:
                    temp = [int(current), int(added)]
                    multi_targets_inter.append(temp)
                else:
                    temp = [int(current)]
                    multi_targets_inter.append(temp)

            if len(self.record_posi_list) != 0:
                for past in self.record_posi_list:
                    for num, current in enumerate(targets):
                        added = int(self.add_labels[int(current)])
                        past_added = int(past[int(current)])
                        if past_added != added and past_added != 5000:
                            multi_targets_inter[num].append(past_added)
                        else:continue


        else:
            for cid in range(7):
                for current in targets:
                    current = int(current)
                    temp = [current, current+self.num_samples, current+ 2 * self.num_samples, current+3*self.num_samples
                        , current+4*self.num_samples,current+5*self.num_samples,current+6*self.num_samples]
                    multi_targets_inter.append(temp)



        loss_dict = {}
        logits = F.log_softmax(inputs, dim=1)
        loss_dict['total_loss'] = multi_nll_loss(logits, multi_targets_inter)
        return loss_dict


        #generate mutli labels
        #todo 1.imgs as well as corresponding fake imgs in memory and 2. find similar id in memory
        targets = self.labels[indices].clone()
        multi_targets_inter = []
        if any(self.add_labels) is not False:
            for current,added in zip(targets, self.add_labels):
                temp = [int(current), added]
                multi_targets_inter.append(temp)
            if len(self.record_posi_list) != 0:
                for past in self.record_posi_list:
                    for index,(i,j) in enumerate(zip(past,self.add_labels)):
                        if i != j:
                            multi_targets_inter[index].append(i)

        multi_targets_inter = torch.tensor(multi_targets_inter).cuda()
        loss_dict = {}

        logits = F.log_softmax(inputs, dim=1)
        if any(self.add_labels) is not False:
            loss_dict['intra_loss'] = F.nll_loss(logits, targets)
            loss_dict['inter_loss'] = multi_nll_loss(logits, multi_targets_inter)
            return loss_dict
        else:
            loss_dict['intra_loss'] = F.nll_loss(logits, targets)

        '''
