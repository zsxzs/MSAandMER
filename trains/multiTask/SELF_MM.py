import logging
import os
import pickle as plk

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')
emotions = ['happiness', 'sadness', 'anger', 'surprise', 'disgust', 'fear']

class SELF_MM():
    def __init__(self, args):
        assert args.train_mode == 'regression' or args.train_mode == 'classify'

        self.args = args
        # self.args.tasks = "MTAV"
        self.args.tasks = "M"

        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

        self.criterion1 = nn.L1Loss()
        self.criterion2 = nn.CrossEntropyLoss()

        # self.feature_map = {
        #     'fusion': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
        #     'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
        #     'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
        #     'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device),
        # }

        # self.center_map = {
        #     'fusion': {
        #         'pos': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
        #         'neg': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
        #     },
        #     'text': {
        #         'pos': torch.zeros(args.post_text_dim, requires_grad=False).to(args.device),
        #         'neg': torch.zeros(args.post_text_dim, requires_grad=False).to(args.device),
        #     },
        #     'audio': {
        #         'pos': torch.zeros(args.post_audio_dim, requires_grad=False).to(args.device),
        #         'neg': torch.zeros(args.post_audio_dim, requires_grad=False).to(args.device),
        #     },
        #     'vision': {
        #         'pos': torch.zeros(args.post_video_dim, requires_grad=False).to(args.device),
        #         'neg': torch.zeros(args.post_video_dim, requires_grad=False).to(args.device),
        #     }
        # }

        # self.dim_map = {
        #     'fusion': torch.tensor(args.post_fusion_dim).float(),
        #     'text': torch.tensor(args.post_text_dim).float(),
        #     'audio': torch.tensor(args.post_audio_dim).float(),
        #     'vision': torch.tensor(args.post_video_dim).float(),
        # }
        # # new labels
        # self.label_map = {
        #     'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
        #     'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
        #     'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
        #     'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        # }
        # # 'fusion_emotion': torch.zeros([args.train_samples, 6], requires_grad=False).to(args.device),

        # self.name_map = {
        #     'M': 'fusion',
        #     'T': 'text',
        #     'A': 'audio',
        #     'V': 'vision'
        # }
        # # 'M_E': 'fusion_emotion',


    def do_train(self, model, dataloader, return_epoch_results=False):
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        audio_params = list(model.Model.audio_model.named_parameters())
        video_params = list(model.Model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n and \
                                'audio_model' not in n and 'video_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)

        saved_labels = {}
        # init labels
        # logger.info("Init labels...")
        # with tqdm(dataloader['train']) as td:
        #     for batch_data in td:
        #         labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
        #         indexes = batch_data['index'].view(-1)
        #         self.init_labels(indexes, labels_m)

        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            # y_pred = {'M': [], 'M_E': [], 'T': [], 'A': [], 'V': []}
            # y_true = {'M': [], 'M_E': [], 'T': [], 'A': [], 'V': []}
            y_pred = {'M': [], 'M_E': []}
            y_true = {'M': [], 'M_E': []}
            losses = []
            model.train()
            train_loss = 0.0
            train_msa_loss = 0.0
            train_mer_loss = 0.0
            left_epochs = self.args.update_epochs
            ids = []
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    # indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    batch_size = len(cur_id)
                    labels_m = batch_data['labels']['M'].to(self.args.device)
                    labels_em = batch_data['labels']['M_E'].to(self.args.device)
                    ids.extend(cur_id)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths']
                        vision_lengths = batch_data['vision_lengths']
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    # forward
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    # store results
                    y_pred['M'].append(outputs['M'].cpu())
                    y_true['M'].append(labels_m.cpu())

                    # y_pred['M_E'].append(outputs['M_E'].cpu())
                    # y_true['M_E'].append(labels_em.cpu())

                    # for m in self.args.tasks:
                    #     y_pred[m].append(outputs[m].cpu())
                    #     y_true[m].append(self.label_map[self.name_map[m]][indexes].cpu())
                    # # add multi-emotion results
                    # # y_pred['M_E'].append(outputs['M_E'].cpu())
                    # # y_true['M_E'].append(self.label_map[self.name_map['M_E']][indexes].cpu())

                    # compute loss
                    loss = 0.0
                    # for m in self.args.tasks:
                    #     loss += self.weighted_loss(outputs[m], self.label_map[self.name_map[m]][indexes], \
                    #                                 indexes=indexes, mode=self.name_map[m])
                        
                    # # add emotion loss
                    # # loss += self.weighted_loss(outputs['M_E'], self.label_map[self.name_map['M_E']][indexes], 
                    # #                            indexes=indexes, mode='fusion')
                    msa_loss = self.criterion1(outputs['M'], labels_m)
                    loss += msa_loss

                    outputs['M_E'] = outputs['M_E'].view(batch_size * 6, 4)
                    labels_em = labels_em.view(-1)
                    mer_loss = self.criterion2(outputs['M_E'], labels_em) # * 6

                    # mer_loss = F.l1_loss(outputs['M_E'], labels_em)
                    # loss += mer_loss * 5
                    loss += mer_loss

                    # backward
                    loss.backward()
                    train_loss += loss.item()
                    train_msa_loss += msa_loss.item()
                    train_mer_loss += mer_loss.item()

                    # update features
                    # f_fusion = outputs['Feature_f'].detach()
                    # f_text = outputs['Feature_t'].detach()
                    # f_audio = outputs['Feature_a'].detach()
                    # f_vision = outputs['Feature_v'].detach()
                    # if epochs > 1:
                    #     self.update_labels(f_fusion, f_text, f_audio, f_vision, epochs, indexes, outputs)

                    # self.update_features(f_fusion, f_text, f_audio, f_vision, indexes)
                    # self.update_centers()
                    
                    # update parameters
                    if not left_epochs:
                        # update
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])
            train_msa_loss = train_msa_loss / len(dataloader['train'])
            train_mer_loss = train_mer_loss / len(dataloader['train'])
            
            # for m in self.args.tasks:
            #     pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
            #     train_results = self.metrics(pred, true)
            #     logger.info('%s: >> ' %(m) + dict_to_str(train_results))

            train_results = dict()
            # task1
            pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
            train_results['task1'] = self.metrics(pred, true)
            logger.info('%s: >> ' %('MSA (M)') + dict_to_str(train_results['task1']))

            # task2
            # pred, true = torch.cat(y_pred['M_E']), torch.cat(y_true['M_E'])
            # train_results['task2'] = {}
            # for i, emotion in enumerate(emotions):
            #     train_results['task2'][emotion] = self.metrics(pred[:, i], true[:, i])
            #     logger.info('%s: >> ' %('MER (' + emotion + ')') + dict_to_str(train_results['task2'][emotion]))

            # validation
            val_results = self.do_test(model, dataloader['valid'], mode='valid')
            # cur_valid = val_results[self.args.KeyEval]
            cur_valid = val_results['MSA Loss']


            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            # save labels
            if self.args.save_labels:
                tmp_save = {k: v.cpu().numpy() for k, v in self.label_map.items()}
                tmp_save['ids'] = ids
                saved_labels[epochs] = tmp_save

            # epoch results
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode='test')
                epoch_results['test'].append(test_results)
                
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                if self.args.save_labels:
                    with open(os.path.join(self.args.res_save_dir, f'{self.args.model_name}-{self.args.dataset_name}-labels.pkl'), 'wb') as df:
                        plk.dump(saved_labels, df, protocol=4)
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode='test', return_sample_results=False):
        model.eval()
        # y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        # y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        y_pred = {'M': [], 'M_E': []}
        y_true = {'M': [], 'M_E': []}

        eval_loss = 0.0
        eval_msa_loss = 0.0
        eval_mer_loss = 0.0

        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        # criterion = nn.L1Loss()
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths']
                        vision_lengths = batch_data['vision_lengths']
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    labels_m = batch_data['labels']['M'].to(self.args.device)
                    labels_em = batch_data['labels']['M_E'].to(self.args.device)
                    batch_size = labels_m.size()[0]
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        for item in features.keys():
                            features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels_m.cpu().detach().tolist())
                        preds = outputs["M"].cpu().detach().numpy()
                        # test_preds_i = np.argmax(preds, axis=1)
                        sample_results.extend(preds.squeeze())
                    
                    # loss = self.weighted_loss(outputs['M'], labels_m)
                    loss = 0.0
                    msa_loss = F.l1_loss(outputs['M'], labels_m)
                    loss += msa_loss

                    outputs['M_E'] = outputs['M_E'].view(batch_size * 6, 4)
                    labels_em = labels_em.view(-1)
                    mer_loss = self.criterion2(outputs['M_E'], labels_em) # * 6
                    # mer_loss = F.l1_loss(outputs['M_E'], labels_em)
                    # loss += mer_loss * 5
                    loss += mer_loss

                    eval_loss += loss.item()
                    eval_msa_loss += msa_loss.item()
                    eval_mer_loss += mer_loss.item()

                    y_pred['M'].append(outputs['M'].cpu())
                    y_true['M'].append(labels_m.cpu())
                    y_pred['M_E'].append(outputs['M_E'].cpu())
                    y_true['M_E'].append(labels_em.cpu())

        eval_loss = eval_loss / len(dataloader)
        eval_msa_loss = eval_msa_loss / len(dataloader)
        eval_mer_loss = eval_mer_loss / len(dataloader)

        logger.info(mode + "(%s)" % self.args.model_name + " >> sum loss: %.4f " % eval_loss + \
                    " >> msa loss: %.4f " % eval_msa_loss + " >> mer loss: %.4f " % eval_mer_loss)
        
        eval_results = dict()
        eval_results['Loss'] = round(eval_loss, 4)
        eval_results['MSA Loss'] = round(eval_msa_loss, 4)
        eval_results['MER Loss'] = round(eval_mer_loss, 4)


        # task1
        pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
        eval_results['task1'] = self.metrics(pred, true)
        logger.info('%s: >> ' %('MSA (M)') + dict_to_str(eval_results['task1']))

        # task2
        # pred, true = torch.cat(y_pred['M_E']), torch.cat(y_true['M_E'])
        # eval_results['task2'] = {}
        # for i, emotion in enumerate(emotions):
        #     eval_results['task2'][emotion] = self.metrics(pred[:, i], true[:, i])
        #     logger.info('%s: >> ' %('MER (' + emotion + ')') + dict_to_str(eval_results['task2'][emotion]))

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
    
    def weighted_loss(self, y_pred, y_true, indexes=None, mode='fusion'):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if mode == 'fusion':
            weighted = torch.ones_like(y_pred)
        else:
            weighted = torch.tanh(torch.abs(self.label_map[mode][indexes] - self.label_map['fusion'][indexes]))
        loss = torch.mean(weighted * torch.abs(y_pred - y_true))
        return loss
    
    def update_features(self, f_fusion, f_text, f_audio, f_vision, indexes):
        self.feature_map['fusion'][indexes] = f_fusion
        self.feature_map['text'][indexes] = f_text
        self.feature_map['audio'][indexes] = f_audio
        self.feature_map['vision'][indexes] = f_vision

    def update_centers(self):
        def update_single_center(mode):
            neg_indexes = self.label_map[mode] < 0
            if self.args.excludeZero:
                pos_indexes = self.label_map[mode] > 0
            else:
                pos_indexes = self.label_map[mode] >= 0
            self.center_map[mode]['pos'] = torch.mean(self.feature_map[mode][pos_indexes], dim=0)
            self.center_map[mode]['neg'] = torch.mean(self.feature_map[mode][neg_indexes], dim=0)

        update_single_center(mode='fusion')
        update_single_center(mode='text')
        update_single_center(mode='audio')
        update_single_center(mode='vision')
    
    def init_labels(self, indexes, m_labels):
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['audio'][indexes] = m_labels
        self.label_map['vision'][indexes] = m_labels
    
    def update_labels(self, f_fusion, f_text, f_audio, f_vision, cur_epoches, indexes, outputs):
        MIN = 1e-8
        def update_single_label(f_single, mode):
            d_sp = torch.norm(f_single - self.center_map[mode]['pos'], dim=-1) 
            d_sn = torch.norm(f_single - self.center_map[mode]['neg'], dim=-1) 
            delta_s = (d_sn - d_sp) / (d_sp + MIN)
            # d_s_pn = torch.norm(self.center_map[mode]['pos'] - self.center_map[mode]['neg'], dim=-1)
            # delta_s = (d_sn - d_sp) / (d_s_pn + MIN)
            alpha = delta_s / (delta_f + MIN)

            new_labels = 0.5 * alpha * self.label_map['fusion'][indexes] + \
                        0.5 * (self.label_map['fusion'][indexes] + delta_s - delta_f)
            new_labels = torch.clamp(new_labels, min=-self.args.H, max=self.args.H)
            # new_labels = torch.tanh(new_labels) * self.args.H

            n = cur_epoches
            self.label_map[mode][indexes] = (n - 1) / (n + 1) * self.label_map[mode][indexes] + 2 / (n + 1) * new_labels

        d_fp = torch.norm(f_fusion - self.center_map['fusion']['pos'], dim=-1) 
        d_fn = torch.norm(f_fusion - self.center_map['fusion']['neg'], dim=-1) 
        # d_f_pn = torch.norm(self.center_map['fusion']['pos'] - self.center_map['fusion']['neg'], dim=-1)
        # delta_f = (d_fn - d_fp) / (d_f_pn + MIN)
        delta_f = (d_fn - d_fp) / (d_fp + MIN)
        
        update_single_label(f_text, mode='text')
        update_single_label(f_audio, mode='audio')
        update_single_label(f_vision, mode='vision')
