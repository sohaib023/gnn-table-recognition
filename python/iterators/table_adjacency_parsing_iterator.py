import os
import subprocess
from overrides import overrides

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from readers.dataset import TableDataset
from models.basic_model import BasicModel
from iterators.iterator_interface import Iterator
from libs.inference_output_streamer import InferenceOutputStreamer
from libs.inference_output_evaluator import InferenceOutputEvaluator
from libs.configuration_manager import ConfigurationManager as gconfig

class TableAdjacencyParsingIterator (Iterator):
    def __init__(self):
        self.batch_size = gconfig.get_config_param("batch_size", int)
        self.from_scratch = gconfig.get_config_param("from_scratch", type="bool")
        self.summary_path = gconfig.get_config_param("summary_path", type="str")
        self.model_path = gconfig.get_config_param("model_path", type="str")
        self.train_for_iterations = gconfig.get_config_param("train_for_iterations", type="int")
        self.validate_after = gconfig.get_config_param("validate_after", type="int")
        self.save_after_iterations = gconfig.get_config_param("save_after_iterations", type="int")
        self.test_out_path = gconfig.get_config_param("test_out_path", type="str")
        self.visual_feedback_out_path = gconfig.get_config_param("visual_feedback_out_path", type="str")
        self.lr = gconfig.get_config_param("learning_rate", type='float')
        self.device = torch.device("cuda:0") # if torch.cuda.is_available() else "cpu")
        self.test_output_path = gconfig.get_config_param("test_out_path", type="str")

    def __clean(self, path):
        for the_file in os.listdir(path):
            file_path = os.path.join(path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)


    def clean_directories(self):
        self.__clean(self.summary_path)
        print("Cleaned summary directory")
        self.__clean(self.visual_feedback_out_path)
        print("Cleaned visual feedback output directory")

    def reduce_mean_variable_vertices(self, tensor, n_columns, n_rows):
        # tensor = [b, v, v]
        reduced = torch.sum(tensor.float(), dim=-1) / n_columns[:, None] # [b, v]
        reduced = torch.sum(reduced, dim=-1)/ n_rows          # [b]
        # reduced = reduced / (num_vertices**2)
        return torch.mean(reduced, dim=0)

    def softmax_cross_entropy(self, logits, targets, weights=[1., 1.]):
        softmax = torch.softmax(logits, dim=-1)
        p = softmax.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        p = torch.clamp(p, 1e-8, 1)
        loss = -torch.pow(1 - p, 2) * torch.log(p)

        loss_0 = loss * (1 - targets)
        loss_1 = loss * targets

        loss = weights[0] * loss_0 + weights[1] * loss_1

        return loss

    @overrides
    def train(self):
        if self.from_scratch:
            subprocess.call("mkdir -p %s"%(self.summary_path), shell=True)
            subprocess.call("mkdir -p %s"%(self.test_out_path), shell=True)
            subprocess.call("mkdir -p %s"%(os.path.join(self.test_out_path, 'ops')), shell=True)
            subprocess.call("mkdir -p %s" % (self.visual_feedback_out_path), shell=True)
            self.clean_directories()
        else:
            pass

        model = BasicModel()
        def weights_init(m):
            # if isinstance(m, nn.Conv2d):
            #     torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            #     torch.nn.init.constant_(m.bias, 0.1)
            if isinstance(m, nn.Linear):
                # torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(m.bias, 0.2)

        model.apply(weights_init)

        criterion = nn.CrossEntropyLoss(reduction="none")

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=configs.decay_rate)
        summary = SummaryWriter(self.summary_path)

        global_step = 0
        start_epoch = 0

        if not self.from_scratch:
            checkpoint = torch.load(self.model_path + ".pth")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(self.device)
            except Exception as e:
                print(e)

            start_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            global_step = checkpoint['global_step']

        model.to(self.device)
        if gconfig.get("fine_tune_classification_head", "bool"):
            for i, x in list(enumerate(model.named_parameters()))[:97]:
                # print(i, x[0])
                x[1].requires_grad = False
            for i, x in list(enumerate(model.named_parameters())):
                print(i, x[1].requires_grad, x[0])

        num_epochs = 200

        train_data = TableDataset(gconfig.get("training_files_list", str), self.device)
        val_data = TableDataset(gconfig.get("validation_files_list", str), self.device)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, num_workers=1)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, num_workers=1)

        CELL = 0
        ROW = 1
        COL = 2

        for epoch in range(start_epoch, num_epochs):
            for i, sample in enumerate(train_loader):
                print("Epoch #:", epoch, "\tIteration #:", i, "\tGlobal Step #: ", global_step)

                sample = list(sample)
                for i in range(len(sample)):
                    sample[i] = sample[i].to(self.device)

                (vertex_features, cell_ids, images, global_features, gt_down, gt_right, rows, columns, n_rows, n_columns) = sample

                model.train()

                optimizer.zero_grad()

                out_probs = model(images, vertex_features, cell_ids)

                num_vertices = global_features[:, 2]

                labels = [gt_right.long(), gt_down.long()]
                losses = []
                accuracies = []
                precisions = []
                recalls = []
                predictions = []

                masks = [   
                            labels[0] != 2, 
                            labels[1] != 2
                        ]

                suffixes = ["right", "down_"]
                for i in range(1):
                    labels[i] = torch.where(labels[i] != 2, labels[i], torch.zeros(1).long().to(self.device)[:, None, None])

                    predictions.append(torch.argmax(out_probs[i], dim=-1))

                    P_1 = (predictions[i] == 1) * masks[i]
                    P_0 = (predictions[i] == 0) * masks[i]

                    G_1 = (labels[i] == 1) * masks[i]
                    G_0 = (labels[i] == 0) * masks[i]

                    # print(torch.nonzero(P_1))
                    # print(P_1.shape)
                    print("P:", torch.nonzero(P_0).shape[0], torch.nonzero(P_1).shape[0])
                    print("G:", torch.nonzero(G_0).shape[0], torch.nonzero(G_1).shape[0])

                    tp = self.reduce_mean_variable_vertices(P_1 & G_1, n_columns - (1 - i), n_rows - i)
                    fp = self.reduce_mean_variable_vertices(P_1 & G_0, n_columns - (1 - i), n_rows - i)
                    fn = self.reduce_mean_variable_vertices(P_0 & G_1, n_columns - (1 - i), n_rows - i)
                    acc = self.reduce_mean_variable_vertices((predictions[i] == labels[i]) * masks[i], n_columns - (1 - i), n_rows - i)

                    precisions.append((tp/(tp+fp)).item()*100)
                    recalls.append((tp/(tp+fn)).item()*100)
                    accuracies.append(acc.item()*100)

                    # loss = criterion(out_probs[i].permute(0,3,1,2), labels[i]) * masks[i]
                    loss = self.softmax_cross_entropy(out_probs[i], labels[i]) * masks[i]
                    # print(loss[0, :5])
                    # print(loss2[0, :5])
                    # exit(0)

                    # loss_1 = (G_1 * loss)
                    ratio_0 = 1e-8 + torch.sum(G_0, dim=-1) / (1e-8 + n_columns - (1 - i))[:, None] # [b, v]
                    ratio_0 = 1e-8 + torch.sum(ratio_0, dim=-1) / (1e-8 + n_rows - i) # [b, v]

                    ratio_1 = 1e-8 + torch.sum(G_1, dim=-1) / (1e-8 + n_columns - (1 - i))[:, None] # [b, v]
                    ratio_1 = 1e-8 + torch.sum(ratio_1, dim=-1) / (1e-8 + n_rows - i) # [b, v]

                    # loss = ((G_0 * loss) * 0.1) + (G_1 * loss)

                    # print(torch.nonzero(torch.isnan(loss.view(-1))))
                    loss = 0.5 * ((loss * G_0) / ratio_0[:, None, None]) + 0.5 * ((loss * G_1) / ratio_1[:, None, None])
                    # print(torch.nonzero(torch.isnan(loss.view(-1))))

                    loss[torch.isnan(loss)] = 0

                    loss = torch.sum(loss, dim=-1) / (n_columns - (1 - i))[:, None] # [b, v]
                    loss = torch.sum(loss, dim=-1) / (n_rows - i)
                    loss = torch.mean(loss, dim=0)
                    assert loss.shape == torch.Size([])
                    losses.append(loss)
                    # loss_1 = torch.sum(loss_1, dim=-1) / (n_columns - (1 - i))[:, None] # [b, v]

                    # loss = 0.5 * (loss_0 / ratio_0) + 0.5 * (loss_1 / ratio_1)
                    # exit(0)

                total_loss = losses[0]
                total_loss.backward()
                # for i, x in list(enumerate(model.named_parameters())):
                #     if x[1].grad is not None:
                #         print(x[0], "\t", torch.nonzero(torch.isnan(x[1].grad)).shape[0] / torch.prod(torch.Tensor([a for a in x[1].shape])))
                # exit(0)
                optimizer.step()

                for i in range(1):
                    print("\t", suffixes[i], ":\tLoss:\t{:.6f}".format(losses[i].item()), "\tAccuracy:\t{:.3f}".format(accuracies[i]), " \tPrecision:\t{:.3f}".format(precisions[i]), " \tRecall:\t{:.3f}".format(recalls[i]))
                    summary.add_scalar("training_loss_" + suffixes[i], losses[i].item(), global_step)
                    summary.add_scalar("training_accuracy_" + suffixes[i], accuracies[i], global_step)
                    summary.add_scalar("training_precision_" + suffixes[i], precisions[i], global_step)
                    summary.add_scalar("training_recall_" + suffixes[i], recalls[i], global_step)
                print("\tTotal Loss: ", total_loss.item())

                global_step += 1

                if global_step % self.save_after_iterations == 0:
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, self.model_path + ".pth")
                if global_step % self.validate_after == 0:
                    pass    

    @overrides
    def test(self):
        model = BasicModel()
        
        checkpoint = torch.load(self.model_path + ".pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss(reduction="none")

        test_data = TableDataset(gconfig.get("test_files_list", str), self.device)

        test_loader = DataLoader(test_data, batch_size=1, num_workers=1)

        CELL = 0
        ROW = 1
        COL = 2

        self.inference_output_streamer = InferenceOutputStreamer(self.test_output_path)
        self.inference_output_streamer.start_thread()

        with torch.no_grad():
            for i, sample in enumerate(test_loader):
                print("Iteration #:", i)

                sample = list(sample)
                for i in range(len(sample)):
                    sample[i] = sample[i].to(self.device)

                (vertex_features, cell_ids, images, global_features, gt_down, gt_right, rows, columns, n_rows, n_columns) = sample
                
                out_probs = model(images, vertex_features, cell_ids)

                num_vertices = global_features[:, 2]

                labels = [gt_right.long(), gt_down.long()]
                losses = []
                accuracies = []
                precisions = []
                recalls = []
                predictions = []

                masks = [   
                            labels[0] != 2, 
                            labels[1] != 2
                        ]

                suffixes = ["right", "down_"]
                for i in range(2):
                    labels[i] = torch.where(labels[i] != 2, labels[i], torch.ones(1).long().to(self.device)[:, None, None])

                    predictions.append(torch.argmax(out_probs[i], dim=-1))

                    P_1 = (predictions[i] == 1) * masks[i]
                    P_0 = (predictions[i] == 0) * masks[i]

                    G_1 = (labels[i] == 1) * masks[i]
                    G_0 = (labels[i] == 0) * masks[i]

                    tp = self.reduce_mean_variable_vertices(P_1 & G_1, n_columns - (1 - i), n_rows - i)
                    fp = self.reduce_mean_variable_vertices(P_1 & G_0, n_columns - (1 - i), n_rows - i)
                    fn = self.reduce_mean_variable_vertices(P_0 & G_1, n_columns - (1 - i), n_rows - i)
                    acc = self.reduce_mean_variable_vertices((predictions[i] == labels[i]) * masks[i], n_columns - (1 - i), n_rows - i)

                    precisions.append((tp/(tp+fp)).item()*100)
                    recalls.append((tp/(tp+fn)).item()*100)
                    accuracies.append(acc.item()*100)

                    loss = self.softmax_cross_entropy(out_probs[i], labels[i]) * masks[i]

                    loss = torch.sum(loss, dim=-1) / (n_columns - (1 - i))[:, None] # [b, v]
                    loss = torch.sum(loss, dim=-1) / (n_rows - i)
                    loss = torch.mean(loss, dim=0)
                    assert loss.shape == torch.Size([])
                    losses.append(loss)

                for i in range(2):
                    print("\t", suffixes[i], ":\tLoss:\t{:.6f}".format(losses[i].item()), "\tAccuracy:\t{:.3f}".format(accuracies[i]), " \tPrecision:\t{:.3f}".format(precisions[i]), " \tRecall:\t{:.3f}".format(recalls[i]))

                result = {
                    'image': images[0].permute(1,2,0).cpu().numpy(),
                    'sampled_ground_truths': [labels[0][0].cpu().numpy(), labels[1][0].cpu().numpy()],
                    'sampled_predictions': [predictions[0][0].cpu().numpy(), predictions[1][0].cpu().numpy()],
                    'sampled_indices': None,
                    'global_features': global_features[0].cpu().numpy(),
                    'vertex_features': vertex_features[0].cpu().numpy(),
                    'masks': [masks[0][0].cpu().numpy(), masks[1][0].cpu().numpy()],
                    'rows': rows[0][:n_rows[0]].cpu().numpy(),
                    'columns': columns[0][:n_columns[0]].cpu().numpy()
                }

                self.inference_output_streamer.add(result)
        self.inference_output_streamer.close()

    @overrides
    def profile(self):
        return super(TableAdjacencyParsingIterator, self).profile()

    @overrides
    def evaluate(self):
        evaluator = InferenceOutputEvaluator(gconfig.get_config_param("test_out_path", type="str"))
        evaluator.evaluate()
