import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader
import dgl
import matplotlib.pyplot as plt

import pickle
from tqdm import tqdm
import argparse
import os, sys, time
import random
from pathlib import Path

from FJMP.fjmp_modules import *
from FJMP.fjmp_utils import *
from FJMP.dag_utils import *
from FJMP.fjmp_metrics import *

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", default="dev", help="a name to indicate the log path and model save path")
parser.add_argument("--gpu_start", default=0, type=int, help='gpu device i, where training will occupy gpu device i,i+1,...,i+n_gpus-1')
parser.add_argument("--resume_training", action="store_true", help="continue training from checkpoint")
parser.add_argument("--eval_training", action="store_true", help="run evaluation on training set?")

args = parser.parse_args()

GPU_START = args.gpu_start

try:
    import horovod.torch as hvd 
    can_use_hvd = True
except:
    can_use_hvd = False

from torch.utils.data.distributed import DistributedSampler
from mpi4py import MPI

comm = MPI.COMM_WORLD
if can_use_hvd:
    hvd.init()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank() + GPU_START)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_START)

dev = 'cuda:{}'.format(0)
torch.cuda.set_device(0)

# seed = hvd.rank()
# set_seeds(seed)

class FJMP(torch.nn.Module):
    def __init__(self, config):
        super(FJMP, self).__init__()
        self.config = config


        config["config_name"] = args.config_name 
        
        config["resume_training"] = args.resume_training
        config["eval_training"] = args.eval_training
        

        config["log_path"].mkdir(exist_ok=True, parents=True)
        log = os.path.join(config["log_path"], "log")
        # write stdout to log file
        sys.stdout = Logger(log)



        self.dataset = config["dataset"]
        self.num_agenttypes = config["num_agenttypes"]
        self.switch_lr_1 = config["switch_lr_1"]
        self.switch_lr_2 = config["switch_lr_2"]
        self.lr_step = config["lr_step"]
        self.input_size = config["input_size"]
        self.observation_steps = config["observation_steps"]
        self.prediction_steps = config["prediction_steps"]
        self.num_edge_types = config["num_edge_types"]
        self.h_dim = config["h_dim"]
        self.num_joint_modes = config["num_joint_modes"]
        self.num_proposals = config["num_proposals"]
        self.learning_rate = config["lr"]
        self.max_epochs = config["max_epochs"]
        self.log_path = config["log_path"]
        self.batch_size = config["batch_size"]
        self.decoder = config["decoder"]
        self.num_heads = config["num_heads"]
        self.learned_relation_header = config["learned_relation_header"]
        self.resume_training = config["resume_training"]
        self.proposal_coef = config["proposal_coef"]
        self.rel_coef = config["rel_coef"]
        self.proposal_header = config["proposal_header"]
        self.two_stage_training = config["two_stage_training"]
        self.training_stage = config["training_stage"]
        self.ig = config["ig"]
        self.focal_loss = config["focal_loss"]
        self.gamma = config["gamma"]
        self.weight_0 = config["weight_0"]
        self.weight_1 = config["weight_1"]
        self.weight_2 = config["weight_2"]
        self.teacher_forcing = config["teacher_forcing"]
        self.scheduled_sampling = config["scheduled_sampling"]
        # self.eval_training = config["eval_training"]
        self.supervise_vehicles = config["supervise_vehicles"]
        self.no_agenttype_encoder = config["no_agenttype_encoder"]
        # self.train_all = config["train_all"]
        
        if self.two_stage_training and self.training_stage == 2:
            self.pretrained_relation_header = None

        seed = config["seed"]
        set_seeds(seed)
        
        self.build()

    def build(self):
        self.feature_encoder = FJMPFeatureEncoder(self.config).to(dev)
        if self.learned_relation_header:
            self.relation_header = FJMPRelationHeader(self.config).to(dev)
        
        if self.proposal_header:
            self.proposal_decoder = FJMPTrajectoryProposalDecoder(self.config).to(dev)
        
        if (self.two_stage_training and self.training_stage == 2) or not self.two_stage_training:
            if self.decoder == 'dagnn':
                self.trajectory_decoder = FJMPAttentionTrajectoryDecoder(self.config).to(dev)
            elif self.decoder == 'lanegcn':
                self.trajectory_decoder = LaneGCNHeader(self.config).to(dev)

    def process(self, data):
        num_actors = [len(x) for x in data['feats']]
        num_edges = [int(n * (n-1) / 2) for n in num_actors]

        # LaneGCN processing 
        # ctrs gets copied once for each agent in scene, whereas actor_ctrs only contains one per scene
        # same data, but different format so that it is compatible with LaneGCN L2A/A2A function     
        actor_ctrs = gpu(data["ctrs"])
        if not any(graph is None for graph in data["graph"]):
            lane_graph = graph_gather(to_long(gpu(data["graph"])), self.config)
        else:
            lane_graph = None
        # unique index assigned to each scene
        scene_idxs = torch.Tensor([idx for idx in data['idx']])

        graph = data["graph"]

        world_locs = [x for x in data['feat_locs']]
        world_locs = torch.cat(world_locs, 0)

        has_obs = [x for x in data['has_obss']]
        has_obs = torch.cat(has_obs, 0)

        ig_labels = [x for x in data['ig_labels_{}'.format(self.ig)]]
        ig_labels = torch.cat(ig_labels, 0)

        if self.dataset == "argoverse2":
            agentcategories = [x for x in data['feat_agentcategories']]
            # we know the agent category exists at the present timestep
            agentcategories = torch.cat(agentcategories, 0)[:, self.observation_steps - 1, 0]
            # we consider scored+focal tracks for evaluation in Argoverse 2
            is_scored = agentcategories >= 2

        locs = [x for x in data['feats']]
        locs = torch.cat(locs, 0)

        vels = [x for x in data['feat_vels']]
        vels = torch.cat(vels, 0)

        psirads = [x for x in data['feat_psirads']]
        psirads = torch.cat(psirads, 0)

        gt_psirads = [x for x in data['gt_psirads']]
        gt_psirads = torch.cat(gt_psirads, 0)

        gt_vels = [x for x in data['gt_vels']]
        gt_vels = torch.cat(gt_vels, 0)

        agenttypes = [x for x in data['feat_agenttypes']]
        agenttypes = torch.cat(agenttypes, 0)[:, self.observation_steps - 1, 0]
        agenttypes = torch.nn.functional.one_hot(agenttypes.long(), self.num_agenttypes)

        # shape information is only available in INTERACTION dataset
        if self.dataset == "interaction":
            shapes = [x for x in data['feat_shapes']]
            shapes = torch.cat(shapes, 0)

        feats = torch.cat([locs, vels, psirads], dim=2)

        ctrs = [x for x in data['ctrs']]
        ctrs = torch.cat(ctrs, 0)

        orig = [x.view(1, 2) for j, x in enumerate(data['orig']) for i in range(num_actors[j])]
        orig = torch.cat(orig, 0)

        rot = [x.view(1, 2, 2) for j, x in enumerate(data['rot']) for i in range(num_actors[j])]
        rot = torch.cat(rot, 0)

        theta = torch.Tensor([x for j, x in enumerate(data['theta']) for i in range(num_actors[j])])

        gt_locs = [x for x in data['gt_preds']]
        gt_locs = torch.cat(gt_locs, 0)

        has_preds = [x for x in data['has_preds']]
        has_preds = torch.cat(has_preds, 0)

        # does a ground-truth waypoint exist at the last timestep?
        has_last = has_preds[:, -1] == 1
        
        batch_idxs = []
        batch_idxs_edges = []
        actor_idcs = []
        sceneidx_to_batchidx_mapping = {}
        count_batchidx = 0
        count = 0
        for i in range(len(num_actors)):            
            batch_idxs.append(torch.ones(num_actors[i]) * count_batchidx)
            batch_idxs_edges.append(torch.ones(num_edges[i]) * count_batchidx)
            sceneidx_to_batchidx_mapping[int(scene_idxs[i].item())] = count_batchidx
            idcs = torch.arange(count, count + num_actors[i]).to(locs.device)
            actor_idcs.append(idcs)
            
            count_batchidx += 1
            count += num_actors[i]
        
        batch_idxs = torch.cat(batch_idxs).to(locs.device)
        batch_idxs_edges = torch.cat(batch_idxs_edges).to(locs.device)
        batch_size = torch.unique(batch_idxs).shape[0]

        ig_labels_metrics = [x for x in data['ig_labels_sparse']]
        ig_labels_metrics = torch.cat(ig_labels_metrics, 0)

        # 1 if agent has out-or-ingoing edge in ground-truth sparse interaction graph
        # These are the agents we use to evaluate interactive metrics
        is_connected = torch.zeros(locs.shape[0])
        count = 0
        offset = 0
        for k in range(len(num_actors)):
            N = num_actors[k]
            for i in range(N):
                for j in range(N):
                    if i >= j:
                        continue 
                    
                    # either an influencer or reactor in some DAG.
                    if ig_labels_metrics[count] > 0:                      

                        is_connected[offset + i] += 1
                        is_connected[offset + j] += 1 

                    count += 1
            offset += N

        is_connected = is_connected > 0     

        assert count == ig_labels_metrics.shape[0]

        dd = {
            'batch_size': batch_size,
            'batch_idxs': batch_idxs,
            'batch_idxs_edges': batch_idxs_edges, 
            'actor_idcs': actor_idcs,
            'actor_ctrs': actor_ctrs,
            'lane_graph': lane_graph,
            'feats': feats,
            'feat_psirads': psirads,
            'ctrs': ctrs,
            'orig': orig,
            'rot': rot,
            'theta': theta,
            'gt_locs': gt_locs,
            'has_preds': has_preds,
            'scene_idxs': scene_idxs,
            'sceneidx_to_batchidx_mapping': sceneidx_to_batchidx_mapping,
            'ig_labels': ig_labels,
            'gt_psirads': gt_psirads,
            'gt_vels': gt_vels,
            'agenttypes': agenttypes,
            'world_locs': world_locs,
            'has_obs': has_obs,
            'has_last': has_last,
            'graph': graph,
            'is_connected': is_connected
        }

        if self.dataset == "interaction":
            dd['shapes'] = shapes

        elif self.dataset == "argoverse2":
            dd['is_scored'] = is_scored

        # dd = data-dictionary
        return dd

   
    def init_dgl_graph(self, batch_idxs, ctrs, orig, rot, agenttypes, world_locs, has_preds):        
        n_scenarios = len(np.unique(batch_idxs))
        graphs, labels = [], []
        for ii in range(n_scenarios):
            label = None

            # number of agents in the scene (currently > 0)
            si = ctrs[batch_idxs == ii].shape[0]
            assert si > 0

            # start with a fully-connected graph
            if si > 1:
                off_diag = np.ones([si, si]) - np.eye(si)
                rel_src = np.where(off_diag)[0]
                rel_dst = np.where(off_diag)[1]

                graph = dgl.graph((rel_src, rel_dst))
            else:
                graph = dgl.graph(([], []), num_nodes=si)

            # separate graph for each scenario
            graph.ndata["ctrs"] = ctrs[batch_idxs == ii]
            graph.ndata["rot"] = rot[batch_idxs == ii]
            graph.ndata["orig"] = orig[batch_idxs == ii]
            graph.ndata["agenttypes"] = agenttypes[batch_idxs == ii].float()
            # ground truth future in SE(2)-transformed coordinates
            graph.ndata["ground_truth_futures"] = world_locs[batch_idxs == ii][:, self.observation_steps:]
            graph.ndata["has_preds"] = has_preds[batch_idxs == ii].float()
            
            graphs.append(graph)
            labels.append(label)
        
        graphs = dgl.batch(graphs)
        return graphs

    def build_stage_1_graph(self, graph, x, agenttypes, actor_idcs, actor_ctrs, lane_graph):
        all_edges = [x.unsqueeze(1) for x in graph.edges('uv')]
        all_edges = torch.cat(all_edges, 1)
        
        stage_1_graph = dgl.graph((all_edges[:, 0], all_edges[:, 1]), num_nodes = graph.num_nodes())
        stage_1_graph.ndata["ctrs"] = graph.ndata["ctrs"]
        stage_1_graph.ndata["rot"] = graph.ndata["rot"]
        stage_1_graph.ndata["orig"] = graph.ndata["orig"]
        stage_1_graph.ndata["agenttypes"] = graph.ndata["agenttypes"].float()

        stage_1_graph = self.pretrained_relation_header.feature_encoder(stage_1_graph, x, agenttypes, actor_idcs, actor_ctrs, lane_graph)

        return stage_1_graph

    def forward(self, scene_idxs, graph, stage_1_graph, ig_dict, batch_idxs, batch_idxs_edges, actor_ctrs, ks=None, prop_ground_truth = 0.):
    
        if self.learned_relation_header:
            edge_logits = self.relation_header(graph)
            graph.edata["edge_logits"] = edge_logits
        else:
            # use ground-truth interaction graph
            if not self.two_stage_training:
                edge_probs = torch.nn.functional.one_hot(ig_dict["ig_labels"].to(dev).long(), self.num_edge_types)
            elif self.two_stage_training and self.training_stage == 2:
                prh_logits = self.pretrained_relation_header.relation_header(stage_1_graph)
                graph.edata["edge_logits"] = prh_logits
        
        all_edges = [x.unsqueeze(1) for x in graph.edges('all')]
        all_edges = torch.cat(all_edges, 1)
        # remove half of the directed edges (effectively now an undirected graph)
        eids_remove = all_edges[torch.where(all_edges[:, 0] > all_edges[:, 1])[0], 2]
        graph.remove_edges(eids_remove)

        if self.learned_relation_header or (self.two_stage_training and self.training_stage == 2):
            edge_logits = graph.edata.pop("edge_logits")
            edge_probs = my_softmax(edge_logits, -1)

        graph.edata["edge_probs"] = edge_probs

        dag_graph = build_dag_graph(graph, self.config)
        
        if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
            dag_graph = prune_graph_johnson(dag_graph)
        
        if self.proposal_header:
            dag_graph, proposals = self.proposal_decoder(dag_graph, actor_ctrs)
        
        if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
            loc_pred = self.trajectory_decoder(dag_graph, prop_ground_truth, batch_idxs)
        
        # loc_pred: shape [N, prediction_steps, num_joint_modes, 2]
        res = {}

        if self.proposal_header:
            res["proposals"] = proposals # trajectory proposal future coordinates
        
        if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
            res["loc_pred"] = loc_pred # predicted future coordinates
        
        if self.learned_relation_header:
            res["edge_logits"] = edge_logits.float() # edge probabilities for computing BCE loss    
            res["edge_probs"] = edge_probs.float()     
        
        return res

    def get_loss(self, graph, batch_idxs, res, agenttypes, has_preds, gt_locs, batch_size, ig_labels, epoch):
        
        huber_loss = nn.HuberLoss(reduction='none')
        
        if self.proposal_header:
            ### Proposal Regression Loss
            has_preds_mask = has_preds.unsqueeze(-1).unsqueeze(-1)
            has_preds_mask = has_preds_mask.expand(has_preds_mask.shape[0], has_preds_mask.shape[1], self.num_proposals, 2).bool().to(dev)

            proposals = res["proposals"]
            
            if self.supervise_vehicles and self.dataset=='interaction':
                # only compute loss on vehicle trajectories
                vehicle_mask = agenttypes[:, 1].bool()
            else:
                # compute loss on all trajectories
                vehicle_mask = torch.ones(agenttypes[:, 1].shape).bool().to(dev)
            
            has_preds_mask = has_preds_mask[vehicle_mask]
            proposals = proposals[vehicle_mask]
            gt_locs = gt_locs[vehicle_mask]
            batch_idxs = batch_idxs[vehicle_mask]

            target = torch.stack([gt_locs] * self.num_proposals, dim=2).to(dev)

            # Regression loss
            loss_prop_reg = huber_loss(proposals, target)
            loss_prop_reg = loss_prop_reg * has_preds_mask

            b_s = torch.zeros((batch_size, self.num_proposals)).to(loss_prop_reg.device)
            count = 0
            for i, batch_num_nodes_i in enumerate(graph.batch_num_nodes()):
                batch_num_nodes_i = batch_num_nodes_i.item()
                
                batch_loss_prop_reg = loss_prop_reg[count:count+batch_num_nodes_i]    
                # divide by number of agents in the scene        
                b_s[i] = torch.sum(batch_loss_prop_reg, (0, 1, 3)) / batch_num_nodes_i

                count += batch_num_nodes_i

            # sanity check
            assert batch_size == (i + 1)

            loss_prop_reg = torch.min(b_s, dim=1)[0].mean()        
        
        if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
            ### Regression Loss
            # has_preds: [N, 30]
            # res["loc_pred"]: [N, 30, 6, 2]
            has_preds_mask = has_preds.unsqueeze(-1).unsqueeze(-1)
            has_preds_mask = has_preds_mask.expand(has_preds_mask.shape[0], has_preds_mask.shape[1], self.num_joint_modes, 2).bool().to(dev)
            
            loc_pred = res["loc_pred"]
            
            if not self.proposal_header:
                if self.supervise_vehicles and self.dataset=='interaction':
                    vehicle_mask = agenttypes[:, 1].bool()
                else:
                    vehicle_mask = torch.ones(agenttypes[:, 1].shape).bool().to(dev)
    
                gt_locs = gt_locs[vehicle_mask]
                batch_idxs = batch_idxs[vehicle_mask]
            
            has_preds_mask = has_preds_mask[vehicle_mask]
            loc_pred = loc_pred[vehicle_mask]
            
            target = torch.stack([gt_locs] * self.num_joint_modes, dim=2).to(dev)

            # Regression loss
            reg_loss = huber_loss(loc_pred, target)

            # 0 out loss for the indices that don't have a ground-truth prediction.
            reg_loss = reg_loss * has_preds_mask

            b_s = torch.zeros((batch_size, self.num_joint_modes)).to(reg_loss.device)
            count = 0
            for i, batch_num_nodes_i in enumerate(graph.batch_num_nodes()):
                batch_num_nodes_i = batch_num_nodes_i.item()
                
                batch_reg_loss = reg_loss[count:count+batch_num_nodes_i]    
                # divide by number of agents in the scene        
                b_s[i] = torch.sum(batch_reg_loss, (0, 1, 3)) / batch_num_nodes_i

                count += batch_num_nodes_i

            # sanity check
            assert batch_size == (i + 1)

            loss_reg = torch.min(b_s, dim=1)[0].mean()      

        # Relation Loss
        if self.learned_relation_header:
            if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 1):
                if self.focal_loss:
                    ce_loss = FocalLoss(weight=torch.Tensor([self.weight_0, self.weight_1, self.weight_2]).to(dev), gamma=self.gamma, reduction='mean')
                else:
                    ce_loss = nn.CrossEntropyLoss(weight=torch.Tensor([self.weight_0, self.weight_1, self.weight_2]).to(dev))

                # Now compute relation cross entropy loss
                relations_preds = res["edge_logits"]
                relations_gt = ig_labels.to(relations_preds.device).long()

                loss_rel = ce_loss(relations_preds, relations_gt)     
        
        if not self.two_stage_training:
            loss = loss_reg
            
            if self.proposal_header:
                loss = loss + self.proposal_coef * loss_prop_reg

            if self.learned_relation_header:
                loss = loss + self.rel_coef * loss_rel

            loss_dict = {"total_loss": loss,
                        "loss_reg": loss_reg
                        }

            if self.proposal_header:
                loss_dict["loss_prop_reg"] = loss_prop_reg * self.proposal_coef
            
            if self.learned_relation_header:
                loss_dict["loss_rel"] = self.rel_coef * loss_rel                   

        else:
            if self.training_stage == 1:
                loss = self.rel_coef * loss_rel
                if self.proposal_header:
                    loss = loss + loss_prop_reg * self.proposal_coef
                
                loss_dict = {"total_loss": loss,
                             "loss_rel": self.rel_coef * loss_rel} 

                if self.proposal_header:
                    loss_dict["loss_prop_reg"] = loss_prop_reg * self.proposal_coef

            else:
                loss = loss_reg
                
                if self.proposal_header:
                    loss = loss + loss_prop_reg * self.proposal_coef
                
                loss_dict = {"total_loss": loss,
                             "loss_reg": loss_reg} 
                             
                if self.proposal_header:
                    loss_dict["loss_prop_reg"] = loss_prop_reg * self.proposal_coef
        
        return loss_dict

    def save_current_epoch(self, epoch, optimizer, val_best, ade_best, fde_best):
        # save best model to pt file
        path = self.log_path / "current_model_{}.pt".format(epoch)
        state = {
            'epoch': epoch,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_best': val_best, 
            'ade_best': ade_best,
            'fde_best': fde_best
            }
        torch.save(state, path)

    def save(self, epoch, optimizer, val_best, ade_best, fde_best):
        # save best model to pt file
        path = self.log_path / "best_model.pt"
        state = {
            'epoch': epoch,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_best': val_best, 
            'ade_best': ade_best,
            'fde_best': fde_best
            }
        torch.save(state, path)

    def save_relation_header(self, epoch, optimizer, val_edge_acc_best):
        # save best model to pt file
        path = self.log_path / "best_model_relation_header.pt"
        state = {
            'epoch': epoch,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_edge_acc_best': val_edge_acc_best
            }
        torch.save(state, path)

    def load_relation_header(self):
        # load best model from pt file
        path = self.log_path / "best_model_relation_header.pt"
        state = torch.load(path, map_location=dev)
        self.load_state_dict(state['state_dict'])

    def load_for_train_stage_1(self, optimizer):
        path = self.log_path / "best_model_relation_header.pt"
        state = torch.load(path, map_location=dev)
        self.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])

        return optimizer, state['epoch'] + 1, state['val_edge_acc_best']
    
    def load_for_train(self, optimizer):
        # load best model from pt file
        path = self.log_path / "best_model.pt"
        state = torch.load(path, map_location=dev)
        self.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])

        return optimizer, state['epoch'] + 1, state['val_best'], state['ade_best'], state['fde_best']

    def prepare_for_stage_2(self, pretrained_relation_header):
        # first, load model from stage 1 and set weights for stage 2
        path = self.log_path / "best_model_relation_header.pt"
        state = torch.load(path, map_location=dev)
        pretrained_relation_header.load_state_dict(state['state_dict'], strict=False) # TODO check if I missed something

        # second, freeze the weights of the network trained in stage 1
        for param in pretrained_relation_header.parameters():
            param.requires_grad = False

        self.pretrained_relation_header = pretrained_relation_header

    def load_for_eval(self):
        # load best model from pt file
        path = self.log_path / "best_model.pt"
        state = torch.load(path, map_location=dev)
        self.load_state_dict(state['state_dict'])

