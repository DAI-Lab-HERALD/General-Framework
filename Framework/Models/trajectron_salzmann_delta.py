import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import random
from trajectron_salzmann_unicycle import trajectron_salzmann_unicycle
from Trajectron.trajec_model.trajectron import Trajectron
from Trajectron.trajec_model.model_registrar import ModelRegistrar
import json
import os

class trajectron_salzmann_delta(trajectron_salzmann_unicycle):
    '''
    This is the updated version of Trajectron++, a single agent prediction model
    that is mainly based on LSTM cells. In its decoder, it just uses distances
    as its control inputs.
    
    The code was taken from https://github.com/NVlabs/adaptive-prediction/tree/main/src/trajectron
    and the model is published under the following citation:
        
    Ivanovic, B., Harrison, J., & Pavone, M. (2023, May). Expanding the deployment envelope 
    of behavior prediction via adaptive meta-learning. In 2023 IEEE International Conference 
    on Robotics and Automation (ICRA) (pp. 7786-7793). IEEE.
    '''
    def setup_method(self, seed = 0):
        # set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Required attributes of the model
        self.min_t_O_train = 3
        self.max_t_O_train = 100
        self.predict_single_agent = True
        self.can_use_map = True
        # If self.can_use_map = True, the following is also required
        self.target_width = 175
        self.target_height = 100
        self.grayscale = False
        
        config_path = os.sep.join(os.path.dirname(self.model_file).split(os.sep)[:-3])
        config_path += os.sep + 'Models' + os.sep + 'Trajectron' + os.sep + 'config' + os.sep
        
        if (self.provide_all_included_agent_types() == 'P').all():
            config_file = config_path + 'pedestrian.json' 
            with open(config_file) as json_file:
                hyperparams = json.load(json_file)
        else:
            config_file = config_path + 'nuScenes_state_delta.json' 
            with open(config_file) as json_file:
                hyperparams = json.load(json_file)
            
        hyperparams["dec_final_dim"]                 = 32
        
        hyperparams["map_encoding"]                  = self.can_use_map
        hyperparams["incl_robot_node"]               = False
        
        hyperparams["edge_encoding"]                 = True
        hyperparams["edge_influence_combine_method"] = "attention"
        hyperparams["edge_state_combine_method"]     = "sum"
        hyperparams["adaptive"]                      = False
        hyperparams["dynamic_edges"]                 = "yes"
        hyperparams["edge_addition_filter"]          = [0.25, 0.5, 0.75, 1.0]
        
        hyperparams["single_mode_multi_sample"]      = False
        hyperparams["single_mode_multi_sample_num"]  = 50
        
        self.std_pos_ped = 1
        self.std_vel_ped = 2
        self.std_acc_ped = 1
        self.std_pos_veh = 80
        self.std_vel_veh = 15
        self.std_acc_veh = 4
        
        # Prepare models
        model_registrar = ModelRegistrar(None, self.device)
        self.trajectron = Trajectron(model_registrar, hyperparams, None, self.device)
        self.trajectron.set_environment()
        self.trajectron.set_annealing_params()
        
    
    def get_name(self = None):
        names = {'print': 'Trajectron ++ (Dynamic model: Delta)',
                 'file': 'traject_SD',
                 'latex': r'\emph{T++}'}
        return names