import numpy as np
import torch
from commotions.commotions_expanded_model import commotions_template, Commotions_nn, Parameters
from model_template import model_template

    
class commotions_markkula(model_template, commotions_template):
    '''
    This is a model for predicting human behavior in gap acceptance scenarios,
    which is based on cognitive theory and includes concepts such as noisy
    perception, evidence accumulation and theory of mind in a multi agent setting.
    
    In **self.model_kwargs['loss_type']**, the value 'Time_MSE' uses the original loss 
    function :math:`\mathcal{L}_1` from the paper cited below, while 'ADE' uses the
    loss function :math:`\mathcal{L}_2`.
    
    The code is taken from https://github.com/julianschumann/Commotions-model-evaluation
    and the the model is published under the following citation:
        
    Schumann, J. F., Srinivasan, A. R., Kober, J., Markkula, G., & Zgonnikov, A. (2023). 
    Using Models Based on Cognitive Theory to Predict Human Behavior in Traffic: A Case 
    Study. arXiv preprint arXiv:2305.15187.
    '''

    def define_default_kwargs(self):
        if not('loss_type' in self.model_kwargs.keys()):
            self.model_kwargs['loss_type'] = 'Time_MSE'
        assert self.model_kwargs['loss_type'] in ['Time_MSE', 'ADE']

    def setup_method(self):
        self.define_default_kwargs()
        # Required attributes of the model
        self.min_t_O_train = 2
        self.max_t_O_train = 200
        self.can_use_map = False
        self.can_use_graph = False

        # set model settings
        self.adjust_free_speeds = False
        self.vehicle_acc_ctrl   = False
        self.const_accs         = [0, None]
        self.train_loss_type    = self.model_kwargs['loss_type'] 
        
        # prepare torch
        self.prepare_gpu()
        
        # Define parameters
        self.fixed_params = Parameters()
        self.fixed_params.H_e = 1.5 # m; height over ground of the eyes of an observed doing oSNv perception
        self.fixed_params.min_beh_prob = 0.0 # Min probability of behavior needed for it to be considerd at all
        self.fixed_params.ctrl_deltas = torch.tensor([-2, -1, -0.5, 0, 0.5, 1, 2], dtype = torch.float32, device = self.device) # available speed/acc change actions, magnitudes in m/s or m/s^2 dep on agent type
        self.fixed_params.T_acc_regain_spd = 10
        self.fixed_params.FREE_SPEED_PED = 1.5
        self.fixed_params.FREE_SPEED_VEH = 10
        
        # Initialize model
        self.commotions_model = Commotions_nn(self.device, self.num_samples_path_pred, 
                                              self.fixed_params, self.t_e_quantile)
    
    
    def train_method(self):
        n = 5
        # Trainable parameters
        self.Beta_V               = np.logspace(0, np.log10(200), n)
        self.DDeltaV_th_rel       = np.logspace(-3, -1, n)
        self.TT                   = np.linspace(0.1, 0.5, n)
        self.TT_delta             = np.logspace(1, 2, n)
        self.Tau_theta            = np.logspace(np.log10(0.005), np.log10(np.pi * 0.5), n)
        self.Sigma_xdot           = np.logspace(-2, 0, n) #TOP5
        self.DDeltaT              = np.linspace(0.21, 1, n) #TOP5 
        self.TT_s                 = np.linspace(0.01, 2, n)
        self.DD_s                 = np.linspace(0.01, 5, n) #TOP5
        self.VV_0_rel             = np.logspace(0, 2, n)
        self.K_da                 = np.logspace(-1, 1, n)
        self.Kalman_multi_pos     = np.logspace(-1, 1, n)
        self.Kalman_multi_speed   = np.logspace(-1, 1, n)
        self.Free_speed_multi_ego = np.logspace(np.log10(0.5), np.log10(2), n)
        self.Free_speed_multi_tar = np.logspace(np.log10(0.5), np.log10(2), n) #TOP5
        
        Params, Loss = self.BO_EI(iterations = 5) # TODO: Reset to 150
        
        self.param_best = Params[np.argmin(Loss), :]
        
        self.weights_saved = [self.param_best]
        
        
    def load_method(self):
        [self.param_best] = self.weights_saved
        
        
    def predict_method(self):
        [Output_A_pred, Output_T_E_pred] = self.extract_predictions()
        self.save_predicted_classifications(['accepted', 'rejected'], Output_A_pred, Output_T_E_pred)
    
        
    def check_trainability_method(self):
        if self.data_set.scenario_name != 'Gap acceptance problem':
            return "this model is only valid for gap acceptance scenarios."
        
        if not self.general_input_available:
            return " there is no generalized input data available."
        # If data is okay
        return None
     
        
    def get_output_type(self = None):
        # Logit model only produces class outputs
        return 'class_and_time'
    
    
    def get_name(self = None):
        self.define_default_kwargs()

        if self.model_kwargs['loss_type'] == 'Time_MSE':
            number = '1'
        else:
            number = '2'

        names = {'print': 'Commotions (' + self.model_kwargs['loss_type'] + ' loss)',
                 'file': 'commotion' + number,
                 'latex': r'$\text{\emph{CM}}_{\mathcal{L}_' + number + r'}$'}
        return names
        
    def save_params_in_csv(self = None):
        return True
    
    def requires_torch_gpu(self = None):
        return True
        
    def provides_epoch_loss(self = None):
        return True