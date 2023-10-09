import numpy as np
import pygame as pg
import time
import os
from experiment import Experiment


class simulation_game():
    def __init__(self):
        self.running = True
        
        # The scale between positions values and pixels
        self.scale = 5
        
        # set current testing index
        self.test_index = 0
        self.pred_steps = 0
        
        # Set visualization step for time steps
        self.time_step_repeat = 3
        
        # Set the selcted characteristics for the experiment
        self.prepare_experiment()
        
        # Load the experiment data
        [self.data_set, self.data_param, self.splitter, 
         self.Input_path, self.Output_path, 
         self.Output_A, self.Output_T_E, self.Domain] = self.experiment._get_data()
        
        
        # Get model predictions
        self.Model_name = []
        self.Output_path_pred = []
        for model_name in self.experiment.Models:
            model, output_path_pred = self.experiment._get_data_pred(self.data_set, self.splitter, model_name)
            
            self.Model_name.append(model.get_name()['print'])
            self.Output_path_pred.append(output_path_pred)        
        
        
        # Define colors
        self.Colors = {'white': (255, 255, 255),
                       'black': (  0,   0,   0),
                       'grey':  (150, 150, 150),
                       'tar':   (125, 120,  30),
                       'ego':   (120,  30, 125),
                       'v_1':   (120,  30, 125),
                       'v_2':   (120,  30, 125),
                       'v_3':   (120,  30, 125),
                       'v_4':   (240,  30, 250),}
        
        # Load car images
        self.Car_ego = pg.image.load('Car_sketch_ego.png')
        self.Car_tar = pg.image.load('Car_sketch_tar.png')
        
        
        # Define font 
        pg.font.init()
        self.font = pg.font.SysFont(None, 20)
        
        # Prepare framerate
        self.FPS = 25
        self.clock = pg.time.Clock()
        
        # Check for leaderboard
        # Considered metrics: ADE, FDE
        leader_path = 'saved_demo_results.npy'
        if os.path.isfile(leader_path):
            human_leaderboard = np.load(leader_path, allow_pickle = True)
        else:
            human_leaderboard = np.ones((len(self.Input_path), 2), np.float32) * 10000.0
        
        self.leaderboard = np.ones((len(self.Input_path), 2, 3), np.float)
        self.leaderboard[:,:,0] = human_leaderboard
        self.leaderboard = self.leaderboard.astype(object)
        
    
    
    def prepare_experiment(self):
        self.experiment = Experiment('HRI_pres')
        
        data_set = {'scenario': 'RounD_round_about', 'max_num_agents': 6, 't0_type': 'all', 'conforming_t0_types': []}

        # Select the params for the datasets to be considered
        data_param = {'dt': 0.2, 'num_timesteps_in': (15, 15), 'num_timesteps_out': (25, 25)}

        # Select the spitting methods to be considered
        splitter = {'Type': 'Cross_split', 'repetition': [0], 'test_part': 0.2}

        # Select the models to be trained
        Models = ['flomo_schoeller', 'trajflow_meszaros', 'trajflow_meszaros_2',
                  'trajectron_salzmann_old', 'agent_yuan']
        Models = ['flomo_schoeller', 'trajflow_meszaros', 'trajflow_meszaros_2',
                  'trajectron_salzmann_old']

        # Select the metrics to be used
        Metrics = ['minADE20_indep', 'minFDE20_indep', 'ADE_ML_indep', 'FDE_ML_indep',
                        'Oracle_indep', 'KDE_NLL_indep', 'ECE_traj_indep',
                        'ECE_class', 'AUC_ROC']

        self.experiment.set_modules([data_set], [data_param], [splitter], Models, Metrics)

        # Set the number of different trajectories to be predicted by trajectory prediction models.
        num_samples_path_pred = 100

        # Deciding wether to enforce start times (or allow later predictions if not enough input data is available)
        enforce_prediction_times = True

        # determine if the upper bound for n_O should be enforced, or if prediction can be made without
        # underlying output data (might cause training problems)
        enforce_num_timesteps_out = False

        # Determine if the useless prediction (i.e, prediction you cannot act anymore)
        # should be exclude from the dataset
        exclude_post_crit = True

        # Decide wether missing position in trajectory data can be extrapolated
        allow_extrapolation = True

        # Use all available agents for predictions
        agents_to_predict = 'predefined'

        # Determine if allready existing results shoul dbe overwritten, or if not, be used instead
        overwrite_results = True

        # Determine if the model should be evaluated on the training set as well
        evaluate_on_train_set = True

        # Select method used for transformation function to path predictions
        model_for_path_transform = 'trajectron_salzmann_old'

        self.experiment.set_parameters(model_for_path_transform, num_samples_path_pred, 
                                       enforce_num_timesteps_out, enforce_prediction_times, 
                                       exclude_post_crit, allow_extrapolation, 
                                       agents_to_predict, overwrite_results, evaluate_on_train_set)
        
        
    def run_game(self):
        # Prepare window
        self.width = 1500
        self.height = 1000
        self.window = pg.display.set_mode((self.width, self.height), pg.RESIZABLE)
        self.cord_origin_pos = np.array([750, 515])
        
        # Set name of window
        pg.display.set_caption("HRI - Demonstration")
        
        self.current_FPS = self.FPS
        
        
        # Check slower refreshers
        time_spent = 0 
        
        # Load first problem
        self.set_current_problem()
        
        # Preset some checkmarks
        self.dragging = False
        self.started_pressing = False
        self.plot_future_path = False
        
        while self.running:
            # Draw the screen
            self.draw_screen()
            
            if self.plot_future_path:
                self.draw_result_screen()
            
            # Reset flag checks
            self.stopped_pressing = False
            
            # updata screen
            pg.display.update()
            
            # Get mous positions
            self.mouse_pos = pg.mouse.get_pos()
            
            # Check for events
            # - selected dataset
            # - check if next problem from test set is selected
            # - iterate through it if possible
            event_queue = pg.event.get()
            for event in event_queue:
                # check for quit event
                if event.type == pg.QUIT:
                    # end current loop
                    self.running = False
                
                # Check for window resizing
                if event.type == pg.VIDEORESIZE:
                    self.resize(event)
                    
                if event.type == pg.MOUSEWHEEL:
                    if pg.key.get_mods() & pg.KMOD_CTRL:
                        if self.mouse_inbound():
                            self.zooming(event.y)
                            
                if event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.mouse_inbound():
                            if pg.key.get_mods() & pg.KMOD_CTRL:
                                self.dragging = True
                                self.mouse_pos_old = self.mouse_pos
                            else:
                                self.dragging = False
                                self.started_pressing = True
                            
                if event.type == pg.MOUSEBUTTONUP:
                    self.dragging = False
                    if self.started_pressing:
                        self.stopped_pressing = True
                        
                # Check for Key presses
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_z:
                        if pg.key.get_mods() & pg.KMOD_CTRL:
                            if self.pred_steps > 0:
                                if not self.plot_future_path:
                                    self.delete_pred_point()
                                
                    elif event.key == pg.K_RETURN:
                        if not (pg.key.get_mods() & pg.KMOD_CTRL):
                            if self.plot_future_path:
                                self.set_current_problem()
                            else:
                                if self.pred_steps >= self.num_timesteps_out:
                                    self.plot_future_path = True
                                    self.evaluate_predictions()
                                
                    elif event.key == pg.K_r:
                        if pg.key.get_mods() & pg.KMOD_CTRL:
                            if self.plot_future_path:
                                self.set_current_problem(0)
            
            if self.started_pressing and not self.mouse_inbound():
                self.stopped_pressing = True
            
            if not (pg.key.get_mods() and 
                    pg.KMOD_CTRL and
                    self.mouse_inbound()):
                self.dragging = False
            
                
            if self.dragging:
                self.update_center()
                self.mouse_pos_old = self.mouse_pos
            
            if self.started_pressing:
                self.add_pred_point()
                if self.stopped_pressing:
                    self.started_pressing = False
            
            # Wait until next frame update
            time_spent_frame = self.clock.tick(self.FPS)
            current_FPS = 1000 / time_spent_frame
            
            time_spent += time_spent_frame
            if time_spent > 1000:
                time_spent = 0 
                self.current_FPS = current_FPS
            
        # save_leaderboard
        human_leaderboard = self.leaderboard[:,:,0].astype(np.float32)
        np.save('saved_demo_results.npy', human_leaderboard)
            
        pg.quit()
    
    
    def mouse_inbound(self):
        return (5 < self.mouse_pos[0] < self.width) and (5 < self.mouse_pos[1] < self.height)
    
    #%% Standard refreshers
    
    def update_center(self):
        delta_x = self.mouse_pos[0] - self.mouse_pos_old[0]
        delta_y = self.mouse_pos[1] - self.mouse_pos_old[1]
        
        self.cord_origin_pos += np.array([delta_x, delta_y]).astype(int)
            
    
    def draw_result_screen(self):
        
        pg.display.update()
        
        columns = [('Metric \n' + 
                    'ADE (in m): \n' + 
                    'FDE (in m): ')]
        
        
        columns.append('You\n' + 
                       '{:0.2f}'.format(self.human_metric[0]).zfill(5) + '\n' +
                       '{:0.2f}'.format(self.human_metric[1]).zfill(5))
        
        columns.append('Best\n' + 
                       '{:0.2f}'.format(self.leaderboard[self.test_index - 1, 0, 0]).zfill(5) + '\n'  +
                       '{:0.2f}'.format(self.leaderboard[self.test_index - 1, 1, 0]).zfill(5))
           
        columns.append('Trained Model (median)\n' + 
                       '{:0.2f}'.format(self.leaderboard[self.test_index - 1, 0, 1]).zfill(5) +               
                       ' (' + self.leaderboard[self.test_index - 1, 0, 2] + ')\n' +
                       '{:0.2f}'.format(self.leaderboard[self.test_index - 1, 1, 1]).zfill(5) +               
                       ' (' + self.leaderboard[self.test_index - 1, 1, 2] + ')')
        
        
        # get width
        width = 10
        for column in columns:
            max_width = 0
            for row, column_str in enumerate(column.split('\n')):
                column_text = self.font.render(column_str, True, self.Colors['white'])
                max_width = max(max_width, column_text.get_width())
            width += 10 + max_width
        
        x0 = int(0.5 * (self.width - 300))
        y0 = int(0.5 * (self.height - 200))
        x = x0 + 10
        y = y0 + 10
        
        result_box = pg.Rect(x0, y0, width, 90)
        transparent_surface = pg.Surface(result_box.size, pg.SRCALPHA)
        pg.draw.rect(transparent_surface, (150, 150,150,100),  (0, 0, *result_box.size))
        self.window.blit(transparent_surface, result_box.topleft)
        pg.draw.rect(self.window, self.Colors['white'], result_box, 2, 0)
        
        for column in columns:
            max_width = 0
            for row, column_str in enumerate(column.split('\n')):
                column_text = self.font.render(column_str, True, self.Colors['white'])
                y_row = y + row * (25) + 2 * (row > 0)
                self.window.blit(column_text, (x, y_row))
                max_width = max(max_width, column_text.get_width())
            x += 10 + max_width
        
    
    
    def draw_screen(self):
        self.window.fill(self.Colors['black'])
        # Draw image
        image_height, image_width = self.img.shape[:2]
        img_int = (self.img * 255).astype(np.uint8)
        img_surf = pg.surfarray.make_surface(img_int.transpose(1,0,2))
        
        # Apply scale
        img_m_per_px = self.data_set.get_Target_MeterPerPx(self.domain)
        img_surf = pg.transform.scale_by(img_surf, self.scale * img_m_per_px)
        
        
        self.window.blit(img_surf, (self.cord_origin_pos[0] - img_surf.get_width() * 0.5,
                                    self.cord_origin_pos[1] - img_surf.get_height() * 0.5))
        
        # Draw past trajectories
        for i, agent in enumerate(self.agents):
            self.draw_line_with_crosses(self.input[i], self.Colors[agent], mode = 'input_true')
            
        
        # Draw future trajectory
        useful_preds = np.isfinite(self.human_predictions).all(1)
        if useful_preds.sum() > 0:
            pred_traj = self.human_predictions[useful_preds]
            
            
            # Combine with last endpoint
            pred_traj = np.concatenate([self.input[self.i_tar,-1], pred_traj], axis = 0)
            self.draw_line_with_crosses(pred_traj, self.Colors['tar'], mode = 'output_pred')
            
        if self.plot_future_path:
            for i, agent in enumerate(self.agents):
                fut_path = np.concatenate((self.input[i,[-1]], self.output[i]), axis = 0)
                self.draw_line_with_crosses(fut_path, self.Colors[agent], mode = 'output_true')
            
        # Draw Info box
        info_box = pg.Rect(5, 5, self.width - 10, 30)
        info_box_back = pg.Rect(0, 0, self.width, 35)
        pg.draw.rect(self.window, self.Colors['black'], info_box_back, 0, 0)
        pg.draw.rect(self.window, self.Colors['grey'], info_box, 0, 0)
        pg.draw.rect(self.window, self.Colors['white'], info_box, 2, 0)
        
        FPS_text = self.font.render('{:5.1f} FPS'.format(self.current_FPS), True, self.Colors['black'])
        FPS_text_width, FPS_text_height = FPS_text.get_width(), FPS_text.get_height()
        self.window.blit(FPS_text, (70 - FPS_text_width, 20 - 0.5 * FPS_text_height))
        
        dt = self.data_set.dt * self.time_step_repeat
        
        data_str = ('Test_sample {:4.0f}/{:4.0f}'.format(self.test_index, len(self.Input_path)) +
                    ' - predicted {:2.0f}/{:2.0f} future timestpes ({:0.2f} s)'.format(self.pred_steps, self.num_timesteps_out, dt))
        data_text = self.font.render(data_str, True, self.Colors['black'])
        self.window.blit(data_text, (90, 20 - 0.5 * FPS_text_height))
        
        # Draw commands:
        if self.pred_steps > 0:
            if self.plot_future_path:
                command_str = '"CTRL + R" => Return to first sample & "ENTER" => Go to next sample'
            else:
                command_str = '"CTRL + Z" => delete last timestep'
                if self.pred_steps >= self.num_timesteps_out:
                    command_str_new = '"ENTER" => evaluate prediction'
                    command_str = command_str_new + ' & ' + command_str
        
            command_text = self.font.render(command_str, True, self.Colors['black'])
            command_text_width = command_text.get_width()
            self.window.blit(command_text, (self.width - 15 - command_text_width, 20 - 0.5 * FPS_text_height))
        

        
    def draw_line_with_crosses(self, traj, color, mode):
        # plot cars if possible
        if mode == 'input_true':
            # check for vehicle 
            if color == self.Colors['tar']:
                car = self.Car_tar
            elif color == self.Colors['ego']:
                car = self.Car_ego
            else: 
                car = None
             
            if car is not None:
                # scale car
                car_scale = 5 * self.scale / 803
                car_scaled = pg.transform.scale_by(car, car_scale)
                   
                # rotate car
                angle_car = np.angle((traj[-1,0] - traj[-2,0]) + 1j * (traj[-1,1] - traj[-2,1]))
                angle_car_deg = angle_car * 180 / np.pi - 90 # Original orientation is 90 degree
                
                car_sr = pg.transform.rotate(car_scaled, angle_car_deg)
                
                # gt print pos
                car_x = self.cord_origin_pos[0] + (traj[-1,0] * self.scale) - 0.5 * car_sr.get_width()
                car_y = self.cord_origin_pos[1] - (traj[-1,1] * self.scale) - 0.5 * car_sr.get_height()
                
                self.window.blit(car_sr, (car_x, car_y))
                
                
        Timesteps = np.arange(len(traj))
        if mode == 'input_true':
            Used = np.mod(Timesteps.max() - Timesteps, self.time_step_repeat) == 0 
            scale = np.linspace(0.4, 1.0, Used.sum())
        elif mode == 'output_true':
            Used = np.mod(Timesteps, self.time_step_repeat) == 0 
            scale = np.linspace(1.0, 1.0, Used.sum())
            scale[0] = 0.0
        elif mode == 'output_pred':
            Used = np.ones(len(traj), bool)
            scale = np.linspace(1.0, 1.0, Used.sum())
            scale[0] = 0.0
            
            # overwrite color
            color = tuple((255 - 0.5 * (255 - np.array(color))).astype(int))
        else:
            raise KeyError('Julian made a coding mistake.')
        traj = traj[Used]
        
        x_points_in = traj[:,0]
        y_points_in = traj[:,1]
        
        x_points_in = self.cord_origin_pos[0] + (x_points_in * self.scale)
        y_points_in = self.cord_origin_pos[1] - (y_points_in * self.scale)
        
        traj_scaled = np.stack([x_points_in, y_points_in], axis = 1)
            
        # Get heading of line segments
        delta = traj_scaled[1:] - traj_scaled[:-1]
        angle = np.angle(delta[:,0] - 1j * delta[:,1])
        
        
        angle = np.concatenate([angle[[0]], 0.5 * (angle[1:] + angle[:-1]), angle[[-1]]], axis = 0)
        
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]]).transpose(2,0,1)
        # R.shape: num_points x dims x dims
        # fade into past:
        R = R * scale[:,np.newaxis,np.newaxis]
        
        
        
        
        cross_px = 5 ** 0.5 * (self.scale ** 0.5)
        lines = np.array([[[-cross_px, cross_px],
                           [0,  0],
                           [cross_px, -cross_px]],
                          [[-cross_px,-cross_px],
                           [0,0],
                           [cross_px, cross_px]]]).transpose(0,2,1)
        
        # lines.shape: number_lines x dims x points_per_line
        
        Lines = np.dot(R, lines) # shape: num_points x dims x number_lines x points_per_line
        Lines[:, 1] *= -1
        
        # transpose lines
        Lines = Lines + traj_scaled[:,:,np.newaxis,np.newaxis]
        
        points = tuple(map(tuple, traj_scaled.astype(int)))
        pg.draw.lines(self.window, color, False, points, 2)
        
        # draw crosses
        for i_point in range(Lines.shape[0]):
            for i_line in range(Lines.shape[2]):
                L = tuple(map(tuple, Lines[i_point, :, i_line].T.astype(int)))
                pg.draw.lines(self.window, color, False, L, 2)
    
    
    
    #%% Event functions 
    
    def add_pred_point(self):
        if self.pred_steps >= self.num_timesteps_out:
            return
        # transform current mouse point to meter position
        dx_scaled = self.mouse_pos[0] - self.cord_origin_pos[0]
        dy_scaled = self.cord_origin_pos[1] - self.mouse_pos[1]
        
        pos_scaled = np.array([dx_scaled, dy_scaled])
        self.human_predictions[self.pred_steps] = pos_scaled / self.scale
        
        # If final, lock point in
        if self.stopped_pressing:
            self.pred_steps += 1
        
    
    def delete_pred_point(self):
        self.pred_steps -= 1
        self.human_predictions[self.pred_steps] = np.nan
        
    
    def zooming(self, factor):
        orig_old = self.cord_origin_pos
        scale_old = self.scale
        
        scale_adjust = (40 + max(min(factor, 10), -10)) / 40
        
        dx_old = orig_old[0] - self.mouse_pos[0]
        dy_old = orig_old[1] - self.mouse_pos[1]
        
        dx_new = dx_old * scale_adjust
        dy_new = dy_old * scale_adjust
        
        scale_new = scale_old * scale_adjust
        
        if 2 < scale_new < 50: 
            self.scale = scale_old * scale_adjust
            self.cord_origin_pos = np.array([self.mouse_pos[0] + dx_new, 
                                             self.mouse_pos[1] + dy_new]).astype(int)
        
    
    def set_current_problem(self, test_index = None):
        # Reset clear index             
        if test_index is None:                                        
            self.test_index = np.mod(self.test_index + 1, len(self.Input_path))
        else:
            assert isinstance(test_index, int), "Desired sample should be an integer."
            self.test_index = 1
        [self.output, self.input, self.agents, 
         self.min_extent, self.max_extent, 
         _, _, self.img, self.domain] = self.experiment._get_data_sample(self.test_index - 1, self.data_set, 
                                                                         self.Input_path, self.Output_path, 
                                                                         self.Output_A, self.Output_T_E,  self.Domain)
        
        # Get target agent id
        self.i_tar = self.agents == 'tar'
        
        self.pred_steps = 0
        
        
        self.num_timesteps_out = int(np.floor(self.output.shape[1] / self.time_step_repeat))
        
        self.human_predictions = np.ones((self.num_timesteps_out, 2)) * np.nan
        
        self.model_predictions = np.ones((len(self.Output_path_pred), 
                                          self.data_set.num_samples_path_pred,
                                          *self.output.shape[1:]), np.float32)
        for i, output_pred in enumerate(self.Output_path_pred):
            self.model_predictions[i] = self.Output_path_pred[i].iloc[self.test_index - 1]['tar']
        
        # Reset flags 
        self.dragging = False
        self.started_pressing = False
        self.plot_future_path = False
        
    
    
    def resize(self, resize_event):
        self.width = resize_event.w
        self.height = resize_event.h 
        
        failure = False
    
        if self.width < 500:
            self.width = 500
            failure = True
            
        if self.height < 300:
            self.height = 300
            failure = True
        
        if failure:
            self.window = pg.display.set_mode((self.width, self.height), pg.RESIZABLE)
            
        
        
    def evaluate_predictions(self):
        assert self.plot_future_path
        
        Timesteps = np.arange(1, self.output.shape[-2] + 1)
        Used = np.mod(Timesteps, self.time_step_repeat) == 0 
        
        op = self.output[np.newaxis, np.newaxis, self.i_tar, Used]
        
        oph = self.human_predictions[np.newaxis, np.newaxis]
        opm = self.model_predictions[...,Used,:]
        
        diff_human = np.sqrt(((op - oph) ** 2).sum(-1))
        diff_model = np.sqrt(((op - opm) ** 2).sum(-1))
        
        ade_model = np.median(diff_model.mean(-1), 1)
        fde_model = np.median(diff_model[...,-1], 1)
        
        ade_human = diff_human.mean()
        fde_human = diff_human[0,0,-1]
        
        # set ADE leaderboards
        self.leaderboard[self.test_index - 1, 0, 0] = min(self.leaderboard[self.test_index - 1, 0, 0], ade_human)
        ade_model_best = np.argmin(ade_model)
        self.leaderboard[self.test_index - 1, 0, 1] = ade_model[ade_model_best]
        self.leaderboard[self.test_index - 1, 0, 2] = self.Model_name[ade_model_best]
        
        # set FDE leaderboards
        self.leaderboard[self.test_index - 1, 1, 0] = min(self.leaderboard[self.test_index - 1, 1, 0], fde_human)
        fde_model_best = np.argmin(fde_model)
        self.leaderboard[self.test_index - 1, 1, 1] = fde_model[fde_model_best]
        self.leaderboard[self.test_index - 1, 1, 2] = self.Model_name[fde_model_best]
        
        self.human_metric = [ade_human, fde_human]
        


if __name__ == "__main__":
    simulation = simulation_game()
    simulation.run_game()