import numpy as np
import pandas as pd


class scenario_template():
    '''
    This class is used a part of a larger dataset. It is used to classify the scenario
    which the dataset describes, mainly in regard to the classification of certain behaviors.
    For example, in gap acceptance scenarios, one could classify behavior as accepted/rejected,
    while other scenarios might use stuff such as left_turn/straight/right_turn.
    '''
    
    def __init__(self):
        pass
    
    
    #########################################################################################
    #########################################################################################
    ###                                                                                   ###
    ###                         Scenario dependent functions                              ###
    ###                                                                                   ###
    #########################################################################################
    #########################################################################################
        
    def give_default_classification(self = None):
        '''
        Returns the default classification of behavior (i.e., standard or expected behavior of the agents).
        This shoul dbe a key that is can be used in the dictionary provided by give_classification().
        '''
        raise AttributeError('Has to be overridden in actual scenario class')
    
    def give_classifications(self = None):
        '''
        Returns the possibility for classifying outputs. This is a dictionary, where the key 
        is the corresponding string name of the behavior, while the value is the corresponding numerical value.
        
        If this scenario allows for no classification, this will return 'None'.
        
        The second vairable returned will be the number of classifiations
        '''
        raise AttributeError('Has to be overridden in actual scenario class')
        
    def get_name(self = None):
        '''
        Return the name of the scenario, such as 'Gap acceptance'
        '''
        raise AttributeError('Has to be overridden in actual scenario class')
        
    def can_provide_general_input(self = None):
        '''
        Return either True or False, shoul dbe self explanatory
        '''
        raise AttributeError('Has to be overridden in actual scenario class')
        
    def pov_agent(self = None):
        '''
        Returns the name of the agent which can be considered the point of view,
        the agent for whose path planning the prediction is made.
        '''
        raise AttributeError('Has to be overridden in actual scenario class')
        
    def classifying_agents(self = None):
        '''
        Returns the name of the other agents, besides the pov_agent, predictions of whom 
        are necessary to classify a behavior, i.e., they have to be included in the path that is 
        provided to calculate_distance().
        '''
        raise AttributeError('Has to be overridden in actual scenario class')
    
    def calculate_safe_action(self, D_class, t_D_class, data_set, path, t, domain):
        r'''
        This function gives for each time in t the needed time for the pov-agent to 
        ensure a safe interaction.
    
        Parameters
        ----------
        D_class : pandas.Series
            A pandas series of :math:`N_{agents}` dimensions,
            where each entry is itself a numpy array of lenght :math:`|T|`, 
            and provides the distance to classification.
        t_D_class : pandas.Series
            A pandas series of :math:`N_{agents}` dimensions,
            where each entry is itself a numpy array of lenght :math:`|T|`, 
            and provides the time to classification.
    
        Returns
        -------
        t_safe_action : numpy.ndarray
            This is a :math:`|T|` dimensioanl boolean array.
        '''
        raise AttributeError('Has to be overridden in actual scenario class')

        
        
    
        
        