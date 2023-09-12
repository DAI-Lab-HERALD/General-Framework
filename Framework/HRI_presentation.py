import numpy as np
import pygame as pg
import time




class simulation_game():
    def __init__(self):
        self.running = True
        self.FPS = 10
        
        self.clock = pg.time.Clock()
        
        self.width = 1000
        self.height = 500
        
        self.window = pg.display.set_mode((self.width, self.height), pg.RESIZABLE)
        
        
    def run_game(self):
        self.start_screen()
        
        while self.running:
            # Wait until next frame update
            self.clock.tick(self.FPS) 
            
            self.update_screen()
            
        
        pg.quit()
        
        
        
        
        



    

    def start_screen(self):
        pass
    
    
    
    def update_screen(self):
        pass
    
    
    
    
    

# Start the game
def main():
    # Get clock for counting game
    simulation = simulation_game()
    simulation.run_game()
    










# Search a test case scenario



# Aks the user to predict future trajectory of agent


# Compare ageinst true trajectory





# Compare against models








# Select the available model from drop down menu (Search through results if necessary)








if __name__ == "__main__":
    main()