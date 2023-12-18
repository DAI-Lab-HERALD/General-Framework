#%% Import libraries
import pickle
import os
import numpy as np
import fitz 
import imageio
#%% Load the data

# 2D-Distributions
# Noisy Circles
np.random.seed(0)

# noisy_circles = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/noisy_circles_20000samples', 'rb'))

# Noisy Moons
noisy_moons = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/noisy_moons_20000samples', 'rb'))

# Blobs
# blobs = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/blobs_20000samples', 'rb'))

# Varied
varied = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/varied_20000samples', 'rb'))

# Anisotropic
aniso = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/aniso_20000samples', 'rb'))

# Datasets = {'Blobs': blobs, 'Aniso': aniso, 'Varied': varied, 'Two Moons': noisy_moons, 'Two Circles': noisy_circles}
Datasets = {'Aniso': aniso}

for i, name in enumerate(Datasets):
    string = './Distribution Datasets/2D-Distributions/Plots/' + name + '_reachability'
    path = os.path.dirname(string)
    file = os.path.basename(string)

    files = os.listdir(path)
    files = [f for f in files if (f[-4:] == '.pdf' and file in f)]

    Eps_files = [f for f in files if 'Eps' in f]
    Eps       = [float(f.split('=')[-1][:-4]) for f in Eps_files]

    I_Eps = np.argsort(Eps)
    Eps_images = []
    for i in I_Eps:
        load_file = path + os.sep + Eps_files[i]
        doc = fitz.open(load_file)
        pdf = doc[0]
        img = pdf.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = np.frombuffer(img.samples, dtype=np.uint8).reshape((img.h, img.w, 3))
        Eps_images.append(img)
        doc.close()

    Eps_gif_path = path + os.sep + file + '_Eps.gif'
    imageio.mimsave(Eps_gif_path, Eps_images, duration=150)
    
    Xi_files = [f for f in files if 'Xi' in f]
    Xi       = [float(f.split('=')[-1][:-4]) for f in Xi_files]

    I_Xi = np.argsort(Xi)
    Xi_images = []
    img_old = None
    for i in I_Xi:
        load_file = path + os.sep + Xi_files[i]
        doc = fitz.open(load_file)
        pdf = doc[0]
        img = pdf.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = np.frombuffer(img.samples, dtype=np.uint8).reshape((img.h, img.w, 3))

        doc.close()

        if img_old is not None:
            diff = img - img_old
            if not (diff != 0).any():
                continue

        Xi_images.append(img)
        img_old = img


    Xi_gif_path = path + os.sep + file + '_Xi.gif'
    imageio.mimsave(Xi_gif_path, Xi_images, duration=150)
    

    

    
