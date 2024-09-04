import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# Define parameters for the normal distributions (mean, standard deviation)

# Create a function to have two normals.
# Then plot the two normals and the GMM
# Then continuously have the previous GMM and the one first taken (global) and find a new GMM
# SHow that the final GMM is the same as the first global

def get_gmms_with_consecutive(global_params:list, params_1:list, weights_list:list):
    # Generate data for the normal distributions
    w1, w2 = weights_list[0], weights_list[1]
    x = np.linspace(-3, 11, 1000)
    # global_pdf = [norm.pdf(x, mean, std) for mean, std in global_params]
    pdf_list= []
    pdf_list.append([norm.pdf(x, global_params[0], math.sqrt(global_params[1])) ])
    pdf_list.append([norm.pdf(x, params_1[0], math.sqrt(params_1[1])) ])

    # all_params = []
    # Keep finding the GMMS between previous GMM and current GMM and sample data for plot
    params_list = [global_params, params_1]
    all_params = [global_params, params_1]
    for i in range(10):
        mean1, sigma1 = params_list[0][0], math.sqrt(params_list[0][1])
        mean2, sigma2 = params_list[1][0], math.sqrt(params_list[1][1])
        gmm_mean = w1*mean1 + w2*mean2
        gmm_var = w1 * (sigma1**2) + w2 * (sigma2**2) + w1*w2*(mean1 - mean2)**2
        pdf_list.append([norm.pdf(x, gmm_mean, math.sqrt(gmm_var)) ])
        if i == 0:
            params_list = [params_1, [gmm_mean, gmm_var]]
        else:
            params_list = [params_list[-1], [gmm_mean, gmm_var]]
        all_params.append(params_list[-1])
        print(params_list)
    return pdf_list, all_params

def get_gmms_with_global(global_params:list, params_1:list, weights_list:list):
    # Generate data for the normal distributions
    w1, w2 = weights_list[0], weights_list[1]
    x = np.linspace(-3, 11, 1000)
    # global_pdf = [norm.pdf(x, mean, std) for mean, std in global_params]
    pdf_list= []
    pdf_list.append([norm.pdf(x, global_params[0], math.sqrt(global_params[1])) ])
    pdf_list.append([norm.pdf(x, params_1[0], math.sqrt(params_1[1])) ])
    # all_params = []
    # Keep finding the GMMS between global and current local and sample data for plot
    params_list = [global_params, params_1]
    all_params = [global_params, params_1]
    for _ in range(10):
        mean1, sigma1 = params_list[0][0], math.sqrt(params_list[0][1])
        mean2, sigma2 = params_list[1][0], math.sqrt(params_list[1][1])
        gmm_mean = w1*mean1 + w2*mean2
        gmm_var = w1 * (sigma1**2) + w2 * (sigma2**2) + w1*w2*(mean1 - mean2)**2
        pdf_list.append([norm.pdf(x, gmm_mean, math.sqrt(gmm_var)) ])
        
        params_list = [global_params, [gmm_mean, gmm_var]]
        all_params.append(params_list[-1])
        
        print(params_list)
    return pdf_list, all_params

def plot_gmm(pdf_list:list, all_params_list:list, fig_name:str):
    # Plot the individual normal distributions
    all_params_list = [[round(p, 4) for p in params] for params in all_params_list]
    x = np.linspace(-3, 11, 1000)
    plt.figure(figsize=(10, 6))
    print(len(all_params_list))
    for i, pdf in enumerate(pdf_list, 0):
        if i == 0:
            lab = f'Global: N{all_params_list[i]}'
        elif i == 1:
            lab = f"Local 1: N{all_params_list[i]}"
        else:
            print(i)
            lab = f'GMM {i-1}: N{all_params_list[i]}'
        plt.plot(x, pdf[0], label=lab, alpha=0.5)
    
    title = fig_name.replace('.png', '') + ' using weights: ' +f'{weights_list}'
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

    plt.savefig(fig_name)

########

global_params = [0, 1]
params_1 = [2, 1]
weights_list = [0.3, 0.7]

print("Parameters from Global and local models:")
# params_list_gl, pdf_list_gl = get_gmms_with_global(global_params, params_1, weights_list)
plot_gmm(*get_gmms_with_global(global_params, params_1, weights_list), fig_name='gmm_with_global.png')

print("Parameters from consecutive locals:")
plot_gmm(*get_gmms_with_consecutive(global_params, params_1, weights_list), fig_name='gmm_consecutive_locals.png')
