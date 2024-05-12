import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_losses_and_metrics(file_path, output_dir):
    # Count the number of lines in the log file
    with open(file_path, 'r') as file:
        num_lines = sum(1 for line in file)
    print(f"Number of lines in the log file: {num_lines}")

    # Extract training losses
    iterations = []
    l_g_pix = []
    l_g_percep = []
    l_g_gan = []
    l_g_color = []
    l_d = []
    real_score = []
    fake_score = []

    with open(file_path, 'r') as file:
        for line in file:
            if 'INFO: [train..]' in line:
                parts = line.split()
                if 'iter:' in parts:
                    iter_num_str = parts[parts.index('iter:') + 1].strip(',')
                    iter_num = int(iter_num_str.replace(',', ''))
                    if iter_num % 1000 == 0:
                        iterations.append(iter_num)
                        l_g_pix_match = re.search(r'l_g_pix: (\d+\.\d+)', line)
                        l_g_percep_match = re.search(r'l_g_percep: (\d+\.\d+)', line)
                        l_g_gan_match = re.search(r'l_g_gan: (\d+\.\d+)', line)
                        l_g_color_match = re.search(r'l_g_color: (\d+\.\d+)', line)
                        l_d_match = re.search(r'l_d: (\d+\.\d+)', line)
                        real_score_match = re.search(r'real_score: (-?\d+\.\d+)', line)
                        fake_score_match = re.search(r'fake_score: (-?\d+\.\d+)', line)
                        
                        l_g_pix.append(float(l_g_pix_match.group(1)) if l_g_pix_match else np.nan)
                        l_g_percep.append(float(l_g_percep_match.group(1)) if l_g_percep_match else np.nan)
                        l_g_gan.append(float(l_g_gan_match.group(1)) if l_g_gan_match else np.nan)
                        l_g_color.append(float(l_g_color_match.group(1)) if l_g_color_match else np.nan)
                        l_d.append(float(l_d_match.group(1)) if l_d_match else np.nan)
                        real_score.append(float(real_score_match.group(1)) if real_score_match else np.nan)
                        fake_score.append(float(fake_score_match.group(1)) if fake_score_match else np.nan)
                        
                        
    val_iterations = []
    val_fid = []
    val_cf = []

    with open(file_path, 'r') as file:
    	for line in file:
    		if 'INFO: Validation' in line:
    			val_iter = re.search(r'iter: (\d+)', line)
    			if val_iter:
				    fid_match = re.search(r'fid: (\d+\.\d+)', line)
				    cf_match = re.search(r'cf: (\d+\.\d+)', line)

				    if fid_match:
					val_fid.append(float(fid_match.group(1)))
				    else:
					val_fid.append(np.nan)

				    if cf_match:
					val_cf.append(float(cf_match.group(1)))
				    else:
					val_cf.append(np.nan)

				    val_iterations.append(iterations[-1])

    # Create plots
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(iterations, l_g_pix, label='L_G_Pix')
    ax1.plot(iterations, l_g_percep, label='L_G_Percep')
    ax1.plot(iterations, l_g_gan, label='L_G_GAN')
    ax1.plot(iterations, l_g_color, label='L_G_Color')
    ax1.plot(iterations, l_d, label='L_D')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses')
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(val_iterations, val_fid, label='FID')
    ax2.plot(val_iterations, val_cf, label='CF')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Metric')
    ax2.set_title('Validation Metrics')
    ax2.legend()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(iterations, real_score, label='Real Score')
    ax3.plot(iterations, fake_score, label='Fake Score')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Score')
    ax3.set_title('Real and Fake Scores')
    ax3.legend()

    # Save the plot images
    os.makedirs(output_dir, exist_ok=True)
    fig1.savefig(os.path.join(output_dir, 'training_losses.png'))
    fig2.savefig(os.path.join(output_dir, 'validation_metrics.png'))
    fig3.savefig(os.path.join(output_dir, 'real_fake_scores.png'))

    plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training and validation losses from a log file.')
    parser.add_argument('file_path', type=str, help='Path to the log file')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save the plot images')
    args = parser.parse_args()

    plot_losses_and_metrics(args.file_path, args.output_dir)
