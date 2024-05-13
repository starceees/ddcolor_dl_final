import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_losses_and_metrics(log_dir, output_dir):
    training_losses = {
        'iterations': [],
        'l_g_pix': [],
        'l_g_percep': [],
        'l_g_gan': [],
        'l_g_color': [],
        'l_d': [],
        'real_score': [],
        'fake_score': []
    }

    validation_metrics = {
        'iterations': [],
        'fid': [],
        'cf': []
    }

    best_fid = float('inf')
    best_fid_iter = None
    best_cf = 0
    best_cf_iter = None

    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                for line in file:
                    if 'INFO: [train..]' in line:
                        training_data = extract_training_data(line)
                        if training_data:
                            for key, value in training_data.items():
                                training_losses[key].append(value)

                    elif 'INFO: Validation ImageNet' in line:
                        validation_data = extract_validation_data(file, training_losses['iterations'][-1])
                        if validation_data:
                            validation_metrics['iterations'].append(validation_data['iteration'])
                            validation_metrics['fid'].append(validation_data['fid'])
                            validation_metrics['cf'].append(validation_data['cf'])

                            # Update best FID
                            if validation_data['fid'] < best_fid:
                                best_fid = validation_data['fid']
                                best_fid_iter = validation_data['iteration']

                            # Update best CF
                            if validation_data['cf'] > best_cf:
                                best_cf = validation_data['cf']
                                best_cf_iter = validation_data['iteration']

    # Print the best FID and CF values with their corresponding iterations
    print(f"Best FID: {best_fid} at iteration {best_fid_iter}")
    print(f"Best CF: {best_cf} at iteration {best_cf_iter}")

    # Create plots
    plot_training_losses(training_losses, output_dir)
    plot_validation_metrics(validation_metrics, output_dir)
    plot_real_fake_scores(training_losses, output_dir)

def extract_training_data(line):
    data = {}
    parts = line.split()
    if 'iter:' in parts:
        iter_num_str = parts[parts.index('iter:') + 1].strip(',')
        iter_num = int(iter_num_str.replace(',', ''))
        if iter_num % 1000 == 0:
            data['iterations'] = iter_num
            data['l_g_pix'] = extract_value(line, r'l_g_pix: (\d+\.\d+)')
            data['l_g_percep'] = extract_value(line, r'l_g_percep: (\d+\.\d+)')
            data['l_g_gan'] = extract_value(line, r'l_g_gan: (\d+\.\d+)')
            data['l_g_color'] = extract_value(line, r'l_g_color: (\d+\.\d+)')
            data['l_d'] = extract_value(line, r'l_d: (\d+\.\d+)')
            data['real_score'] = extract_value(line, r'real_score: (-?\d+\.\d+)')
            data['fake_score'] = extract_value(line, r'fake_score: (-?\d+\.\d+)')
    return data

def extract_validation_data(file, current_iter):
    data = {}
    fid_line = next(file)
    cf_line = next(file)

    fid_match = re.search(r'fid: (\d+\.\d+)', fid_line)
    cf_match = re.search(r'cf: (\d+\.\d+)', cf_line)

    if fid_match and cf_match:
        data['iteration'] = current_iter
        data['fid'] = float(fid_match.group(1))
        data['cf'] = float(cf_match.group(1))
    return data

def extract_value(line, pattern):
    match = re.search(pattern, line)
    return float(match.group(1)) if match else np.nan

def plot_training_losses(training_losses, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, values in training_losses.items():
        if key != 'iterations':
            ax.plot(training_losses['iterations'], values, label=key)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'training_losses.png'))
    plt.close(fig)

def plot_validation_metrics(validation_metrics, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(validation_metrics['iterations'], validation_metrics['fid'], label='FID')
    ax.plot(validation_metrics['iterations'], validation_metrics['cf'], label='CF')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Metric')
    ax.set_title('Validation Metrics')
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'validation_metrics.png'))
    plt.close(fig)

def plot_real_fake_scores(training_losses, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(training_losses['iterations'], training_losses['real_score'], label='Real Score')
    ax.plot(training_losses['iterations'], training_losses['fake_score'], label='Fake Score')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Real and Fake Scores')
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'real_fake_scores.png'))
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training and validation losses from log files.')
    parser.add_argument('log_dir', type=str, help='Path to the directory containing log files')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save the plot images')
    args = parser.parse_args()

    plot_losses_and_metrics(args.log_dir, args.output_dir)