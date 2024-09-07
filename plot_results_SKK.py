import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import os
# print(os.getcwd())

def plot_average_accuracies_over_rounds_new(sgd_accuracies, bgd_accuracies,new_bgd_accuracies_0_1, new_bgd_accuracies_0_3):
    plt.figure(figsize=(10, 6))
    tasks = range(0, len(sgd_accuracies))
    
    sgd_means = [np.mean(np.array(task_accs)) for task_accs in sgd_accuracies]
    print(f"sgd_means: {sgd_means}")
    plt.plot(tasks, sgd_means, marker='o', label='CFL_SGD',linestyle='--', linewidth=2.5)
    
    bgd_means = [np.mean(np.array(task_accs)) for task_accs in bgd_accuracies]
    print(f"bgd_means: {bgd_means}")
    plt.plot(tasks, bgd_means, marker='s', label='CFL_OLD_BGD', linewidth=2.5)

    new_bgd_0_1_means = [np.mean(np.array(task_accs)) for task_accs in new_bgd_accuracies_0_1]
    print(f"new_bgd_means_0.1: {new_bgd_0_1_means}")
    plt.plot(tasks, new_bgd_0_1_means, marker='*', label='CFL_NEW_BGD_0.1_alpha_mg', linewidth=2.5)

    new_bgd_0_3_means = [np.mean(np.array(task_accs)) for task_accs in new_bgd_accuracies_0_3]
    print(f"new_bgd_means_0.3: {new_bgd_0_3_means}")
    plt.plot(tasks, new_bgd_0_3_means, marker='>', label='CFL_NEW_BGD_0.3_alpha_mg', linewidth=2.5)
    
    plt.xlabel('Task ID', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    # plt.title(f'Average Accuracies per round for SGD and BGD - FedAvg BGD ')
    plt.title(f'Average Accuracy across tasks', fontweight='bold')
    plt.xticks(range(max(len(sgd_accuracies), len(bgd_accuracies))), [f'{i+1}' for i in range(max(len(sgd_accuracies), len(bgd_accuracies)))], fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./plot_images/average_accuracies_over_rounds_new.png')
    plt.show()



# tasks_5_bgd_sgd_iid_split_mnist_alpha_100000 = {'bgd_new_update_0.1' : [[98.44],[77.589, 90.353],[56.643, 83.252, 90.715], [56.548, 73.066, 48.826, 97.281],[43.83, 63.663, 21.078, 95.77, 95.512]],
#                                                         'bgd_new_update_0.3' : [[97.92],[91.489, 86.729], [61.04, 81.783, 86.766],[67.518, 86.533, 67.93, 95.972],[46.856, 73.017, 36.98, 97.281, 92.335]],
#                               'bgd_old_update' : [[99.574],[78.014, 97.845],[74.657, 89.373, 97.012], [47.139, 68.56, 33.885, 98.691],[34.326, 58.031, 18.356, 96.123, 98.689]],
#                               'sgd' : [[99.669],[77.967, 99.363],[71.348, 91.577, 98.239],[44.539, 60.578, 15.742, 99.295],[28.322, 60.774, 13.501, 96.878, 99.395]]}

"""
tasks_5_bgd_sgd_iid_split_mnist_alpha_100000 = {'bgd_new_update_0.1' : [[54.28],[46.62, 59.78],[56.643, 83.252, 90.715], [56.548, 73.066, 48.826, 97.281],[43.83, 63.663, 21.078, 95.77, 95.512]],
                                                        'bgd_new_update_0.3' : [[97.92],[91.489, 86.729], [61.04, 81.783, 86.766],[67.518, 86.533, 67.93, 95.972],[46.856, 73.017, 36.98, 97.281, 92.335]],
                              'bgd_old_update' : [[99.574],[78.014, 97.845],[74.657, 89.373, 97.012], [47.139, 68.56, 33.885, 98.691],[34.326, 58.031, 18.356, 96.123, 98.689]],
                              'sgd' : [[99.669],[77.967, 99.363],[71.348, 91.577, 98.239],[44.539, 60.578, 15.742, 99.295],[28.322, 60.774, 13.501, 96.878, 99.395]]}


plot_average_accuracies_over_rounds_new(tasks_5_bgd_sgd_iid_split_mnist_alpha_100000['sgd'], tasks_5_bgd_sgd_iid_split_mnist_alpha_100000['bgd_old_update'],tasks_5_bgd_sgd_iid_split_mnist_alpha_100000['bgd_new_update_0.1'],tasks_5_bgd_sgd_iid_split_mnist_alpha_100000['bgd_new_update_0.3'])
"""

# plot_task_accuracies_roundwise(tasks_5_bgd_sgd_iid_split_mnist_alpha_100000['sgd'], tasks_5_bgd_sgd_iid_split_mnist_alpha_100000['bgd'],tasks_5_bgd_sgd_iid_split_mnist_alpha_100000['bgd_new_update_0.1'],tasks_5_bgd_sgd_iid_split_mnist_alpha_100000['ngd_new_update_0.3'])
# sgd_avg_forgetting, bgd_avg_forgetting = calculate_forgetting(tasks_5_bgd_sgd_iid_split_mnist_alpha_100000['sgd'], tasks_5_bgd_sgd_iid_split_mnist_alpha_100000['bgd_old_update'],tasks_5_bgd_sgd_iid_split_mnist_alpha_100000['bgd_new_update_0.1'],tasks_5_bgd_sgd_iid_split_mnist_alpha_100000['bgd_new_update_0.3'])
# sgd_avg_forgetting, bgd_avg_forgetting



# Function to extract task-wise accuracies for rounds 10, 20, ...
def extract_accuracies(file_path, non_iid: bool=False, rounds_per_task=10):
    iid_accuracies, non_iid_accuracies = {}, {}
    acc_list = []
    iid_files, non_iid_files = [], []
    for filename in os.listdir(file_path):
        if "non_iid" in filename:
            non_iid_files.append(filename)
        else:
            iid_files.append(filename)
    print(f"iid_files: {iid_files}")
    print(f"non_iid_files: {non_iid_files}")

    for filename in iid_files:
        accuracy_dict = {}
        with open(file_path + filename, 'r') as file:
            for line in file:
                # Regex pattern for extracting round number and accuracies
                match = re.search(r'Task wise accuracies after round (\d+) are \[([0-9.,\s]+)\]', line)
                if match:
                    round_num = int(match.group(1))
                    # Check if round number is a multiple of 10
                    if round_num % rounds_per_task == 0:
                        accuracy_values = [float(x) for x in match.group(2).split(',')]
                        accuracy_dict[round_num] = accuracy_values
                        # acc_list.append(accuracy_values)
                if "Loss is NaN" in line:
                    accuracy_dict[round_num+1] = [0.0]*(len(accuracy_values)+1)
        iid_accuracies[filename[-22:-4]] = accuracy_dict.values()
    print(iid_accuracies)
    for filename in non_iid_files:
        accuracy_dict = {}
        with open(file_path + filename, 'r') as file:
            for line in file:
                # Regex pattern for extracting round number and accuracies
                match = re.search(r'Task wise accuracies after round (\d+) are \[([0-9.,\s]+)\]', line)
                if match:
                    round_num = int(match.group(1))
                    # Check if round number is a multiple of 10
                    if round_num % rounds_per_task == 0:
                        accuracy_values = [float(x) for x in match.group(2).split(',')]
                        accuracy_dict[round_num] = accuracy_values
                        # acc_list.append(accuracy_values)
                if "Loss is NaN" in line:
                    accuracy_dict[round_num+1] = [0.0]*(len(accuracy_values)+1)
        non_iid_accuracies[filename[-22:-4]] = accuracy_dict.values()
    print(non_iid_accuracies)
    return iid_accuracies, non_iid_accuracies


def extract_avg_accs_from_ds_file(file_dir, rounds_per_task=10):
    accuracies = {}
    pattern = r'\[(.*?)\]'
    for filename in os.listdir(file_path):
        print(f"############   {filename} #############")
        with open(file_dir+filename, 'r') as file:
            for line in file:
                # Regex pattern for extracting round number and accuracies
                if "Average accuracies" in line:
                    matches = re.search(pattern, line)
                    # Convert the extracted string to a list of floats
                    if matches:
                        numbers = [float(num) for num in matches.group(1).split(', ') if num]
                        numbers = [numbers[idx] for idx in range(0, len(numbers), rounds_per_task)]
                        print(numbers)
                        # print(numbers)
                        file_name = filename[-112:-41]
                        accuracies[file_name] = numbers
    return accuracies

def extract_avg_test_accuracies_log_file(file_dir, rounds_per_task=10):
    accuracies = {}
    # Pattern to match the required parameters
    pattern_namespace = r'Namespace\((.*?)\)'
    # pattern_args = re.compile(r"beta='([^']*)'.*?nn_arch='([^']*)'.*?logname='([^']*)'")
    #  a = r"(beta|alpha_mg|alpha|mean_eta|n_clients|num_aggs_per_task)=([0-9.]+)"
    pattern_avg_acc = r"Avg Test Accuracies\s*:\s*(\[[^\]]*\])"

    for filename in os.listdir(file_path):
        print(f"############   {filename} #############")
        with open(file_dir+filename, 'r') as file:
            for line in file:
                # Find all matches
                matches_args = re.findall(pattern_namespace, line)
                if len(matches_args) > 0:
                    args = matches_args[0].split(', ')
                    args_list = [tuple(i.split("=")) for i in args]
                    matches_args = [i for i in args_list if i[0] in ['optimizer','beta', 'alpha_mg', 'alpha', 'mean_eta', 'n_clients', 'num_aggs_per_task']]
                    file_name = "_".join([f"{key}_{val}" for key, val in matches_args])
                    # if "100000" in file_name:
                    #     break
                # Create the output string
                    args_used = "_".join([f"{key}_{val}" for key, val in matches_args])
                # Regex pattern for extracting round number and accuracies
                if "Avg Test Accuracies" in line:
                    matches = re.search(pattern_avg_acc, line)
                    # Convert the extracted string to a list of floats
                    if matches:
                        ext_list = matches.group(1)
                        ext_list = ext_list.split(', ')
                        ext_list = [i.replace('[', '') for i in ext_list]
                        ext_list = [i.replace(']', '') for i in ext_list]
                        # ext_list = map(str.replace('[', ''), ext_list)
                        # ext_list = map(str.replace(']', ''), ext_list)
                        numbers = list(map(float, ext_list))
                        numbers = [numbers[idx] for idx in range(rounds_per_task-1, len(numbers), rounds_per_task)]
                        print(numbers)
                        # print(numbers)
                        file_name = args_used
                        accuracies[file_name] = numbers
    return accuracies


def plot_average_accuracies_over_rounds_multi_runs(accuracies, fig_name:str):
    plt.figure(figsize=(10, 6))
    
    
    max_length = 0
    lengths = []
    colors = plt.get_cmap('tab20').colors  # Get 15 distinct colors
    for id, (file, task_accuracies) in enumerate(accuracies.items()):
        tasks = range(0, len(task_accuracies))
        task_means = [np.mean(np.array(task_accs)) for task_accs in task_accuracies]
        print(f"{file}_means: {task_means}")
        plt.plot(tasks, task_means, label=file, linewidth=2.5, color=colors[id % 30])
        lengths.append(len(task_accuracies))
        max_length = max(max_length, len(task_accuracies))
    
    plt.xlabel('Task ID', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    # plt.title(f'Average Accuracies per round for SGD and BGD - FedAvg BGD ')
    plt.title(f'Average Accuracy across tasks', fontweight='bold')
    plt.xticks(range(max_length), [f'{i+1}' for i in range(max(*lengths))], fontweight='bold')
    # plt.yticks(fontweight='bold')
    plt.legend(fontsize=16)
    plt.grid(True)
    # plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('./plot_images/baselines/'+fig_name+'.png', bbox_inches='tight')
    # plt.show()



# # Example usage:
# filepath = './icassp_output_txt/'  # Replace with the directory of your files
# accuracies, acc_list = extract_accuracies(filepath)

# Exract accuracies from the file
file_path = './plot_logs_txts/' #'./icassp_output_txt/'
# file_path = './icassp_output_txt/'
# iid_accuracies, non_iid_accuracies = extract_accuracies(file_path, rounds_per_task=10)
accuracies = extract_avg_test_accuracies_log_file(file_path)
print(accuracies)
plot_average_accuracies_over_rounds_multi_runs(accuracies, fig_name='logs_plot_comaprison_pmnist')


# Plot
#iid plot
# plot_average_accuracies_over_rounds_multi_runs(iid_accuracies, fig_name='iid_plots')
#non_iid plot
# plot_average_accuracies_over_rounds_multi_runs(non_iid_accuracies, fig_name='Nazreen_pmnist_baseline_non_iid_plots')