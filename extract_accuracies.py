import re
import os
print(os.getcwd())

# Function to extract task-wise accuracies for rounds 10, 20, ...
def extract_accuracies(filename):
    accuracies = {}
    acc_list = []
    with open(filename, 'r') as file:
        for line in file:
            # Regex pattern for extracting round number and accuracies
            match = re.search(r'Task wise accuracies after round (\d+) are \[([0-9.,\s]+)\]', line)
            if match:
                round_num = int(match.group(1))
                # Check if round number is a multiple of 10
                if round_num % 10 == 0:
                    accuracy_values = [float(x) for x in match.group(2).split(',')]
                    accuracies[round_num] = accuracy_values
                    acc_list.append(accuracy_values)
    return accuracies, acc_list

# Example usage:
filename = './icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_1_alpha_mg_0.1.txt'  # Replace with the path to your file
accuracies, acc_list = extract_accuracies(filename)
# print(accuracies)
# print(acc_list)
print(list(accuracies.values()) )
