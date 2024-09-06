python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --alpha_mg .3 --grad_clip 

## Abalation studies hyperparameters
--non_iid_split --alpha 100000, --alpha .01 (fixed- iid, non-iid)
--batch_size 32 , --batch_size 128 (batch 32 done)
--contpermuted_beta 4 (fixed)
--n_clients 5 (fixed)
--num_aggs_per_task 10 (default), 20
--mean_eta 1(default), 2, 3, 4, 5 ( done till 1,2,3)
--alpha_mg 0.1, 0.2, 0.3, 0.4,0.5 (0.5 default) (all done)

tmux file name
# only map infer
"./icassp_output_txt/iid_batch_32_beta_4_clients_2_mean_eta_1_aggs_2_alpha_mg_0.3.txt"

# all infer file name
"./icassp_output_txt/iid_all_infer_batch_val_beta_val_clients_val_num_ggs_val_mean_eta_val_alpha_mg_val.txt"

Commands ran till now:


###
############ common command #####
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --federated_learning  --grad_clip

##
########### to be added ###########
--
--non_iid_split val --alpha val --batch_size val --n_clients val --num_aggs_per_task val --mean_eta val --alpha_mg val
##

----------
IID CASES
---------
## trial (clients -2, mean eta 1_aggs_2)
tmux - 0 is for trials
python main.py --seed 1000 --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_split_mnist --num_epochs $(( 5 * 30)) --batch_size 32 --train_mc_iters 10 --inference_map --federated_learning --n_clients 5 --mean_eta 1 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --non_iid --alpha 100000 --alpha_mg 0.3 --grad_clip --num_aggs_per_task 2

## conda activate FL_env

## SAMPLE COMMANDS
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --num_aggs_per_task 10 --mean_eta 1

## Actual Runs

(batch 32 varaiations)
1-5 - changes in alpha_mg (mean eta 1)
6- 10 - changes in alpha_mg (mean eta 2)


1. tmux 1 - running - 6:07 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 1 --alpha_mg 0.1 > "./icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_1_alpha_mg_0.1.txt"

2. tmux 2  - running - 6:10 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 1 --alpha_mg 0.2 > "./icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_1_alpha_mg_0.2.txt"

3. tmux 3  - running - 6:11 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 1 --alpha_mg 0.3 > "./icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_1_alpha_mg_0.3.txt"


4. tmux 4  - running - 6:12 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 1 --alpha_mg 0.4 > "./icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_1_alpha_mg_0.4.txt"

5. tmux 5  - running - 6:13 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 1 --alpha_mg 0.5 > "./icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_1_alpha_mg_0.5.txt"

# mean eta =2

1. tmux 6 - running - 6:19 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 2 --alpha_mg 0.1 > "./icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_2_alpha_mg_0.1.txt"

2. tmux 7  - running - 6:26 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 2 --alpha_mg 0.2 > "./icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_2_alpha_mg_0.2.txt"

3. tmux 8  - running - 6:28 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 2 --alpha_mg 0.3 > "./icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_2_alpha_mg_0.3.txt"


4. tmux 9  - running - 6:28 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 2 --alpha_mg 0.4 > "./icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_2_alpha_mg_0.4.txt"

5. tmux 10  - running - 6:29 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 2 --alpha_mg 0.5 > "./icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_2_alpha_mg_0.5.txt"


# mean eta = 3

1. tmux 11 - running - 6:33 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 3 --alpha_mg 0.1 > "./icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_3_alpha_mg_0.1.txt"

2. tmux 12  - running - 6:34 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 3 --alpha_mg 0.2 > "./icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_3_alpha_mg_0.2.txt"

3. tmux 13  - running - 6:34 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 3 --alpha_mg 0.3 > "./icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_3_alpha_mg_0.3.txt"


4. tmux 14  - running - 6:34 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 3 --alpha_mg 0.4 > "./icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_3_alpha_mg_0.4.txt"

5. tmux 15  - running - 6:35 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 3 --alpha_mg 0.5 > "./icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_3_alpha_mg_0.5.txt"



#### non-IID cases:
mean_eta = 1

1. tmux 16 - running - 6:41 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 1 --alpha_mg 0.1 > "./icassp_output_txt/non_iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_1_alpha_mg_0.1.txt"

2. tmux 17  - running - 6:41 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 1 --alpha_mg 0.2 > "./icassp_output_txt/non_iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_1_alpha_mg_0.2.txt"

3. tmux 18  - running - 6:42 PM (this one ran by mistake once in normal terminal)
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 1 --alpha_mg 0.3 > "./icassp_output_txt/non_iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_1_alpha_mg_0.3.txt"


4. tmux 19  - running - 6:43 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 1 --alpha_mg 0.4 > "./icassp_output_txt/non_iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_1_alpha_mg_0.4.txt"

5. tmux 20  - running - 6:45 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 1 --alpha_mg 0.5 > "./icassp_output_txt/non_iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_1_alpha_mg_0.5.txt"

# mean eta =2

1. tmux 21 - running - 6:46 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 2 --alpha_mg 0.1 > "./icassp_output_txt/non_iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_2_alpha_mg_0.1.txt"

2. tmux 22  - running - 6:46 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 2 --alpha_mg 0.2 > "./icassp_output_txt/non_iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_2_alpha_mg_0.2.txt"

3. tmux 23  - running - 6:47 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 2 --alpha_mg 0.3 > "./icassp_output_txt/non_iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_2_alpha_mg_0.3.txt"


4. tmux 24  - running - 6:47 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 2 --alpha_mg 0.4 > "./icassp_output_txt/non_iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_2_alpha_mg_0.4.txt"

5. tmux 25  - running - 6:47 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 2 --alpha_mg 0.5 > "./icassp_output_txt/non_iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_2_alpha_mg_0.5.txt"


# mean eta = 3

1. tmux 26 - running - 6:48 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 3 --alpha_mg 0.1 > "./icassp_output_txt/non_iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_3_alpha_mg_0.1.txt"

2. tmux 27  - running - 6:48 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 3 --alpha_mg 0.2 > "./icassp_output_txt/non_iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_3_alpha_mg_0.2.txt"

3. tmux 28  - running - 6:48 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 3 --alpha_mg 0.3 > "./icassp_output_txt/non_iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_3_alpha_mg_0.3.txt"


4. tmux 29  - running - 6:49 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 3 --alpha_mg 0.4 > "./icassp_output_txt/non_iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_3_alpha_mg_0.4.txt"

5. tmux 30  - running - 6:49 PM
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 5 --num_aggs_per_task 10 --mean_eta 3 --alpha_mg 0.5 > "./icassp_output_txt/non_iid_all_infer_batch_32_beta_4_clients_5_num_aggs_10_mean_eta_3_alpha_mg_0.5.txt"







#### NAZREEN SPLIT MNIST
python main.py --seed 1000 --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_split_mnist --num_epochs $(( 5 * 30)) --batch_size 32 --train_mc_iters 10 --inference_map --federated_learning --n_clients 5 --mean_eta 1 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --non_iid --alpha 100000 --alpha_mg 0.3 --grad_clip 
