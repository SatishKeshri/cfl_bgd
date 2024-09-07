# File for running with beta 4, eta =2 and different alphas

# (alpha_mg 0.1)
python main.py --num_workers 8 --permute_seed 2019 --seed 1000   --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 10))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 128 --n_clients 5 --num_aggs_per_task 1 --mean_eta 2 --alpha_mg 0.1 --contpermuted_beta 4 > './non_iid_full_pmnist_trial_txt/test_17_1.txt'


# (alpha_mg 0.15)
python main.py --num_workers 8 --permute_seed 2019 --seed 1000   --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 10))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 128 --n_clients 5 --num_aggs_per_task 1 --mean_eta 2 --alpha_mg 0.15 --contpermuted_beta 4 > './non_iid_full_pmnist_trial_txt/test_17_2.txt'


# (alpha_mg 0.2)
python main.py --num_workers 8 --permute_seed 2019 --seed 1000   --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 10))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 128 --n_clients 5 --num_aggs_per_task 1 --mean_eta 2 --alpha_mg 0.2 --contpermuted_beta 4 > './non_iid_full_pmnist_trial_txt/test_17_3.txt'


# (alpha_mg 0.25)
python main.py --num_workers 8 --permute_seed 2019 --seed 1000   --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 10))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 128 --n_clients 5 --num_aggs_per_task 1 --mean_eta 2 --alpha_mg 0.25 --contpermuted_beta 4 > './non_iid_full_pmnist_trial_txt/test_17_4.txt'


#(alpha_mg 0.3)
python main.py --num_workers 8 --permute_seed 2019 --seed 1000   --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 10))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 128 --n_clients 5 --num_aggs_per_task 1 --mean_eta 2 --alpha_mg 0.3 --contpermuted_beta 4 > './non_iid_full_pmnist_trial_txt/test_17_5.txt'