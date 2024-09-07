###### Best one - Nazreen
#num_agg 20, alpha 0.01, alpha_mg 0.3, mean_eta 1, num_clients 5


# (num_agg 20, batch 128, clients 5)
python main.py --num_workers 8 --permute_seed 2019 --seed 1000   --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 10))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 128 --n_clients 5 --num_aggs_per_task 20 --mean_eta 1 --alpha_mg 0.3 --contpermuted_beta 4 > './non_iid_full_pmnist_trial_txt/nazBest_19_1.txt'

#(clients 5->10, batch 128)
python main.py --num_workers 8 --permute_seed 2019 --seed 1000   --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 10))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 128 --n_clients 10 --num_aggs_per_task 20 --mean_eta 1 --alpha_mg 0.3 --contpermuted_beta 4 > './non_iid_full_pmnist_trial_txt/nazBest_19_2.txt'

# (num_agg 20, clients 5, batch 32, Nazreen's original best one on full dataset)
python main.py --num_workers 8 --permute_seed 2019 --seed 1000   --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 10))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 5 --num_aggs_per_task 20 --mean_eta 1 --alpha_mg 0.3 --contpermuted_beta 4 > './non_iid_full_pmnist_trial_txt/nazBest_19_3.txt'

#(num_agg 20, clients 10, batch 32)
python main.py --num_workers 8 --permute_seed 2019 --seed 1000   --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 10))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 0.01 --batch_size 32 --n_clients 10 --num_aggs_per_task 20 --mean_eta 1 --alpha_mg 0.3 --contpermuted_beta 4 > './non_iid_full_pmnist_trial_txt/nazBest_19_4.txt'