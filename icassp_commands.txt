python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --alpha_mg .3 --grad_clip 


--non_iid_split --alpha 100000, --alpha .01 (fixed- iid, non-iid)
--batch_size 32 , --batch_size 128
--contpermuted_beta 4 (fixed)
--n_clients 5 (fixed)
--num_aggs_per_task 10 (default), 20
--mean_eta 1(default), 2, 3, 4, 5
--alpha_mg .3 (0.5 default)

tmux file name
"./icassp_output_txt/iid_batch_32_beta_4_clients_2_mean_eta_1_aggs_2_alpha_mg_0.3.txt"

"./icassp_output_txt/iid_all_infer_batch_32_beta_4_clients_2_mean_eta_1_aggs_2_alpha_mg_0.3.txt"

Commands ran till now:


###
############ common command #####
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --federated_learning  --grad_clip

##
########### to be added ###########
--
--non_iid_split --alpha val --batch_size val --n_clients val --num_aggs_per_task val --mean_eta val --alpha_mg val
##

----------
IID CASES
---------
trial (clients -2, mean eta 1_aggs_2)
tmux - 0 is for trials
python main.py --seed 1000 --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_split_mnist --num_epochs $(( 5 * 30)) --batch_size 32 --train_mc_iters 10 --inference_map --federated_learning --n_clients 5 --mean_eta 1 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --non_iid --alpha 100000 --alpha_mg 0.3 --grad_clip --num_aggs_per_task 2


SAMPLE COMMANDS
python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32 --num_aggs_per_task 10 --mean_eta 1

tmux 1 - pending
(batch 32)
1. python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip --non_iid_split --alpha 100000 --batch_size 32




non-IID cases:
1. python main.py --num_workers 1 --permute_seed 2019 --seed 1000 --iterations_per_virtual_epc 469  --num_of_permutations $(( 5 - 1 )) --optimizer bgd_new_update --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_permuted_mnist --num_epochs $(( 5 * 30))  --train_mc_iters 10 --inference_map --federated_learning  --grad_clip 







#### NAZREEN SPLIT MNIST
python main.py --seed 1000 --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --dataset ds_cont_split_mnist --num_epochs $(( 5 * 30)) --batch_size 32 --train_mc_iters 10 --inference_map --federated_learning --n_clients 5 --mean_eta 1 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --non_iid --alpha 100000 --alpha_mg 0.3 --grad_clip 
