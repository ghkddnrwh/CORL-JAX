

first_args=(1.0)

# second_args=(1.0 1.5 2.0 2.5 3.0)
second_args=(0.5)


third_args=(1000)

fourth_args=(0.0003)
# fourth_args=(0.0)


# fifth_args=(0.1 0.5 1.0 2.5 5.0)
# fifth_args=(1.2 1.4 1.6 1.8 2.0)
fifth_args=(1.0)


# sixth_args=(0.0 0.2 0.4 0.6 0.8 1.0)
# sixth_args=(0.1 0.3 0.5 0.7 0.9)
# sixth_args=(0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)
sixth_args=(1.0)

# conda activate corl && ./fitting.sh

# Outer loop for robustness_level
for first_arg in "${first_args[@]}"
do
    for second_arg in "${second_args[@]}"
    do
        # Inner loop for implicit_tau
        for third_arg in "${third_args[@]}"
        do
            for fourth_arg in "${fourth_args[@]}"
            do
                for fifth_arg in "${fifth_args[@]}"
                do
                    for sixth_arg in "${sixth_args[@]}"
                    do
                        # for i in {1..2}
                        for i in 0
                        do
                            # XLA_PYTHON_CLIENT_PREALLOCATE=false  CUDA_VISIBLE_DEVICES=0 python algorithms/uk_offline/cdaf_jax.py \
                            #     --env antmaze-large-play-v2 \
                            #     --device gpu \
                            #     --seed $i \
                            #     --load_model logs/tuning/cdaf_jax_prev/increased_discount/${first_arg}/${second_arg}/${third_arg} \
                            #     --max_timesteps 0 \
                            #     --refit_actor_steps 100000 \
                            #     --eval_actor_eval_freq 100000 \
                            #     --actor_refit_dir_name actor_refit/${fourth_arg}/${fifth_arg} \
                            #     --bc_coef ${fifth_arg} \
                            #     --n_episodes 10 \
                            #     --eval_actor_batch_size 256 \
                            #     --log_wandb False &

                            # XLA_PYTHON_CLIENT_MEM_FRACTION=0.15  CUDA_VISIBLE_DEVICES=0 python algorithms/uk_offline/sa_cdaf_jax.py \
                            #     --env antmaze-large-diverse-v2 \
                            #     --device gpu \
                            #     --seed $i \
                            #     --load_model logs/tuning/sa_cdaf_jax/init_max/increased_discount/${first_arg}/${third_arg} \
                            #     --max_timesteps 0 \
                            #     --refit_actor_steps 1000000 \
                            #     --eval_actor_eval_freq 50000 \
                            #     --actor_fit_method "weighted_bc" \
                            #     --actor_refit_dir_name actor_refit/weighted_bc/${fourth_arg}/${sixth_arg} \
                            #     --actor_lr ${fourth_arg} \
                            #     --policy_weight_exponent ${sixth_arg} \
                            #     --n_episodes 10 \
                            #     --eval_actor_batch_size 256 \
                            #     --log_wandb False &

                            # XLA_PYTHON_CLIENT_MEM_FRACTION=0.15  CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/cdaf_jax_coverage_margin.py \
                            #     --env antmaze-large-diverse-v2 \
                            #     --device gpu \
                            #     --seed $i \
                            #     --load_model logs/tuning/cdaf_jax_coverage_margin/init_max/increased_discount/${first_arg}/${third_arg} \
                            #     --max_timesteps 0 \
                            #     --refit_actor_steps 1000000 \
                            #     --eval_actor_eval_freq 50000 \
                            #     --actor_fit_method "weighted_bc" \
                            #     --actor_refit_dir_name actor_refit/weighted_bc/${fourth_arg}/${sixth_arg} \
                            #     --actor_lr ${fourth_arg} \
                            #     --policy_weight_exponent ${sixth_arg} \
                            #     --n_episodes 10 \
                            #     --eval_actor_batch_size 256 \
                            #     --log_wandb False &

                            # XLA_PYTHON_CLIENT_MEM_FRACTION=0.15  CUDA_VISIBLE_DEVICES=0 python algorithms/uk_offline/sa_cdaf_jax.py \
                            #     --env antmaze-large-diverse-v2 \
                            #     --device gpu \
                            #     --seed $i \
                            #     --load_model logs/tuning/sa_cdaf_jax/init_max/increased_discount/${first_arg}/${third_arg} \
                            #     --max_timesteps 0 \
                            #     --refit_actor_steps 1000000 \
                            #     --eval_actor_eval_freq 50000 \
                            #     --actor_fit_method "td3_weighted_bc" \
                            #     --actor_refit_dir_name actor_refit/td3_weighted_bc/${fourth_arg}/${fifth_arg}/${sixth_arg} \
                            #     --actor_lr ${fourth_arg} \
                            #     --bc_coef ${fifth_arg} \
                            #     --policy_weight_exponent ${sixth_arg} \
                            #     --n_episodes 10 \
                            #     --eval_actor_batch_size 256 \
                            #     --log_wandb False &

                            # XLA_PYTHON_CLIENT_MEM_FRACTION=0.15  CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/cdaf_jax_coverage_margin.py \
                            #     --env antmaze-large-diverse-v2 \
                            #     --device gpu \
                            #     --seed $i \
                            #     --load_model logs/tuning/cdaf_jax_coverage_margin/init_max/increased_discount/${first_arg}/${third_arg} \
                            #     --max_timesteps 0 \
                            #     --refit_actor_steps 1000000 \
                            #     --eval_actor_eval_freq 50000 \
                            #     --actor_fit_method "td3_weighted_bc" \
                            #     --actor_refit_dir_name actor_refit/td3_weighted_bc/${fourth_arg}/${fifth_arg}/${sixth_arg} \
                            #     --actor_lr ${fourth_arg} \
                            #     --bc_coef ${fifth_arg} \
                            #     --policy_weight_exponent ${sixth_arg} \
                            #     --n_episodes 10 \
                            #     --eval_actor_batch_size 256 \
                            #     --log_wandb False &

                            XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/basic/iql_jax.py \
                            --env antmaze-large-diverse-v2 \
                            --device gpu \
                            --seed $i \
                            --load_model logs/basic/jax \
                            --max_timesteps 0 \
                            --refit_actor_steps 1000000 \
                            --refit_actor_eval_freq 50000 \
                            --actor_refit_dir_name actor_refit/ \
                            --n_episodes 10 \
                            --refit_actor_batch_size 256 \
                            --log_wandb False &

                        done      
                    done
                done
            done
        done
    done
done
wait

