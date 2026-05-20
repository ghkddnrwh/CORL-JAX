

first_args=(1.0)

# second_args=(1.0 1.5 2.0 2.5 3.0)
second_args=(0.1)
# second_args=(0.6)

third_args=(1000)

fourth_args=(100 1000)
# fourth_args=(0.0 0.2 0.4 0.6 0.8 1.0)

fifth_args=(0.1)
# fifth_args=(0.2 0.4 0.6 0.8 1.0)


# sixth_args=(0.0 0.2 0.4 0.6 0.8 1.0)
# sixth_args=(0.1 0.3 0.5 0.7 0.9)
# sixth_args=(0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)
sixth_args=(1.0)

# conda activate corl && ./ant.sh

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

                # for i in {1..2}
                for i in 0
                do
                    # XLA_PYTHON_CLIENT_PREALLOCATE=false
                    

                    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 CUDA_VISIBLE_DEVICES=0 python algorithms/uk_offline/sa_cdaf_jax.py \
                    # --env antmaze-large-diverse-v2 \
                    # --hyperparams_path hyperparams/sa_cdaf_jax.yml \
                    # --device gpu \
                    # --seed $i \
                    # --checkpoints_path logs/tuning/ensemble/sa_cdaf_jax/${fourth_arg}/${first_arg}/${third_arg} \
                    # --min_weight_exponent ${first_arg} \
                    # --max_weight_exponent ${first_arg} \
                    # --policy_weight_exponent ${first_arg} \
                    # --delayed_update_period ${third_arg} \
                    # --ensemble_size ${fourth_arg} \
                    # --actor_update_method "td3_weighted_bc" \
                    # --discount 0.999 &

                    # sleep 1

                    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/cdaf_jax_coverage_margin.py \
                    # --env antmaze-large-diverse-v2 \
                    # --hyperparams_path hyperparams/cdaf_jax_coverage_margin.yml \
                    # --device gpu \
                    # --seed $i \
                    # --checkpoints_path logs/tuning/ensemble/cdaf_jax_coverage_margin/${fourth_arg}/${first_arg}/${third_arg} \
                    # --min_weight_exponent ${first_arg} \
                    # --max_weight_exponent ${first_arg} \
                    # --policy_weight_exponent ${first_arg} \
                    # --delayed_update_period ${third_arg} \
                    # --ensemble_size ${fourth_arg} \
                    # --actor_update_method "td3_weighted_bc" \
                    # --discount 0.999 &

                    # sleep 1


                    XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 CUDA_VISIBLE_DEVICES=0 python algorithms/uk_offline/cdaf_jax.py \
                    --env antmaze-large-diverse-v2 \
                    --hyperparams_path hyperparams/cdaf_jax.yml \
                    --device gpu \
                    --seed $i \
                    --checkpoints_path logs/tuning/ensemble/cdaf_jax/${fourth_arg}/${first_arg}/${second_arg}/${third_arg} \
                    --min_weight_exponent ${first_arg} \
                    --max_weight_exponent ${first_arg} \
                    --policy_weight_exponent ${first_arg} \
                    --beta_min ${second_arg} \
                    --delayed_update_period ${third_arg} \
                    --ensemble_size ${fourth_arg} \
                    --actor_update_method "td3_weighted_bc" \
                    --discount 0.999 \
                    --n_episodes 0



                    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 CUDA_VISIBLE_DEVICES=0 python algorithms/uk_offline/basic/iql_jax.py \
                    # --env antmaze-large-play-v2 \
                    # --hyperparams_path hyperparams/iql_jax.yml \
                    # --device gpu \
                    # --seed $i \
                    # --checkpoints_path logs/basic/jax

                    # python algorithms/uk_offline/basic/iql.py \
                    # --config configs/offline/iql/antmaze/large_play_v2.yaml \
                    # --checkpoints_path logs/basic/pytorch \
                    # --device "cuda:1" --seed $i


                    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.045 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/dave_iql_jax.py \
                    # --env antmaze-large-play-v2 \
                    # --hyperparams_path hyperparams/dave_iql_jax.yml \
                    # --device gpu \
                    # --seed $i \
                    # --checkpoints_path logs/tuning/dave_iql_jax/${first_arg}/${third_arg} \
                    # --beta ${first_arg} \
                    # --v_filter_temperature ${first_arg} \
                    # --delayed_update_freq ${third_arg} 
                done      
                wait
            done
        done
    done
done