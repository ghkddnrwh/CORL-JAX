

first_args=(1)

# second_args=(1.0 1.5 2.0 2.5 3.0)
# second_args=(0.0 0.1 0.2 0.3 0.4)
second_args=(0.0)

third_args=(10000)

fourth_args=(100)
# fourth_args=(0.0 0.2 0.4 0.6 0.8 1.0)

fifth_args=(0.9)
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
                for fifth_arg in "${fifth_args[@]}"
                do
                    # for i in {0..2}
                    for i in 0
                    do
                        # XLA_PYTHON_CLIENT_PREALLOCATE=false
                        

                        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/sa_cdaf_jax.py \
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

                        # sleep 10

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

                        # sleep 10


                        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/cdaf_jax.py \
                        # --env antmaze-large-diverse-v2 \
                        # --hyperparams_path hyperparams/cdaf_jax.yml \
                        # --device gpu \
                        # --seed $i \
                        # --checkpoints_path logs/tuning/ensemble/cdaf_jax/${fourth_arg}/${first_arg}/${second_arg}/${third_arg} \
                        # --min_weight_exponent ${first_arg} \
                        # --max_weight_exponent ${first_arg} \
                        # --policy_weight_exponent ${first_arg} \
                        # --beta_min ${second_arg} \
                        # --delayed_update_period ${third_arg} \
                        # --ensemble_size ${fourth_arg} \
                        # --actor_update_method "td3_weighted_bc" \
                        # --discount 0.999 \
                        # --n_episodes 0



                        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.085 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/dave_iql_jax.py \
                        # --env antmaze-large-diverse-v2 \
                        # --hyperparams_path hyperparams/dave_iql_jax.yml \
                        # --device gpu \
                        # --seed $i \
                        # --checkpoints_path logs/tuning/ensemble/dave_iql_jax/${fourth_arg}/${first_arg}/${second_arg}/${third_arg} \
                        # --v_filter_exponent ${first_arg} \
                        # --v_filter_floor ${second_arg} \
                        # --delayed_update_period ${third_arg} \
                        # --ensemble_size ${fourth_arg} \
                        # --n_episodes 10 &



                        ################################################################
                        ### IQL Based
                        ################################################################

                        XLA_PYTHON_CLIENT_MEM_FRACTION=0.09 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/basic/iql_jax.py \
                        --env kitchen-complete-v0 \
                        --hyperparams_path hyperparams/iql_jax.yml \
                        --device gpu \
                        --seed $i \
                        --checkpoints_path logs/tests/basic/iql_jax/${first_arg} \
                        --discount ${first_arg} \
                        --n_episodes 10 &

                        sleep 10

                        XLA_PYTHON_CLIENT_MEM_FRACTION=0.09 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/basic/iql_jax.py \
                        --env kitchen-partial-v0 \
                        --hyperparams_path hyperparams/iql_jax.yml \
                        --device gpu \
                        --seed $i \
                        --checkpoints_path logs/tests/basic/iql_jax/${first_arg} \
                        --discount ${first_arg} \
                        --n_episodes 10 &

                        sleep 10
                        XLA_PYTHON_CLIENT_MEM_FRACTION=0.09 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/basic/iql_jax.py \
                        --env kitchen-mixed-v0 \
                        --hyperparams_path hyperparams/iql_jax.yml \
                        --device gpu \
                        --seed $i \
                        --checkpoints_path logs/tests/basic/iql_jax/${first_arg} \
                        --discount ${first_arg} \
                        --n_episodes 10 &

                        sleep 10




                        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/basic/rebrac_jax.py \
                        # --env antmaze-large-play-v2 \
                        # --hyperparams_path hyperparams/rebrac_jax.yml \
                        # --device gpu \
                        # --seed $i \
                        # --checkpoints_path logs/tests/basic/rebrac_jax/${first_arg} \
                        # --discount ${first_arg} \
                        # --max_timesteps 100000 \
                        # --n_episodes 10 &



                        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.09 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/decoupled_delayed_iql_jax.py \
                        # --env relocate-human-v1 \
                        # --hyperparams_path hyperparams/decoupled_delayed_iql_jax.yml \
                        # --device gpu \
                        # --seed $i \
                        # --checkpoints_path logs/tuning/iql_based/decoupled_delayed_iql_jax/${first_arg}/${fourth_arg}/${third_arg} \
                        # --discount ${first_arg} \
                        # --delayed_update_period ${third_arg} \
                        # --ensemble_size ${fourth_arg} \
                        # --n_episodes 10 &
                        # sleep 10

                        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/decoupled_delayed_iql_jax.py \
                        # --env hammer-human-v1 \
                        # --hyperparams_path hyperparams/decoupled_delayed_iql_jax.yml \
                        # --device gpu \
                        # --seed $i \
                        # --checkpoints_path logs/tuning/iql_based/decoupled_delayed_iql_jax/${first_arg}/${fourth_arg}/${third_arg} \
                        # --discount ${first_arg} \
                        # --delayed_update_period ${third_arg} \
                        # --ensemble_size ${fourth_arg} \
                        # --n_episodes 10 &
                        # sleep 10

                        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/decoupled_delayed_iql_jax.py \
                        # --env door-human-v1 \
                        # --hyperparams_path hyperparams/decoupled_delayed_iql_jax.yml \
                        # --device gpu \
                        # --seed $i \
                        # --checkpoints_path logs/tuning/iql_based/decoupled_delayed_iql_jax/${first_arg}/${fourth_arg}/${third_arg} \
                        # --discount ${first_arg} \
                        # --delayed_update_period ${third_arg} \
                        # --ensemble_size ${fourth_arg} \
                        # --n_episodes 10 &
                        # sleep 10

                        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.09 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/decoupled_delayed_iql_jax.py \
                        # --env antmaze-large-diverse-v2 \
                        # --hyperparams_path hyperparams/decoupled_delayed_iql_jax.yml \
                        # --device gpu \
                        # --seed $i \
                        # --checkpoints_path logs/tuning/iql_based/decoupled_delayed_iql_jax/${first_arg}/${fourth_arg}/${third_arg} \
                        # --discount ${first_arg} \
                        # --delayed_update_period ${third_arg} \
                        # --ensemble_size ${fourth_arg} \
                        # --n_episodes 10 &
                        # sleep 10


                        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/delayed_iql_jax.py \
                        # --env antmaze-large-play-v2 \
                        # --hyperparams_path hyperparams/delayed_iql_jax.yml \
                        # --device gpu \
                        # --seed $i \
                        # --checkpoints_path logs/tuning/iql_based/delayed_iql_jax/${first_arg}/${third_arg} \
                        # --discount ${first_arg} \
                        # --delayed_update_period ${third_arg} \
                        # --n_episodes 10 &

                        # sleep 10

                        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/delayed_iql_jax.py \
                        # --env antmaze-large-diverse-v2 \
                        # --hyperparams_path hyperparams/delayed_iql_jax.yml \
                        # --device gpu \
                        # --seed $i \
                        # --checkpoints_path logs/tuning/iql_based/delayed_iql_jax/${first_arg}/${third_arg} \
                        # --discount ${first_arg} \
                        # --delayed_update_period ${third_arg} \
                        # --n_episodes 10 &

                        # sleep 10

                        # python algorithms/uk_offline/basic/iql.py \
                        # --config configs/offline/iql/antmaze/large_play_v2.yaml \
                        # --checkpoints_path logs/basic/pytorch \
                        # --device "cuda:1" --seed $i


                        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.09 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/dave_iql_jax.py \
                        # --env antmaze-large-play-v2 \
                        # --hyperparams_path hyperparams/dave_iql_jax.yml \
                        # --device gpu \
                        # --seed $i \
                        # --checkpoints_path logs/tuning/dave_iql_jax/${first_arg}/${third_arg} \
                        # --beta ${first_arg} \
                        # --v_filter_temperature ${first_arg} \
                        # --delayed_update_freq ${third_arg} 
                    done
                done      
            done
        done
    done
done
