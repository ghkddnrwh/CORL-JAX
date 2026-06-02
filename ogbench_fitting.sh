

first_args=(0.999)

# second_args=(1.0 1.5 2.0 2.5 3.0)
# second_args=(0.0 0.1 0.2 0.3 0.4)
# second_args=(0.1 0.3 0.5 1.0 3.0 5.0 10.0)
second_args=(3.0)

third_args=(1000)

fourth_args=(10)
# fourth_args=(0.0 0.2 0.4 0.6 0.8 1.0)

fifth_args=(5 6 7)
# fifth_args=(0.2 0.4 0.6 0.8 1.0)


# sixth_args=(0.0 0.2 0.4 0.6 0.8 1.0)
# sixth_args=(0.1 0.3 0.5 0.7 0.9)
# sixth_args=(0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)
sixth_args=(1.0)


# conda activate ogbench && ./ogbench_fitting.sh

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
                    # for i in {0..1}
                    for i in 0
                    do
                        # XLA_PYTHON_CLIENT_PREALLOCATE=false

                        ################################################################
                        ### IQL Based
                        ################################################################
                        # --env puzzle-4x4-play-singletask-v0 \


                        XLA_PYTHON_CLIENT_MEM_FRACTION=0.14 CUDA_VISIBLE_DEVICES=1 python algorithms/uk_offline/decoupled_delayed_iql_jax.py \
                        --project "ORL-BIAS-Refit" \
                        --mode "refit" \
                        --env puzzle-4x4-play-singletask-v0 \
                        --device gpu \
                        --seed $i \
                        --load_model logs/tuning/iql_based/ogbench/decoupled_delayed_iql_jax/${first_arg}/${fourth_arg}/${third_arg} \
                        --max_timesteps 100000 \
                        --eval_freq 10000 \
                        --beta ${fifth_arg} \
                        --actor_refit_dir_name actor_refit/${fifth_arg} \
                        --n_episodes 10 &

                        sleep 10


                        XLA_PYTHON_CLIENT_MEM_FRACTION=0.14 CUDA_VISIBLE_DEVICES=0 python algorithms/uk_offline/decoupled_delayed_iql_jax.py \
                        --project "ORL-BIAS-Refit" \
                        --mode "refit" \
                        --env puzzle-3x3-play-singletask-v0 \
                        --device gpu \
                        --seed $i \
                        --load_model logs/tuning/iql_based/ogbench/decoupled_delayed_iql_jax/${first_arg}/${fourth_arg}/${third_arg} \
                        --max_timesteps 100000 \
                        --eval_freq 10000 \
                        --beta ${fifth_arg} \
                        --actor_refit_dir_name actor_refit/${fifth_arg} \
                        --n_episodes 10 &

                        sleep 10
                        wait
                    done
                done      
            done
        done
    done
done
