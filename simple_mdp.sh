
## Noise std
first_args=(0.01)

## A
second_args=(10)

## N
third_args=(100)

fourth_args=(0.0001)

fifth_args=(2.5)

# conda activate corl && ./simple_mdp.sh

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
                    # # # for i in {4..13}
                    # for i in 10
                    # do
                        
                        
                    #     # python simple_mdp/run_experiment.py --algo clipped_q_learning --num-q 100 --N 1000
                    #     # python simple_mdp/divide_mdp/run_experiment.py --algo clipped_q_learning --N 100 --seed $i
                        
                        
                    #     # python simple_mdp/divide_mdp/aggregate_saved_runs.py --root "logs/simple_mdp/divide_mdp/A=${second_arg}/10/${first_arg}/q_learning" --seed-start 1 --seed-end 10
                    #     python simple_mdp/divide_mdp/aggregate_saved_runs.py --root "logs/simple_mdp/divide_mdp/A=${second_arg}/10/${first_arg}/fronzen_test" --seed-start 1 --seed-end 10
                        
                    # done

                    for i in {1..10}
                    do
                        # python simple_mdp/divide_mdp/run_experiment.py --algo clipped_q_learning --save-path "logs/simple_mdp/divide_mdp/A=${second_arg}/${third_arg}/${first_arg}/q_learning_test" --num-epochs 1000 --A ${second_arg} --N ${third_arg} --noise-std ${first_arg} --seed $i &
                        python simple_mdp/divide_mdp/run_experiment.py --algo frozen_ratio_value_learning --save-path "logs/simple_mdp/divide_mdp/A=${second_arg}/${third_arg}/${first_arg}/fronzen" --num-epochs 1000 --A ${second_arg} --N ${third_arg} --noise-std ${first_arg} --seed $i &
                    done
                    wait
                done
            done
        done
    done
done
