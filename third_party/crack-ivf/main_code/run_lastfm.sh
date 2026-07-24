#!/bin/bash
#
# run_lastfm.sh
#
# Standalone script to run the lastfm-64-dot dataset (baselines + CrackIVF),
# added after the main 6-dataset run since it wasn't in the original
# run_baselines.sh / run_ours.sh. Kept separate here so it can be re-run
# on its own without redoing the full multi-hour grid.
#
# Usage: ./run_lastfm.sh
# (run from main_code/, with the crack-paper-final conda env active)

run_command() {
    echo "Running: $1"
    eval $1

    if [ $? -ne 0 ]; then
        echo "Error: Command failed. Exiting."
        exit 1
    fi

    echo "Command completed successfully: $1"
}

run_command "python run_baselines.py --store --index_name BruteForce IVFFlat --dbname lastfm-64-dot --runid 250227_rerun_baselines --nthreads 16 --niter 10 --nprobe 1 2 4 8 16 32 64 128 256 512 1024 --nlist 100 1000 5000 10000 16000"

run_command "python run_ours.py --run_desc 20250227-1M-alpha_0.5-MINPTS_2-Pts2CentThres_64-conv_200-nthread_16 --nthreads 16 --store --detailed --clear_results --target_queries 1000000 --nruns 1 --nlist 100 --niter 10 --nprobe 99999 --get_qps --dbname lastfm-64-dot"
