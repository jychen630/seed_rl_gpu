#!/bin/bash

# Define the output CSV file
OUTPUT_CSV="timing_results_numpy.csv"

echo "K,Time(seconds)" > $OUTPUT_CSV
K_values=(10 30 100 300)
for K in "${K_values[@]}"
do
    echo "Running experiment with K=$K"
    start_time=$(date +%s)
    python cartpole_numpy.py --K $K
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "$K,$duration" >> $OUTPUT_CSV
    echo "Completed K=$K in $duration seconds"
done

echo -e "\nTiming Results Summary:"
echo "------------------------"
cat $OUTPUT_CSV