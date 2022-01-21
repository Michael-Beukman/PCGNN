# Generates levels of all difficulties.
for diff in {1..10}; do
    mkdir -p ../../../runs/proper_experiments/v300_metrics/personal_maze/raw/$diff/
    for i in {1..500}; do 
        echo $i && java Handler $diff > ../../../runs/proper_experiments/v300_metrics/personal_maze/raw/$diff/$i; 
    done
done