# Must be inside `src/pipelines`
cd ../
# Metrics
echo "Running Metrics Analysis Now"
./run.sh analysis/proper_experiments/v400/analyse_all_metrics_properly.py

# Stats
echo "Running Statistical Analysis Now"
./run.sh analysis/proper_experiments/v400/analyse_all_statistical_tests.py