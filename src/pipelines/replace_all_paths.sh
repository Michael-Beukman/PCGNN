# Replace all paths with correct ones. Also exclude a few files that should not be changed.
if [[ $OSTYPE == 'darwin'* ]]; then
    grep -I --exclude "replace_all_paths.sh" --exclude "v106_metrics.py" --exclude "v202_mario_analyse_metrics.py" -rl /home/NAME/PATH_TO_ROOT/ . | xargs sed -i '' -e 's=/home/NAME/PATH_TO_ROOT/=/home/NEW_NAME/NEW_PATH_TO_ROOT/=g'
    grep -I --exclude "replace_all_paths.sh" --exclude "v106_metrics.py" --exclude "v202_mario_analyse_metrics.py" -rl NAME . | xargs sed -i '' -e 's=NAME=/NEW_NAME/=g'
else 
    grep -I --exclude "replace_all_paths.sh" --exclude "v106_metrics.py" --exclude "v202_mario_analyse_metrics.py" -rl /home/NAME/PATH_TO_ROOT/ . | xargs sed -i 's=/home/NAME/PATH_TO_ROOT/=/home/NEW_NAME/NEW_PATH_TO_ROOT/=g'
    grep -I --exclude "replace_all_paths.sh" --exclude "v106_metrics.py" --exclude "v202_mario_analyse_metrics.py" -rl NAME . | xargs sed -i 's=NAME=/NEW_NAME/=g'
fi

cd ./results/experiments/104b/runs/2021-10-29_15-37-53/
cp data_compressed.p.gz data.p.gz
gunzip data.p.gz

cd -

cd ./results/experiments/104b/runs/2021-11-01_01-12-10
cp data_compressed.p.gz data.p.gz
gunzip data.p.gz