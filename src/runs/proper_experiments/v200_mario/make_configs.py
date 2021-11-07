X = [
    './novelty_neat/configs/tiling_mario_12_7_balanced',
    './novelty_neat/configs/tiling_mario_20_28_2pred_size',
    './novelty_neat/configs/tiling_mario_56_7_1pred_size_one_hot',
    './novelty_neat/configs/tiling_mario_20_28_2pred_size_one_hot_116',
]

for file in X:
    with open(file, 'r') as f:
        lines = f.readlines()

        new_lines = [l if 'pop_size' not in l else 'pop_size              = 20\n' for l in lines]
        with open(file + "_20_pop_clean", 'w+') as F:
            F.writelines(new_lines)