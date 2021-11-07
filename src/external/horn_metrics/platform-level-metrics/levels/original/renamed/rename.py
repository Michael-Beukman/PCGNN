import glob
I = 0
for f in glob.glob("../*.txt"):
    #print(f)
    #../SMB-w1-l1_final.txt
    good_levels = [
        (8, 1),
        (7, 1),
        (6, 3),
        (6, 2),
        (6, 1),
        (5, 3),
        (5, 1),
        (4, 2),
        (4, 1),
        (3, 3),
        (3, 1),
        (2, 1),
        (1, 3),
        (1, 2),
        (1, 1),
    ]

    good_indices = [ 1, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15,]
    good_indices = [i - 1 for i in good_indices]
    good_levels = list(reversed(good_levels))
    name, world, level = f.split('/')[1].split("_")[0].split("-")
    w = int(world[1])
    l = int(level[1])

    new_number = w * 3 + l
    new_number = I
    if (w, l) not in good_levels: continue
    else: new_number = good_levels.index((w, l))
    if new_number in good_indices:
        print((w, l), "IS GOOD")

    with open(f, 'r') as file: text = file.read()
    with open(f'new_lvl-{new_number}.txt', 'w+') as f:
        f.write(text)
    I += 1
