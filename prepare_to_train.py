# Usage
#
# First, set fg/a/bg_path in config.py
# Then:
#    python prepare_to_train.py

import os
import random
from config import fg_path, a_path, bg_path
from config import fg_names_path, bg_names_path
from config import train_names_path, valid_names_path
from config import num_bgs_per_fg, training_fraction
from math import floor, ceil
from random import shuffle

def shuffle_data(num_fgs):
    # num_fgs = 431
    # num_bgs = 43100
    # num_bgs_per_fg = 100
    num_samples = num_fgs * num_bgs_per_fg
    num_train_samples = int(ceil(training_fraction * num_samples))
    num_valid_samples = num_samples - num_train_samples

    from config import num_valid_samples as config_num_valid_samples
    from config import num_train_samples as config_num_train_samples
    from config import num_samples as config_num_samples
    
    # Validate the config
    a = num_samples != config_num_samples
    b = num_train_samples != config_num_train_samples
    c = num_valid_samples != num_valid_samples
    if a or b or c:
        raise Exception("config.py values are miscalculated: set num_samples={},num_valid_samples={},num_train_samples={}".format(num_samples, num_valid_samples, num_train_samples))
       
    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for i in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1

    from config import num_valid_samples as config_num_valid_samples
    from config import num_train_samples as config_num_train_samples
    from config import num_samples as config_num_samples

    valid_names = random.sample(names, num_valid_samples)
    train_names = [n for n in names if n not in valid_names]
    shuffle(valid_names)
    shuffle(train_names)

    with open(valid_names_path, 'w') as file:
        file.write('\n'.join(valid_names))
        print("Generated %s" % valid_names_path)

    with open(train_names_path, 'w') as file:
        file.write('\n'.join(train_names))
        print("Generated '%s'" % train_names_path)

if __name__ == '__main__':

    fg_files = os.listdir(fg_path) 
    bg_files = os.listdir(bg_path) 
    a_files = os.listdir(a_path)

    # (0) Validate
    #
    if len(fg_files) * num_bgs_per_fg > len(bg_files):
        raise Exception("Not enough backgrounds (lower num_bgs_per_fg to a maximum of {}".format(floor(len(bg_files) / len(fg_files))))

    if len(a_files) != len(fg_files):
        raise Exception("There should be as many masks as foregrounds!")

    if a_files != fg_files:
        raise Exception("The mask files and foreground files should have the same names!")

    # (1) Generate the files with the names of the images
    #
    with open(fg_names_path, 'w') as file:
        file.write('\n'.join(fg_files))
        print("Generated {}".format(fg_names_path))

    with open(bg_names_path, 'w') as file:
        file.write('\n'.join(bg_files))
        print("Generated {}".format(bg_names_path))


    # (2) Assign each foreground its backgrounds (and store the reuslts in test_names.txt and
    # valid_names.txt)
    shuffle_data(len(fg_files))
