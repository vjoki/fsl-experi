# Make a smaller subset of a bigger dataset.
#
# Made adhoc for Plenary Sessions of the Parliament of Finland, Downloadable Version 1 dataset.
import os
import random
import glob
import shutil

PICKS = 80
DEST_DIR = 'data/edus80/'
DATA_DIR = 'data/eduskunta/'

# Take PICKS samples from each speaker, skipping speakers that don't have enough samples.
for d in os.listdir(DATA_DIR):
    # Each .wav file is accompanied by a .txt file.
    if len(os.listdir(DATA_DIR + d)) < 2*PICKS:
        print('skipping ' + d)
        continue

    newd = DEST_DIR + os.path.basename(d)
    os.makedirs(newd)
    #print('mkdir ' + newd)

    picks = random.sample(glob.glob(DATA_DIR + d + '/*wav'), k=PICKS)

    for f in picks:
        base, _ = os.path.splitext(f)
        #print('copy ' + DATA_DIR + d + base + ' to ' + newd)
        shutil.copy(base + '.wav', newd)
        shutil.copy(base + '.txt', newd)
