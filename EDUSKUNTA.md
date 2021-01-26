# Preparing Plenary Sessions of the Parliament of Finland dataset for use in speaker verification

The [Plenary Sessions of the Parliament of
Finland](http://urn.fi/urn:nbn:fi:lb-2017030901) dataset is a sizable corpus of
transcripted finnish audio. The transcriptions constitute of per word aligned
annotations in EAF-format files. To convert them for more convenient form for
use in speaker verification tasks it's necessary to group the annotations and
split the utterances from the larger WAV files.

## Caveats of the described process

1. Samples have not been verified for overlapping speech or misaligned
   timestamps.

2. Word grouping logic is quite rough and likely has fair bit of room for
   improvement.

3. Hashing speaker ids (tier ids) is currently quite frail, encoding and spaces
   affect results.

4. Some speakers have multiple tier ids due to additional prefixes (ministerial
   portfolio).

5. Due to 3. and 4. the speaker ids (tier ids) of the resulting files likely
   require manual tweaking.

6. There might be some finland swedish mixed in the audio.

# 1. Preparations

- Install `scripts/eaf-word2sentence` dependencies:
```shell
$ python -m venv word2sentence
$ source word2sentence/bin/activate
$ pip install pympi-ling
```

- Build `elan2split`, requires Boost.Filesystem and Xerces-C++:
TODO: This could be replaced with a simple iteration step in `scripts/eaf-word2sentence.py`.
```shell
$ git clone https://github.com/vjoki/ELAN2split
$ cd elan2split/
$ mkdir build/
$ cmake ../
$ make
```

# 2. Converting the eaf per word annotations to longer groups of words.

1. Unpack the dataset.

2. Iterate through the `.eaf` files:
```shell
$ source word2sentence/bin/activate
$ for eaf in 2016-kevat/2016-*/*.eaf; do
    python scripts/eaf-word2sentence.py --file_path "$eaf"
    elan2split --name -o ./eduskunta/ "$eaf"
  done
```

3. Organize the files into directories per speaker with `scripts/organize.sh`.

4. Optionally take a subset of the dataset using `scripts/sample_dataset.py`.

4. Manually fix any issues with tier ids.
