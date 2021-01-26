# Script for grouping per word annotations in an EAF file into sentences or longer groups of words.
# Intended for converting Plenary Sessions of the Parliament of Finland, Downloadable Version 1 EAF
# files into suitable form for use in speaker verification testing.
#
# Replaces the original EAF file (original is moved out of the way by appending ".bak" to the filename).
import os
import argparse
import pympi.Elan as e

MIN_DURATION = 3000
MAX_DURATION = 25000

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str)
args = parser.parse_args()

eaf = e.Eaf(args.file_path)

# Dataset MEDIA_DESCRIPTORs only have MEDIA_URL defined to the original mp4 video,
# however we need path to the wav files for the extraction step.
linked_path, ext = os.path.splitext(args.file_path)
for linked in eaf.get_linked_files():
    eaf.remove_linked_files(linked['MEDIA_URL'])
eaf.add_linked_file(linked_path + '.wav',
                    relpath=os.path.basename(linked_path) + '.wav',
                    mimetype='audio/wav')

# Try to group word annotations into sentences, only keep sentences < MAX_DURATION.
for tid, (anno, _, _, _) in eaf.tiers.items():
    utterances = []
    utterance = []
    utterance_str = []
    utterance_start = 0

    for aid, (start_ts, end_ts, val, _) in anno.items():
        utterance.append(aid)
        utterance_str.append(val)
        if utterance_start == 0:
            utterance_start = eaf.timeslots[start_ts]

        duration = eaf.timeslots[end_ts] - utterance_start

        if (val.rstrip().endswith('.') and duration > MIN_DURATION) \
           or duration > MAX_DURATION*0.8:
            value = ' '.join(utterance_str)

            if duration < MAX_DURATION:
                utterances.append((utterance, utterance_start,
                                   eaf.timeslots[end_ts], value))
                # print('added {}s [{}]: {}'.format(duration/1000,
                #                                   tid, value))
            else:
                print('skipped {}s [{}]: {}'.format(duration/1000,
                                                    tid, value))

            utterance = []
            utterance_start = 0
            utterance_str = []

    eaf.remove_all_annotations_from_tier(tid)

    for (aid, start, end, val) in utterances:
        eaf.add_annotation(tid, start, end, val)

print('Collected {} annotations for {} speakers.'
      .format(len(eaf.annotations), len(eaf.tiers)))
eaf.to_file(args.file_path)
