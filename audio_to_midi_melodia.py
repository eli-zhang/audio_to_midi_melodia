# CREATED: 11/9/15 3:57 PM by Justin Salamon <justin.salamon@nyu.edu>

import soundfile
import resampy
import vamp
import argparse
import os
import numpy as np
from midiutil.MidiFile import MIDIFile
from scipy.signal import medfilt
from scipy.signal import butter, sosfilt, sosfreqz
from scipy import signal
import jams
import __init__

from scipy.signal import argrelextrema

import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import time


'''
Extract the melody from an audio file and convert it to MIDI.

The script extracts the melody from an audio file using the Melodia algorithm,
and then segments the continuous pitch sequence into a series of quantized
notes, and exports to MIDI using the provided BPM. If the --jams option is
specified the script will also save the output as a JAMS file. Note that the
JAMS file uses the original note onset/offset times estimated by the algorithm
and ignores the provided BPM value.

Note: Melodia can work pretty well and is the result of several years of
research. The note segmentation/quantization code was hacked in about 30
minutes. Proceed at your own risk... :)

usage: audio_to_midi_melodia.py [-h] [--smooth SMOOTH]
                                [--minduration MINDURATION] [--jams]
                                infile outfile bpm


Examples:
python audio_to_midi_melodia.py --smooth 0.25 --minduration 0.1 --jams
                                ~/song.wav ~/song.mid 60
'''


def save_jams(jamsfile, notes, track_duration, orig_filename):

    # Construct a new JAMS object and annotation records
    jam = jams.JAMS()

    # Store the track duration
    jam.file_metadata.duration = track_duration
    jam.file_metadata.title = orig_filename

    midi_an = jams.Annotation(namespace='pitch_midi',
                              duration=track_duration)
    midi_an.annotation_metadata = \
        jams.AnnotationMetadata(
            data_source='audio_to_midi_melodia.py v%s' % __init__.__version__,
            annotation_tools='audio_to_midi_melodia.py (https://github.com/'
                             'justinsalamon/audio_to_midi_melodia)')

    # Add midi notes to the annotation record.
    for n in notes:
        midi_an.append(time=n[0], duration=n[1], value=n[2], confidence=0)

    # Store the new annotation in the jam
    jam.annotations.append(midi_an)

    # Save to disk
    jam.save(jamsfile)


def save_midi(outfile, notes, tempo):

    track = 0
    time = 0
    midifile = MIDIFile(1)

    # Add track name and tempo.
    midifile.addTrackName(track, time, "MIDI TRACK")
    midifile.addTempo(track, time, tempo)

    channel = 0
    volume = 100

    for note in notes:
        onset = note[0] * (tempo/60.)
        duration = note[1] * (tempo/60.)
        # duration = 1
        pitch = note[2].__int__()
        midifile.addNote(track, channel, pitch, onset, duration, volume)

    # And write it to disk.
    binfile = open(outfile, 'wb')
    midifile.writeFile(binfile)
    binfile.close()


def midi_to_notes(midi, fs, hop, smooth, minduration):

    # smooth midi pitch sequence first
    if (smooth > 0):
        filter_duration = smooth  # in seconds
        filter_size = int(filter_duration * fs / float(hop))
        if filter_size % 2 == 0:
            filter_size += 1
        midi_filt = medfilt(midi, filter_size)
    else:
        midi_filt = midi
    # print(len(midi),len(midi_filt))

    notes = []
    p_prev = 0
    duration = 0
    onset = 0
    for n, p in enumerate(midi_filt):
        if p == p_prev:
            duration += 1
        else:
            # treat 0 as silence
            if p_prev > 0:
                # add note
                duration_sec = duration * hop / float(fs)
                # only add notes that are long enough
                if duration_sec >= minduration:
                    onset_sec = onset * hop / float(fs)
                    notes.append((onset_sec, duration_sec, p_prev))

            # start new note
            onset = n
            duration = 1
            p_prev = p

    # add last note
    if p_prev > 0:
        # add note
        duration_sec = duration * hop / float(fs)
        onset_sec = onset * hop / float(fs)
        notes.append((onset_sec, duration_sec, p_prev))

    return notes


def hz2midi(hz):

    # convert from Hz to midi note
    hz_nonneg = hz.copy()
    idx = hz_nonneg <= 0
    hz_nonneg[idx] = 1
    midi = 69 + 12*np.log2(hz_nonneg/440.)
    midi[idx] = 0

    # round
    midi = np.round(midi)

    return midi


def audio_to_midi_melodia(infile, outfile, bpm, smooth=0.25, minduration=0.1,
                          savejams=False):

    # define analysis parameters
    fs = 6000 # Usually 44100, but the vocal range of humans only goes to around ~3000 Hz and we only care about the melody
    hop = 128

    # load audio using librosa
    print("Loading audio...")
    data, sr = soundfile.read(infile)
    # mixdown to mono if needed
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = data.mean(axis=1)
    # resample to 44100 if needed
    if sr != fs:
        data = resampy.resample(data, sr, fs)
        sr = fs

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(suppress=True)

    # interval = int(len(data)/3)

    # data = data[:interval] # Temporary: limit testing range

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

    data = butter_bandpass_filter(data, 50, 2000, fs, order=2)  # Run data through a bandpass filter to only include melodies within potential singing vocal range

    # M = 2048 # Window length
    # H = 128 # H is hopsize (defaults to nperseg - noverlap), used later
    # N = 8192 # Length of FFT used (defaults to nperseg)
    M = 800
    H = 128
    N = 1000
    noverlap = M - H
    nperseg = M
    
    f, t, Zxx = signal.stft(data, fs, nperseg=nperseg, noverlap=noverlap, nfft=N)
    print("Finished fetching FFT")

    time_info = Zxx.T   # we use the transpose, which is t x f

    def get_thresholds(a_info):
        gamma = 40
        a_m = np.amax(a_info)
        return ((20 * np.log10(a_m / a_info)) < gamma).astype(int)

    def get_weights(bins, harmonics, harmonic_weight, peaks):  # tuples is an f x (N_h) x I array
        # Need to find the weight for all weight(b, h, f_info[i])
        # t1 = time.time()
        semitone_dists = np.abs(np.floor(1200 * np.log2(peaks / harmonics / 55) / 10 + 1) - bins) / 10
        # t3 = time.time()
        # # alpha = 0.8

        # t4 = time.time()
        # logical_or = np.logical_or(semitone_dists > 1, peaks <= 0)
        # t5 = time.time()
        # cosine = np.cos(semitone_dists * np.pi / 2)
        # t6 = time.time()

        # squared = cosine ** 2

        # t8 = time.time()

        # product = squared * harmonic_weight
        # t9 = time.time()

        # result = np.where(logical_or, 0, product)
        # t10 = time.time()

        # print("positive_peaks calc time {}".format(t2 - t1))
        # print("semitone_dist calc time {}".format(t3 - t2))
        # print("or calc time {}".format(t5 - t4))
        # print("cosine calc time {}".format(t6 - t5))
        # print("squared calc time {}".format(t8 - t6))
        # print("product calc time {}".format(t9 - t8))
        # print("where calc time {}".format(t10 - t9))

        # return result
        return np.where(semitone_dists > 1, 0, np.cos(semitone_dists * np.pi / 2) ** 2 * harmonic_weight)

    # Finds the indices of magnitude peaks for time step (frame number) l at time_info[l] (which is Zxx.T[l])
    def find_peaks(l, time_info):
        return argrelextrema(np.abs(time_info[l]), np.greater)[0]

    def get_bin_offsets(l, time_info, peak_idxs):
        phi_l = np.angle(time_info[l][peak_idxs])
        phi_l_minus_one = np.zeros(phi_l.shape[0]) if l == 0 else np.angle(time_info[l - 1][peak_idxs])
        return  N / 2 / np.pi / H * (((phi_l - phi_l_minus_one - 2 * np.pi * H / N * peak_idxs) + np.pi) % (2 * np.pi) - np.pi)

    b = np.arange(1, 601) # bins
    salience = np.zeros((len(t), len(b)))  # salience represents the likelihood that a frequency at a given time is the melody
    N_h = 10
    beta = 1
    alpha = 0.8

    for l in tqdm(range(len(t))): # for every time step
        time1 = time.time()
        peak_idxs = find_peaks(l, time_info)    # indices of frequency peaks
        time2 = time.time()
        b_offsets = get_bin_offsets(l, time_info, peak_idxs)
        time3 = time.time()
        hann_info = 1 / 2 * np.sinc(np.pi * b_offsets) / (1 - np.square(b_offsets))
        time4 = time.time()

        f_info = (peak_idxs + b_offsets) * fs / N # size = number of frequency peaks
        a_info = 1 / 2 * np.abs(time_info[l][peak_idxs]) / hann_info

        # we ignore invalid values for future processing
        invalid_idxs = np.logical_or(f_info <= 0, a_info <= 0)  
        valid_idxs = ~invalid_idxs
        peak_idxs = peak_idxs[valid_idxs]
        b_offsets = b_offsets[valid_idxs]
        hann_info = hann_info[valid_idxs]
        f_info = f_info[valid_idxs]
        a_info = a_info[valid_idxs]

        I = len(peak_idxs)

        time5 = time.time()
        bins = np.reshape(np.repeat(b, I * (N_h)), (len(b), N_h, I))
        harmonics = np.reshape(np.tile(np.repeat(np.arange(1, N_h + 1), I), len(b)), (len(b), N_h, I))
        peaks = np.reshape(np.tile(f_info, (len(b) * (N_h))), (len(b), N_h, I))

        time6 = time.time()

        # print("find peaks time {}".format(time2 - time1))
        # print("offset time {}".format(time3 - time2))
        # print("hann time {}".format(time4 - time3))
        # print("info time {}".format(time5 - time4))
        # print("list comp time {}".format(time6 - time5))

        harmonic_weight = np.power(alpha, harmonics - 1)

        # get_thresholds   # size: I (# of peaks)
        # get_weights      # size: f x (N_h) x I array

        if (len(a_info) == 0):
            salience[l] = np.zeros(len(b))
        else:
            # do some timing
            # t1 = time.time()
            # weight_info = get_weights(bins, harmonics, harmonic_weight, peaks)
            # t2 = time.time()
            # threshold_info = get_thresholds(a_info)[None, None, :]
            # t3 = time.time()
            # other_info = (a_info ** beta)[None, None, :]
            # t4 = time.time()
            # sum = np.sum(weight_info * threshold_info * other_info, axis=(1, 2))
            # t5 = time.time()
            # print("weight calc time {}".format(t2 - t1))
            # print("threshold calc time {}".format(t3 - t2))
            # print("other calc time {}".format(t4 - t3))
            # print("sum calc time {}".format(t5 - t4))
            # salience[l] = sum
            salience[l] = np.sum(get_weights(bins, harmonics, harmonic_weight, peaks) * get_thresholds(a_info)[None, None, :] * (a_info ** beta)[None, None, :], axis=(1, 2))
        # print(f[np.argmax(salience[l])])

        salience_threshold = 0.1
        if np.max(salience[l]) < salience_threshold:
            salience[l] = np.zeros(len(b))
        # else:
            # print(salience[l])
            # print(np.argmax(salience[l]))
        # print("Max magnitude: {}".format(np.max(salience[l])))

    # debug
    def bin_to_freq(bins):
        diff = bins * 10
        return np.where(bins == 0, 0, 55 * np.power(2, (diff / 1200)))
    
    bins = np.array([np.argmax(salience[l]) for l in range(len(t))])
    pitch = bin_to_freq(bins)
    np.asarray(salience).dump('salience.npy')

    # convert f0 to midi notes
    print("Converting Hz to MIDI notes...")
    midi_pitch = hz2midi(pitch)

    # segment sequence into individual midi notes
    notes = midi_to_notes(midi_pitch, fs, hop, smooth, minduration)

    # save note sequence to a midi file
    print("Saving MIDI to disk...")
    save_midi(outfile, notes, bpm)

    if savejams:
        print("Saving JAMS to disk...")
        jamsfile = os.path.splitext(outfile)[0] + ".jams"
        track_duration = len(data) / float(fs)
        save_jams(jamsfile, notes, track_duration, os.path.basename(infile))

    print("Conversion complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="Path to input audio file.")
    parser.add_argument("outfile", help="Path for saving output MIDI file.")
    parser.add_argument("bpm", type=int, help="Tempo of the track in BPM.")
    parser.add_argument("--smooth", type=float, default=0.25,
                        help="Smooth the pitch sequence with a median filter "
                             "of the provided duration (in seconds).")
    parser.add_argument("--minduration", type=float, default=0.1,
                        help="Minimum allowed duration for note (in seconds). "
                             "Shorter notes will be removed.")
    parser.add_argument("--jams", action="store_const", const=True,
                        default=False, help="Also save output in JAMS format.")

    args = parser.parse_args()

    audio_to_midi_melodia(args.infile, args.outfile, args.bpm,
                          smooth=args.smooth, minduration=args.minduration,
                          savejams=args.jams)
