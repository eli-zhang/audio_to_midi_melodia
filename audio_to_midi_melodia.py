# CREATED: 11/9/15 3:57 PM by Justin Salamon <justin.salamon@nyu.edu>

import soundfile
import resampy
import vamp
import argparse
import os
import numpy as np
from midiutil.MidiFile import MIDIFile
from scipy.signal import medfilt
from scipy.signal import butter, sosfilt
from scipy import signal, interpolate
import jams
import __init__

from scipy.signal import argrelextrema

import matplotlib.pyplot as plt
from matplotlib import rcParams as defaults
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

def many_midi_to_notes(midi, fs, hop, smooth, minduration):
    midi_filt = midi    # ignore smoothing for now

    notes = []
    durations = {}
    onsets = {}
    for n, pitches in enumerate(midi_filt):
        for p in pitches:
            if p in durations:
                durations[p] += 1
            else:
                for (pitch, note_dur) in durations.items():
                    # treat 0 as silence
                    if pitch > 0:
                        # add note
                        duration_sec = note_dur * hop / float(fs)
                        # only add notes that are long enough
                        if duration_sec >= minduration:
                            onset_sec = onsets[pitch] * hop / float(fs)
                            notes.append((onset_sec, duration_sec, pitch))

                durations = {}
                onsets = {}

                # start new note
                onsets[p] = n
                durations[p] = 1

    # add last notes
    for (pitch, note_dur) in durations.items():
        # treat 0 as silence
        if pitch > 0:
            # add note
            duration_sec = note_dur * hop / float(fs)
            # only add notes that are long enough
            if duration_sec >= minduration:
                onset_sec = onsets[pitch] * hop / float(fs)
                notes.append((onset_sec, duration_sec, pitch))
    return notes


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

def hz2midi_many(hz):
    midi = [[] for _ in range(len(hz))]
    for i, frame in enumerate(hz):
        # convert from Hz to midi note
        hz_nonneg = np.array(frame)
        idx = hz_nonneg <= 0
        hz_nonneg[idx] = 1
        midi_frame = 69 + 12*np.log2(hz_nonneg/440.)
        midi_frame[idx] = 0
        midi[i] = np.round(midi_frame)

    return midi

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

# The following is from:
# https://crackedbassoon.com/writing/equal-loudness-contours
f = np.array(
    [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 
    800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500]
)

af = np.array(
    [0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330, 0.315, 
    0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244, 0.243, 0.243, 
    0.243, 0.242, 0.242, 0.245, 0.254, 0.271, 0.301]
)
Lu = np.array(
    [-31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5, 
    -3.1, -2.0, -1.1, -0.4, 0.0, 0.3, 0.5, 0.0, -2.7, -4.1, -1.0, 1.7, 
    2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1,]
)
Tf = np.array(
    [78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 
    11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6.0, 
    -5.4, -1.5, 6.0, 12.6, 13.9, 12.3]
)

def elc(phon, frequencies=None):
    """Returns an equal-loudness contour.

    Args:
        phon (float): Phon value of the contour.
        frequencies (:obj:`np.ndarray`, optional): Frequencies to evaluate. If not
            passed, all 29 points of the ISO standard are returned. Any frequencies not
            present in the standard are found via spline interpolation.

    Returns:
        contour (np.ndarray): db SPL values.

    """
    assert 0 <= phon <= 90, f"{phon} is not [0, 90]"
    Ln = phon
    Af = (
        4.47e-3 * (10 ** (0.025 * Ln) - 1.15)
        + (0.4 * 10 ** (((Tf + Lu) / 10) - 9)) ** af
    )
    Lp = ((10.0 / af) * np.log10(Af)) - Lu + 94

    if frequencies is not None:
        assert frequencies.min() >= f.min(), "Frequencies are too low"
        assert frequencies.max() <= f.max(), "Frequencies are too high"
        tck = interpolate.splrep(f, Lp, s=0)
        Lp = interpolate.splev(frequencies, tck, der=0)
    return Lp
    
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

    interval = int(len(data)/3)

    data = data[:interval] # Temporary: limit testing range

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
    M = 1000
    H = 100
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
        semitone_dists = np.abs(np.floor(1200 * np.log2(peaks / harmonics / 55) / 10 + 1) - bins) / 10
        # # alpha = 0.8

        # logical_or = np.logical_or(semitone_dists > 1, peaks <= 0)
        # cosine = np.cos(semitone_dists * np.pi / 2)

        # squared = cosine ** 2

        # product = squared * harmonic_weight

        # result = np.where(logical_or, 0, product)

        # return result
        return np.where(semitone_dists > 1, 0, np.cos(semitone_dists * np.pi / 2) ** 2 * harmonic_weight)

    # Finds the indices of magnitude peaks for time step (frame number) l at time_info[l] (which is Zxx.T[l])
    def find_peaks(l, time_info):
        return argrelextrema(np.abs(time_info[l]), np.greater)[0]

    def get_bin_offsets(l, time_info, peak_idxs):
        phi_l = np.angle(time_info[l][peak_idxs])
        phi_l_minus_one = np.zeros(phi_l.shape[0]) if l == 0 else np.angle(time_info[l - 1][peak_idxs])
        return  N / 2 / np.pi / H * (((phi_l - phi_l_minus_one - 2 * np.pi * H / N * peak_idxs) + np.pi) % (2 * np.pi) - np.pi)

    def bin_to_freq(bins):
        diff = bins * 10
        return np.where(bins == 0, 0, 55 * np.power(2, (diff / 1200)))

    b = np.arange(600) # bins

    # salience = np.zeros((len(t), len(b)))  # salience represents the likelihood that a frequency at a given time is the melody
    # N_h = 10
    # beta = 1
    # alpha = 0.8

    # for l in tqdm(range(len(t))): # for every time step
    #     time1 = time.time()
    #     peak_idxs = find_peaks(l, time_info)    # indices of frequency peaks
    #     time2 = time.time()
    #     b_offsets = get_bin_offsets(l, time_info, peak_idxs)
    #     time3 = time.time()
    #     hann_info = 1 / 2 * np.sinc(np.pi * b_offsets) / (1 - np.square(b_offsets))
    #     time4 = time.time()

    #     f_info = (peak_idxs + b_offsets) * fs / N # size = number of frequency peaks
    #     a_info = 1 / 2 * np.abs(time_info[l][peak_idxs]) / hann_info

    #     # we ignore invalid values for future processing
    #     invalid_idxs = np.logical_or(f_info <= 0, a_info <= 0)  
    #     valid_idxs = ~invalid_idxs
    #     peak_idxs = peak_idxs[valid_idxs]
    #     b_offsets = b_offsets[valid_idxs]
    #     hann_info = hann_info[valid_idxs]
    #     f_info = f_info[valid_idxs]
    #     a_info = a_info[valid_idxs]

    #     I = len(peak_idxs)

    #     time5 = time.time()
    #     bins = np.reshape(np.repeat(b, I * (N_h)), (len(b), N_h, I))
    #     harmonics = np.reshape(np.tile(np.repeat(np.arange(1, N_h + 1), I), len(b)), (len(b), N_h, I))
    #     peaks = np.reshape(np.tile(f_info, (len(b) * (N_h))), (len(b), N_h, I))

    #     time6 = time.time()

    #     # print("find peaks time {}".format(time2 - time1))
    #     # print("offset time {}".format(time3 - time2))
    #     # print("hann time {}".format(time4 - time3))
    #     # print("info time {}".format(time5 - time4))
    #     # print("list comp time {}".format(time6 - time5))

    #     harmonic_weight = np.power(alpha, harmonics - 1)

    #     # get_thresholds   # size: I (# of peaks)
    #     # get_weights      # size: f x (N_h) x I array

    #     if (len(a_info) == 0):
    #         salience[l] = np.zeros(len(b))
    #     else:
    #         # do some timing
    #         # t1 = time.time()
    #         # weight_info = get_weights(bins, harmonics, harmonic_weight, peaks)
    #         # t2 = time.time()
    #         # threshold_info = get_thresholds(a_info)[None, None, :]
    #         # t3 = time.time()
    #         # other_info = (a_info ** beta)[None, None, :]
    #         # t4 = time.time()
    #         # sum = np.sum(weight_info * threshold_info * other_info, axis=(1, 2))
    #         # t5 = time.time()
    #         # print("weight calc time {}".format(t2 - t1))
    #         # print("threshold calc time {}".format(t3 - t2))
    #         # print("other calc time {}".format(t4 - t3))
    #         # print("sum calc time {}".format(t5 - t4))
    #         # salience[l] = sum
    #         salience[l] = np.sum(get_weights(bins, harmonics, harmonic_weight, peaks) * get_thresholds(a_info)[None, None, :] * (a_info ** beta)[None, None, :], axis=(1, 2))
    #     # print(f[np.argmax(salience[l])])

    #     # salience_threshold = 0.1
    #     # if np.max(salience[l]) < salience_threshold:
    #     #     salience[l] = np.zeros(len(b))
    #     # print("Max magnitude: {}".format(np.max(salience[l])))

    # # debug
    
    # # apply equal loudness contour (ideally should be done before)
    # # volume_scale = elc(60, bin_to_freq(b)[1:])
    # # min_val = min(volume_scale)
    # # volume_scale = min_val / volume_scale
    # # volume_scale = np.insert(volume_scale, 0, 0, axis=0)
    
    # # salience = np.multiply(salience, volume_scale)

    

    # # Up to this point, you can already output a midi; just run 
    # # bins = np.array([np.argmax(salience[l]) for l in range(len(t))])
    # # pitch = bin_to_freq(bins)

    ## Everything afterwards is creating pitch contours

    salience = np.load('salience.npy', allow_pickle=True)
    

    salience_threshold = 0.9    # every value below (salience_threshold * max_val) in the frame is set to 0
    std_threshold = 0.9   # we can sweep this 

    max_vals = np.reshape(np.repeat(np.amax(salience, axis=1), len(b)), salience.shape)
    salience_filter = salience < max_vals * salience_threshold
    s_plus = np.where(salience_filter, 0, salience)   # filter out irrelevant values
    nonzero_salience = np.copy(s_plus)
    nonzero_salience[~salience_filter] = np.nan    # this helps ignore 0 values
    total_mean = np.nanmean(nonzero_salience)
    total_std = np.nanstd(nonzero_salience)
    deviation_filter = s_plus < (total_mean - total_std * std_threshold)
    
    s_plus = np.where(deviation_filter, 0, s_plus)   # filter out weaker peaks
    s_minus = np.where(np.logical_or(salience_filter, deviation_filter), salience, 0)

    bin_threshold = 8   # contours are maintained if the next frame has a note within 8 bins up/down
    allowed_contour_gap = 100  # contours may not always show up in s_plus, can go to s_minus for this many frames
    contours = [] # 2d list of lists of tuples

    # construct contours
    while np.sum(s_plus) > 0:
        contour = []
        highest_peak_idx = np.unravel_index(np.argmax(s_plus, axis=None), s_plus.shape)
        s_plus[highest_peak_idx] = 0
        curr_frame, curr_bin = highest_peak_idx
        contour.append((curr_frame, curr_bin))
        while curr_frame < (s_plus.shape[0] - 1):   # go forward in time to expand the contour
            curr_frame += 1
            found_bin = False
            new_bin = -1
            contour_gap_counter = 0 # tracks how long contour has not been in s_plus but still in s_minus
            for test_bin in range(curr_bin - bin_threshold, curr_bin + bin_threshold + 1):
                if test_bin > 0 and test_bin < len(b): # valid bin number
                    if s_plus[curr_frame][test_bin] > 0 and (not found_bin or s_plus[curr_frame][test_bin] > s_plus[curr_frame][new_bin]):
                        found_bin = True
                        new_bin = test_bin
                        contour_gap_counter = 0
            if not found_bin:   # now we check s_minus in case weaker contour continues
                for test_bin in range(curr_bin - bin_threshold, curr_bin + bin_threshold + 1):
                    if test_bin > 0 and test_bin < len(b):
                        if s_minus[curr_frame][test_bin] > 0 and (not found_bin or s_minus[curr_frame][test_bin] > s_minus[curr_frame][new_bin]):
                            found_bin = True
                            new_bin = test_bin
                            contour_gap_counter += 1
                if not found_bin or contour_gap_counter > allowed_contour_gap:
                    break
            else:
                s_plus[curr_frame][new_bin] = 0
                curr_bin = new_bin
                contour.append((curr_frame, curr_bin))  # append to end of contour
        while curr_frame > 0:   # go backwards in time to expand the contour
            curr_frame -= 1
            found_bin = False
            new_bin = -1
            for test_bin in range(curr_bin - bin_threshold, curr_bin + bin_threshold + 1):
                if test_bin > 0 and test_bin < len(b): # valid bin number
                    if s_plus[curr_frame][test_bin] > 0 and (not found_bin or s_plus[curr_frame][test_bin] > s_plus[curr_frame][new_bin]):
                        found_bin = True
                        new_bin = test_bin
                        contour_gap_counter = 0
            if not found_bin:   # now we check s_minus in case weaker contour continues
                for test_bin in range(curr_bin - bin_threshold, curr_bin + bin_threshold + 1):
                    if test_bin > 0 and test_bin < len(b):
                        if s_minus[curr_frame][test_bin] > 0 and (not found_bin or s_minus[curr_frame][test_bin] > s_minus[curr_frame][new_bin]):
                            found_bin = True
                            new_bin = test_bin
                            contour_gap_counter += 1
                if not found_bin or contour_gap_counter > allowed_contour_gap:
                    break
            else:
                s_plus[curr_frame][new_bin] = 0
                curr_bin = new_bin
                contour.insert(0, (curr_frame, curr_bin))   # prepend info to contour
        contours.append(contour)

    print("Done extracting contours")

    def get_mean(contour):
        return np.mean([salience[frame, bin] for (frame, bin) in contour])

    contour_means = [get_mean(contour) for contour in contours]  # mean salience of each contour
    distribution_mean = np.mean(contour_means)  # mean of means (mean mean salience)
    distribution_std = np.std(contour_means)

    filter_lenience = 0.2
    voicing_threshold = distribution_mean - filter_lenience * distribution_std

    # contour_means.sort()
    # print(contour_means)
    # print(distribution_mean)
    # print(distribution_std)
    # print(voicing_threshold)

    passing_contours = list(filter(lambda contour: get_mean(contour) >= voicing_threshold, contours))
    print(len(contours))
    print(len(passing_contours))
    
    final_saliences = np.zeros(salience.shape)
    for contour in passing_contours:
        for (frame, bin) in contour:
            final_saliences[frame][bin] = salience[frame][bin]

    # bins = np.array([np.argmax(final_saliences[l]) for l in range(len(t))])
    # pitch = bin_to_freq(bins)
    # np.asarray(salience).dump('salience.npy')
    

    def single_bin_to_freq(bin):
        diff = bin * 10
        return 0 if bin == 0 else (55 * 2 ** (diff / 1200))

    all_bins = [np.nonzero(salience_frame)[0] for salience_frame in final_saliences]
    # print([[single_bin_to_freq(bin) for bin in bin_frame] for bin_frame in all_bins])
    all_pitches = [[single_bin_to_freq(bin) for bin in bin_frame] for bin_frame in all_bins]

    midi_pitches = hz2midi_many(all_pitches)
    # midi_pitch = hz2midi(all_pitches)

    # convert f0 to midi notes
    print("Converting Hz to MIDI notes...")
    # midi_pitch = hz2midi(pitch)

    # segment sequence into individual midi notes
    notes = many_midi_to_notes(midi_pitches, fs, hop, smooth, minduration)

    # notes = midi_to_notes(midi_pitch, fs, hop, smooth, minduration)

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
