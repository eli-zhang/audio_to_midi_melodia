# Melody Extraction (based on Melodia algorithm)
_This fork uses a custom Python-only implementation (mainly numpy and some scipy) of the Melodia algorithm. It is based on the following [research paper](https://repositori.upf.edu/bitstream/handle/10230/42183/Gomez_iee_melo.pdf?sequence=1&isAllowed=y), but has more naive contour extraction and takes longer to run (although a lot of numpy operations are vectorized). It also can run on Python3. The original intention of this implementation was to be able to run melody extraction without any external dependencies (like the Vamp plugin) that didn't work on new Apple Silicon macs. It has only been tested on an M1 Macbook Air and mainly used with vocals extracted using [spleeter](https://github.com/deezer/spleeter)._

The script extracts the melody from an audio file using the [Melodia](http://mtg.upf.edu/technologies/melodia) algorithm, and then segments the continuous pitch sequence into a series of quantized notes, and exports to MIDI using the provided BPM. If the `--jams` option is specified the script will also save the output as a JAMS file. Note that the JAMS file uses the original note onset/offset times estimated by the algorithm and ignores the provided BPM value.

Note: extracting a MIDI melody from a polyphonic audio file involves two main steps: 
1. melody extraction 
2. note segmentation. 

**Melody extraction** is the task of estimating the continuous fundamental frequency (f0) of the melody from a polyphonic audio recording. This is achieved using the Melodia melody extraction algorithm, which is the result of [several years of research](http://www.justinsalamon.com/phd-thesis.html). 

**Note segmentation** is the task of converting the continuous f0 curve estimated by Melodia (which can contain e.g. glissando and vibrato) into a sequence of quantized notes each with a start time, end time, and fixed pitch value. **Unlike Melodia, the note segmentation code used here was written during a single-day hackathon** and designed to be as simple as possible. Peformance will vary depending on musical content, and it will most likely not provide results that are as good as those provided by state-of-the-art note segmentation/quantization algorithms.

# Usage
```bash
>python audio_to_midi_melodia.py infile outfile bpm [--smooth SMOOTH] [--minduration MINDURATION] [--jams]
```
For example:
```bash
>python audio_to_midi_melodia.py ~/song.wav ~/song.mid 60 --smooth 0.25 --minduration 0.1 --jams
```
Details:
```
usage: audio_to_midi_melodia.py [-h] [--smooth SMOOTH]
                                [--minduration MINDURATION] [--jams]
                                infile outfile bpm

positional arguments:
  infile                Path to input audio file.
  outfile               Path for saving output MIDI file.
  bpm                   Tempo of the track in BPM.

optional arguments:
  -h, --help            show this help message and exit
  --smooth SMOOTH       Smooth the pitch sequence with a median filter of the
                        provided duration (in seconds).
  --minduration MINDURATION
                        Minimum allowed duration for note (in seconds).
                        Shorter notes will be removed.
  --jams                Also save output in JAMS format.
```

# Installation
**Windows users:** if you run into any installation issues please [read and try the solutions on this thread](https://github.com/justinsalamon/audio_to_midi_melodia/issues/4) before posting an issue, thanks!

### Python dependencies
This program requires Python 3.

All python dependencies (listed below) can be installed by calling `pip install -r requirements.txt`.
- soundfile: https://pypi.org/project/SoundFile/
- resampy: https://pypi.org/project/resampy/
- midiutil: https://pypi.org/project/MIDIUtil/
- jams: https://pypi.org/project/jams/
- numpy: https://pypi.org/project/numpy/
- scipy: https://pypi.org/project/scipy/

Known to work with the following module versions on python 3.9.7:
- SoundFile==0.10.3.post1
- resampy==0.2.2
- MIDIUtil==1.2.1
- jams==0.3.4
- numpy==1.20.0
- scipy==1.7.3
