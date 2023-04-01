# speaker-transcription

This repository contains the Cog definition files for the associated speaker transcription model [deployed on Replicate](https://replicate.com/meronym/speaker-transcription).

The pipeline transcribes the speech segments of an audio file, identifies the individual speakers and annotates the transcript with timestamps and speaker labels. The pipeline outputs additional global information about the number of detected speakers and an embedding vector for each speaker to describe the quality of their voice.

## Model description

There are two main components involved in this process:

- a pre-trained speaker diarization pipeline from the [`pyannote.audio`](pyannote.github.io) package (also available as a [stand-alone diarization model](https://replicate.com/meronym/speaker-diarization) without transcription):

  - `pyannote/segmentation` for permutation-invariant speaker segmentation on temporal slices
  - `speechbrain/spkrec-ecapa-voxceleb` for generating speaker embeddings
  - `AgglomerativeClustering` for matching embeddings across temporal slices

- OpenAI's `whisper` model for general-purpose speech transcription:
  - the `medium` model size is used for a good balance between accuracy and performance

The audio data is first passed in to the speaker diarization pipeline, which computes a list of time segments and their associated speakers. Each segment is then transcribed using `whisper`.

## Input format

The model uses `ffmpeg` to decode the input audio, so it supports a wide variety of input formats - including, but not limited to `mp3`, `aac`, `flac`, `ogg`, `opus`, `wav`.

## Output format

The model outputs a single `output.json` file with the following structure:

```json
{
  "segments": [
    {
      "speaker": "A",
      "start": "0:00:00.497812",
      "stop": "0:00:09.779063",
    },
    {
      "speaker": "B",
      "start": "0:00:09.863438",
      "stop": "0:03:34.962188"
    }
  ],
  "speakers": {
    "count": 2,
    "labels": [
      "A",
      "B"
    ],
    "embeddings": {
      "A": [<array of 192 floats>],
      "B": [<array of 192 floats>]
    }
  }
}
```

## Performance

The current T4 deployment has an average processing speed factor of 12x (relative to the length of the audio input) - e.g. it will take the model approx. 1 minute of computation to process 12 minutes of audio.

## Intended use

Data augmentation and segmentation for a variety of transcription and captioning tasks (e.g. interviews, podcasts, meeting recordings, etc.). Speaker recognition can be implemented by matching the speaker embeddings against a database of known speakers.

## Ethical considerations

This model may have biases based on the data it has been trained on. It is important to use the model in a responsible manner and adhere to ethical and legal standards.

## Citations

If you use `pyannote.audio` please use the following citations:

```bibtex
@inproceedings{Bredin2020,
  Title = {{pyannote.audio: neural building blocks for speaker diarization}},
  Author = {{Bredin}, Herv{\'e} and {Yin}, Ruiqing and {Coria}, Juan Manuel and {Gelly}, Gregory and {Korshunov}, Pavel and {Lavechin}, Marvin and {Fustes}, Diego and {Titeux}, Hadrien and {Bouaziz}, Wassim and {Gill}, Marie-Philippe},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Year = {2020},
}
```

```bibtex
@inproceedings{Bredin2021,
  Title = {{End-to-end speaker segmentation for overlap-aware resegmentation}},
  Author = {{Bredin}, Herv{\'e} and {Laurent}, Antoine},
  Booktitle = {Proc. Interspeech 2021},
  Year = {2021},
}
```
