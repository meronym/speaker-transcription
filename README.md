# speaker-transcription

This repository contains the Cog definition files for the associated speaker transcription model [deployed on Replicate](https://replicate.com/meronym/speaker-transcription).

The pipeline transcribes the speech segments of an audio file, identifies the individual speakers and annotates the transcript with timestamps and speaker labels. An optional `prompt` string can guide the transcription by providing additional context. The pipeline outputs additional global information about the number of detected speakers and an embedding vector for each speaker to describe the quality of their voice.

## Model description

There are two main components involved in this process:

- a pre-trained speaker diarization pipeline from the [`pyannote.audio`](pyannote.github.io) package (also available as a [stand-alone diarization model](https://replicate.com/meronym/speaker-diarization) without transcription):

  - `pyannote/segmentation` for permutation-invariant speaker segmentation on temporal slices
  - `speechbrain/spkrec-ecapa-voxceleb` for generating speaker embeddings
  - `AgglomerativeClustering` for matching embeddings across temporal slices

- OpenAI's `whisper` model for general-purpose English speech transcription (the `medium.en` model size is used for a good balance between accuracy and performance).

The audio data is first passed in to the speaker diarization pipeline, which computes a list of timestamped segments and associates each segment with a speaker. The segments are then transcribed with `whisper`.

## Input format

The pipeline uses `ffmpeg` to decode the input audio, so it supports a wide variety of input formats - including, but not limited to `mp3`, `aac`, `flac`, `ogg`, `opus`, `wav`.

The `prompt` string gets injected as (off-screen) additional context at the beginning of the first Whisper transcription window for each segment. It won't be part of the final output, but it can be used for guiding/conditioning the transcription towards a specific domain.

## Output format

The pipeline outputs a single `output.json` file with the following structure:

```json
{
  "segments": [
    {
      "speaker": "A",
      "start": "0:00:00.497812",
      "stop": "0:00:09.762188",
      "transcript": [
        {
          "start": "0:00:00.497812",
          "text": " What are some cool synthetic organisms that you think about, you dream about?"
        },
        {
          "start": "0:00:04.357812",
          "text": " When you think about embodied mind, what do you imagine?"
        },
        {
          "start": "0:00:08.017812",
          "text": " What do you hope to build?"
        }
      ]
    },
    {
      "speaker": "B",
      "start": "0:00:09.863438",
      "stop": "0:03:34.962188",
      "transcript": [
        {
          "start": "0:00:09.863438",
          "text": " Yeah, on a practical level, what I really hope to do is to gain enough of an understanding of the embodied intelligence of the organs and tissues, such that we can achieve a radically different regenerative medicine, so that we can say, basically, and I think about it as, you know, in terms of like, okay, can you what's the what's the what's the goal, kind of end game for this whole thing? To me, the end game is something that you would call an"
        },
        {
          "start": "0:00:39.463438",
          "text": " anatomical compiler. So the idea is you would sit down in front of the computer and you would draw the body or the organ that you wanted. Not molecular details, but like, here, this is what I want. I want a six legged, you know, frog with a propeller on top, or I want I want a heart that looks like this, or I want a leg that looks like this. And what it would do if we knew what we were doing is put out, convert that anatomical description into a set of stimuli that would have to be given to cells to convince them to build exactly that thing."
        },
        {
          "start": "0:01:08.503438",
          "text": " Right? I probably won't live to see it. But I think it's achievable. And I think what that if, if we can have that, then that is basically the solution to all of medicine, except for infectious disease. So birth defects, right, traumatic injury, cancer, aging, degenerative disease, if we knew how to tell cells what to build, all of those things go away. So those things go away, and the positive feedback spiral of economic costs, where all of the advances are increasingly more"
        },
      ]
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

The current T4 deployment has an average processing speed factor of 4x (relative to the length of the audio input) - e.g. it will take the model approx. 1 minute of computation to process 4 minutes of audio.

## Intended use

Data augmentation and segmentation for a variety of transcription and captioning tasks (e.g. interviews, podcasts, meeting recordings, etc.). Speaker recognition can be implemented by matching the speaker embeddings against a database of known speakers.

## Ethical considerations

This model may have biases based on the data it has been trained on. It is important to use the model in a responsible manner and adhere to ethical and legal standards.

## Citations

For `pyannote.audio`:

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

For OpenAI `whisper`:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2212.04356,
  doi = {10.48550/ARXIV.2212.04356},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  keywords = {Audio and Speech Processing (eess.AS), Computation and Language (cs.CL), Machine Learning (cs.LG), Sound (cs.SD), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
