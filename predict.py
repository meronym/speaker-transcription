"""
download model weights to /data
wget -O - https://pyannote-speaker-diarization.s3.eu-west-2.amazonaws.com/data-2023-03-25-02.tar.gz | tar xz -C /
wget -P /data/whisper https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt
# wget -P /data/whisper https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt
"""

import json
import tempfile
from typing import Optional

import numpy as np
import torch

from cog import BasePredictor, Input, Path
from pyannote.audio import Audio
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment
from whisper.model import Whisper, ModelDimensions

from lib.diarization import DiarizationPostProcessor, format_ts
from lib.audio import AudioPreProcessor


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.audio_pre = AudioPreProcessor()

        self.diarization = SpeakerDiarization(
            segmentation="/data/pyannote/segmentation/pytorch_model.bin",
            embedding="/data/speechbrain/spkrec-ecapa-voxceleb",
            clustering="AgglomerativeClustering",
            segmentation_batch_size=32,
            embedding_batch_size=32,
            embedding_exclude_overlap=True,
        )
        self.diarization.instantiate({
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 15,
                "threshold": 0.7153814381597874,
            },
            "segmentation": {
                "min_duration_off": 0.5817029604921046,
                "threshold": 0.4442333667381752,
            },
        })
        self.diarization_post = DiarizationPostProcessor()

        with open(f"/data/whisper/medium.en.pt", "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
            dims = ModelDimensions(**checkpoint["dims"])
        self.whisper = Whisper(dims)
        self.whisper.load_state_dict(checkpoint["model_state_dict"])

    def run_diarization(self):
        closure = {'embeddings': None}

        def hook(name, *args, **kwargs):
            if name == "embeddings" and len(args) > 0:
                closure['embeddings'] = args[0]

        print('diarizing audio file...')
        diarization = self.diarization(self.audio_pre.output_path, hook=hook)
        embeddings = {
            'data': closure['embeddings'],
            'chunk_duration': self.diarization.segmentation_duration,
            'chunk_offset': self.diarization.segmentation_step * self.diarization.segmentation_duration,
        }
        return self.diarization_post.process(diarization, embeddings)

    def run_transcription(self, audio, segments, whisper_prompt):
        print('transcribing segments...')
        self.whisper.to("cuda")
        trimmer = Audio(sample_rate=16000, mono=True)
        for seg in segments:
            start = seg['start']
            stop = seg['stop']
            print(
                f"transcribing segment {format_ts(start)} to {format_ts(stop)}")
            frames, _ = trimmer.crop(audio, Segment(start, stop))
            # audio data was already downmixed to mono, so exract the first (only) channel
            frames = frames[0]
            seg['transcript'] = self.transcribe_segment(frames, start, whisper_prompt)

    def transcribe_segment(self, audio, ctx_start, whisper_prompt):
        # `temperature`: temperature to use for sampling
        # `temperature_increment_on_fallback``: temperature to increase when
        # falling back when the decoding fails to meet either of the thresholds below
        temperature = 0
        temperature_increment_on_fallback = 0.2
        temperature = tuple(
            np.arange(temperature, 1.0 + 1e-6,
                      temperature_increment_on_fallback))

        # `patience`: "optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424
        # the default (1.0) is equivalent to conventional beam search
        patience = None

        # `suppress_tokens`: "comma-separated list of token ids to suppress during sampling
        # '-1' will suppress most special characters except common punctuations
        suppress_tokens = "-1"

        # `initial_prompt`: optional text to provide as a prompt for the first window
        initial_prompt = whisper_prompt

        # `condition_on_previous_text`: if True, provide the previous output of the model
        # as a prompt for the next window; disabling may make the text inconsistent across windows,
        # but the model becomes less prone to getting stuck in a failure loop
        condition_on_previous_text = True

        # `compression_ratio_threshold`: if the gzip compression ratio is higher than this value,
        # treat the decoding as failed
        compression_ratio_threshold = 2.4

        # `logprob_threshold`: if the average log probability is lower than this value,
        # treat the decoding as failed
        logprob_threshold = -1.0

        # `no_speech_threshold`: if the probability of the <|nospeech|> token is higher than this value
        # AND the decoding has failed due to `logprob_threshold`, consider the segment as silence
        no_speech_threshold = 0.6

        args = {
            "language": "en",  # this is an English-only model
            "patience": patience,
            "suppress_tokens": suppress_tokens,
            "initial_prompt": initial_prompt,
            "condition_on_previous_text": condition_on_previous_text,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
        }
        trs = self.whisper.transcribe(
            audio, temperature=temperature, **args)

        result = []
        for s in trs.get('segments', []):
            timestamp = ctx_start + s['start']
            result.append({
                'start': format_ts(timestamp),
                'text': s['text']
            })
        return result

    def predict(
        self,
        audio: Path = Input(description="Audio file"),
        whisper_prompt: Optional[str] = Input(
            default=None,
            description="Optional text to provide as a prompt for each Whisper model call.",
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        self.audio_pre.process(audio)

        if self.audio_pre.error:
            print(self.audio_pre.error)
            result = self.diarization_post.empty_result()
        else:
            result = self.run_diarization()

        # transcribe segments
        self.run_transcription(self.audio_pre.output_path, result["segments"], whisper_prompt)

        # format segments
        result["segments"] = self.diarization_post.format_segments(
            result["segments"])

        # cleanup
        self.audio_pre.cleanup()

        # write output
        output = Path(tempfile.mkdtemp()) / "output.json"
        with open(output, "w") as f:
            f.write(json.dumps(result, indent=2))
        return output
