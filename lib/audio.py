import pathlib
import os
import tempfile

import ffmpeg


class AudioPreProcessor:
    def __init__(self):
        self.tmpdir = None
        self.output_path = None
        self.error = None

    def process(self, audio_file):
        # create a new temp dir for every run
        self.tmpdir = pathlib.Path(tempfile.mkdtemp())
        self.output_path = str(self.tmpdir / 'audio.wav')
        self.error = None

        # converts audio file to 16kHz 16bit mono wav...
        print('pre-processing audio file...')
        stream = ffmpeg.input(audio_file, vn=None, hide_banner=None)
        stream = stream.output(self.output_path, format='wav',
                               acodec='pcm_s16le', ac=1, ar='16k').overwrite_output()
        try:
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            self.error = e.stderr.decode('utf8')

    def cleanup(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        if self.tmpdir:
            self.tmpdir.rmdir()
