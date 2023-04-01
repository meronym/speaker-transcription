import collections
import datetime

import numpy as np


def format_ts(ts):
    return str(datetime.timedelta(seconds=ts))


class SpeakerLabelGenerator:
    def __init__(self):
        self.speakers = {}
        self.labels = []
        self.next_speaker = ord('A')
        self.count = 0

    def get(self, name):
        if name not in self.speakers:
            current = chr(self.next_speaker)
            self.speakers[name] = current
            self.labels.append(current)
            self.next_speaker += 1
            self.count += 1
        return self.speakers[name]

    def get_all(self):
        return self.labels


class DiarizationPostProcessor:
    def __init__(self):
        self.MIN_SEGMENT_DURATION = 1.0
        self.labels = None

    def process(self, diarization, embeddings):
        print('post-processing diarization...')
        # create a new label generator
        self.labels = SpeakerLabelGenerator()

        # process the diarization
        clean_segments = self.clean_segments(diarization)
        merged_segments = self.merge_segments(clean_segments)
        emb_segments = self.segment_embeddings(merged_segments, embeddings)

        # create the speaker embeddings
        speaker_embeddings = self.create_speaker_embeddings(emb_segments)
        speaker_count = self.labels.count
        speaker_labels = self.labels.get_all()
        speaker_emb_map = {}
        for label in speaker_labels:
            speaker_emb_map[label] = speaker_embeddings[label].tolist()

        # create the final output
        # segments = self.format_segments(emb_segments)
        # segments = self.format_segments_extra(emb_segments, speaker_embeddings)
        return {
            "segments": emb_segments,
            "speakers": {
                "count": speaker_count,
                "labels": speaker_labels,
                "embeddings": speaker_emb_map,
            },
        }

    def empty_result(self):
        return {
            "segments": [],
            "speakers": {
                "count": 0,
                "labels": [],
                "embeddings": {},
            },
        }

    def clean_segments(self, diarization):
        speaker_time = collections.defaultdict(float)
        total_time = 0.0
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            # filter out segments that are too short
            if segment.duration < self.MIN_SEGMENT_DURATION:
                continue
            speaker_time[speaker] += segment.duration
            total_time += segment.duration

        # filter out speakers that have spoken too little
        # (these are likely overlaps misclassified as separate speakers)
        speakers = set([
            speaker
            for speaker, time in speaker_time.items()
            if time > total_time * 0.01
        ])

        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if (speaker not in speakers) or segment.duration < self.MIN_SEGMENT_DURATION:
                continue
            segments.append({
                "speaker": self.labels.get(speaker),
                "start": segment.start,
                "stop": segment.end,
                "embeddings": np.empty((0, 192)),
            })
        return segments

    def merge_segments(self, clean_segments):
        # merge adjacent segments if they have the same speaker and are close enough
        merged = []
        for segment in clean_segments:
            if not merged:
                merged.append(segment)
                continue
            if merged[-1]["speaker"] == segment["speaker"]:
                if segment["start"] - merged[-1]["stop"] < 2.0 * self.MIN_SEGMENT_DURATION:
                    merged[-1]["stop"] = segment["stop"]
                    continue
            merged.append(segment)
        return merged

    def segment_embeddings(self, merged_segments, embeddings):
        # process the embeddings
        for i, chunk in enumerate(embeddings['data']):
            # chunk shape: (local_num_speakers, dimension)
            speakers = []
            for speaker_embedding in chunk:
                if not np.all(np.isnan(speaker_embedding)):
                    speakers.append(speaker_embedding)
            if len(speakers) != 1:
                # ignore this chunk
                continue
            # now we have a single speaker for this chunk
            speaker = speakers[0]

            # find the segment that this chunk belongs to
            chunk_start = i * embeddings['chunk_offset']
            chunk_end = chunk_start + embeddings['chunk_duration']

            for segment in merged_segments:
                if (segment['start'] <= chunk_start) and (chunk_end <= segment['stop']):
                    # this is the segment we're looking for
                    segment['embeddings'] = np.append(
                        segment['embeddings'],
                        [speaker],
                        axis=0,
                    )
                    break
        return merged_segments

    def create_speaker_embeddings(self, emb_segments):
        speaker_embeddings = collections.defaultdict(
            lambda: np.empty((0, 192)))

        for segment in emb_segments:
            if segment["embeddings"].size == 0:
                continue
            speaker_embeddings[segment["speaker"]] = np.vstack([
                speaker_embeddings[segment["speaker"]],
                segment["embeddings"],
            ])
        for speaker in speaker_embeddings:
            speaker_embeddings[speaker] = speaker_embeddings[speaker].mean(
                axis=0)
        return speaker_embeddings

    def format_segments(self, emb_segments):
        segments = []
        for segment in emb_segments:
            new = segment.copy()
            new['start'] = format_ts(new['start'])
            new['stop'] = format_ts(new['stop'])
            del new['embeddings']
            segments.append(new)
        return segments

    def format_segments_extra(self, emb_segments, speaker_embeddings):
        from sklearn.metrics.pairwise import cosine_distances

        def format_ts(ts):
            return str(datetime.timedelta(seconds=ts))

        def get_mean(embeddings):
            if len(embeddings) == 0:
                return None
            return embeddings.mean(axis=0)

        def dist(embedding, label):
            if embedding is None:
                return None
            ref = speaker_embeddings[label].reshape(1, -1)
            current = embedding.reshape(1, -1)
            return cosine_distances(ref, current)[0][0]

        segments = []
        for segment in emb_segments:
            embedding = get_mean(segment["embeddings"])
            segments.append({
                "speaker": segment["speaker"],
                "start": format_ts(segment["start"]),
                "stop": format_ts(segment["stop"]),
                "edist": dict((label, dist(embedding, label)) for label in self.labels.get_all()),
            })
        return segments
