import os
import json
import numpy as np
from torch.utils.data import Dataset


class VidChapters7M_Dataset(Dataset):
    """VidChapters-7M dataset loader"""
    def __init__(self, json_path, video_folder, audio_folder, asr_folder, summary_folder):
        self.video_folder = video_folder
        self.audio_folder = audio_folder
        self.asr_folder = asr_folder
        self.summary_folder = summary_folder
        self.data = [f for f in os.listdir(self.video_folder) if f.lower().endswith('.mp4')]

        with open(json_path, 'r') as file:
            self.annotations = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_filename = self.data[idx].split(".")[0]
        video_extension = self.data[idx].split(".")[1]

        assert len(video_filename.split("_")) > 2
        chunk_times = filename_to_timestamps(self.data[idx])
        video_id = '_'.join(video_filename.split("_")[:-2])

        # store chunk start time * 10 to handle it as an integer
        chunk_id = (video_id, round(10 * chunk_times[0]))

        gt_chapter_timestamps = self.annotations[video_id]["timestamps"]
        iou_list = iou_1d(gt_chapter_timestamps, chunk_times)
        idx_max_iou = np.argmax(iou_list)
        sentence_id = (video_id, idx_max_iou)

        video_path = os.path.join(self.video_folder, f"{video_filename}.{video_extension}")
        audio_path = os.path.join(self.audio_folder, f"{video_filename}.mp3")
        asr_path = os.path.join(self.asr_folder, f"{video_filename}.txt")
        with open(asr_path, "r") as asr_file:
            asr_text = asr_file.read()
        summary_path = os.path.join(self.summary_folder, f"{video_filename}.json")
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            summary_report = json.load(f)
        summary = summary_report["Summary"]

        # store summary stats for transparency
        summary_stats = {
            "n_words_asr": summary_report["n_words_asr"],
            "only_copy": summary_report["only_copy"],
        }

        return chunk_id, sentence_id, video_path, audio_path, asr_text, summary, summary_stats


class VidChapText_Dataset(Dataset):
    """Description sentence dataset loader for VidChapters-7M"""
    def __init__(self, json_path, video_folder):
        self.video_folder = video_folder
        with open(json_path, 'r') as file:
            self.annotations = json.load(file)

        video_identifiers = {'_'.join(f.split("_")[:-2]) for f in os.listdir(self.video_folder) if f.lower().endswith('.mp4')}  # noqa
        self.data = []
        for video_id in video_identifiers:
            for sentence_index in range(len(self.annotations[video_id]["sentences"])):
                self.data.append((video_id, sentence_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_id, sentence_index = self.data[idx]
        sentence_id = self.data[idx]
        sentence = self.annotations[video_id]["sentences"][sentence_index]

        sentence_times = self.annotations[video_id]["timestamps"][sentence_index]

        chunk_filenames = [f for f in os.listdir(self.video_folder) if f.startswith(f'{video_id}_')]
        chunk_timestamps = [filename_to_timestamps(f_name) for f_name in chunk_filenames]

        iou_list = iou_1d(chunk_timestamps, sentence_times)
        idx_max_iou = np.argmax(iou_list)
        assert iou_list[idx_max_iou] > 0
        chunk_id = (video_id, round(10 * chunk_timestamps[idx_max_iou][0]))

        return sentence_id, chunk_id, sentence


def filename_to_timestamps(filename):
    fname_no_ext = ".".join(filename.split(".")[:-1])
    f_parts = fname_no_ext.split("_")
    assert len(f_parts) >= 3
    timestamps = f_parts[-2:]
    timestamps = [float(ts.replace("-", ".")) for ts in timestamps]
    return timestamps


def duration(start_stop):
    assert len(start_stop) == 2, "Function only accepts lists with two entries"
    assert start_stop[0] <= start_stop[1], f"Negative duration is not allowed! Received: {start_stop}"
    duration = start_stop[1] - start_stop[0]
    return duration


def iou_1d(gt_chapters, chunk):
    iou_list = []
    for chapter in gt_chapters:
        overlap = min(chunk[1], chapter[1]) - max(chunk[0], chapter[0])
        overlap = max(0, overlap)
        if overlap < 0.001:
            iou = 0
        else:
            union = duration(chunk) + duration(chapter) - overlap
            iou = overlap / union
        iou_list.append(iou)
    return iou_list


if __name__ == "__main__":
    from chunk_vidchap_eval import get_args_vidchap
    args = get_args_vidchap()
    text_ds = VidChapText_Dataset(json_path=args.json_path, video_folder=args.video_folder)
    print(text_ds.__getitem__(1))
