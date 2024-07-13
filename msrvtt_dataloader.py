import os
import os.path as osp
import json
import pandas as pd
from easydict import EasyDict
from torch.utils.data import Dataset

class MSRVTT_Dataset(Dataset):
    """MSRVTT dataset loader"""
    def __init__(self, video_folder, audio_folder, asr_folder, summary_folder, lang_detect_json, csv_path):
        self.video_folder = video_folder
        self.audio_folder = audio_folder
        self.asr_folder = asr_folder
        self.summary_folder = summary_folder
        self.lang_detect_json = lang_detect_json

        # filter data for english asr
        self.data = pd.read_csv(csv_path)
        self.data["video_id"] = self.data["video_id"] + ".mp4"
        lang_detect_set = set(self.get_english_detected_list())
        self.data = self.data.loc[self.data["video_id"].isin(lang_detect_set)]
        self.data.reset_index(drop=True, inplace=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_filename = self.data['video_id'].values[idx].split(".")[0]
        video_extension = self.data['video_id'].values[idx].split(".")[1]

        sentence = self.data['sentence'].values[idx]
        video_path = os.path.join(self.video_folder, f"{video_filename}.{video_extension}")
        audio_path = os.path.join(self.audio_folder, f"{video_filename}.mp3")
        asr_path = os.path.join(self.asr_folder, f"{video_filename}.txt")
        with open(asr_path, "r") as asr_file:
            asr_text = asr_file.read()
        summary_path = os.path.join(self.summary_folder, f"{video_filename}.json")
        assert osp.exists(summary_path)
        with open(summary_path) as f:
            summary_report = json.load(f)
        summary = summary_report["Summary"]

        return sentence, video_path, audio_path, asr_text, summary
    
    def get_english_detected_list(self):
        """
        Return all video names including file extension that contain english ASR.
        """
        assert osp.exists(self.lang_detect_json)
        with open(self.lang_detect_json) as f:
            language_report = json.load(f)
        language_report = language_report["details"]
        # english_list = [k for k, v in language_report.items() if v == "<|en|>"]
        english_list = []
        for key, val in language_report.items():
            if val == "<|en|>":
                video_filename = key.split(".")[0] + ".mp4"
                english_list.append(video_filename)
        return english_list

def get_args_msrvtt():
    # build args
    args = {
        "csv_path": '/raid/1moritz/datasets/MSRVTT/MSRVTT_JSFUSION_test.csv',
        "language_detect_path": '/raid/1moritz/datasets/MSRVTT/original_data/MSRVTT/report_lang_detect.json',
        "video_folder": '/raid/1moritz/datasets/MSRVTT/original_data/MSRVTT/videos/all',
        "audio_folder": '/raid/1moritz/datasets/MSRVTT/original_data/MSRVTT/clip_audios',
        "asr_folder": '/raid/1moritz/datasets/MSRVTT/original_data/MSRVTT/clip_asr',
        "summary_folder": '/raid/1moritz/datasets/MSRVTT/original_data/MSRVTT/clip_summary_asr',
        "batch_size_val": 8,
        "num_thread_reader": 1,
        "cache_dir": '/raid/1moritz/models/languagebind/downloaded_weights',
    }
    args = EasyDict(args)
    return args

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    args = get_args_msrvtt()
    msrvtt_ds = MSRVTT_Dataset(
        video_folder=args.video_folder,
        audio_folder=args.audio_folder,
        asr_folder=args.asr_folder,
        summary_folder=args.summary_folder,
        lang_detect_json=args.language_detect_path,
        csv_path=args.csv_path,
    )
    print("Length: ",len(msrvtt_ds))

    dataloader = DataLoader(msrvtt_ds, batch_size=1, shuffle=True)
    for i, data in enumerate(dataloader):
        if i >= 3:
            break
        print(f"Sample {i + 1}: {data}")

