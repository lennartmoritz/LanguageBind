import os
import json
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
        if len(video_filename.split("_")) > 1:
            chapter_id = video_filename.split("_")[-1]
            video_id = '_'.join(video_filename.split("_")[:-1])
        else:
            chapter_id = None
            video_id = video_filename
        assert chapter_id != None
        sentence = self.annotations[video_id]["sentences"][int(chapter_id)]
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

        return sentence, video_path, audio_path, asr_text, summary
