import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from easydict import EasyDict
from torch.utils.data import DataLoader
import numpy as np
import torch
import os.path as osp
from tqdm.auto import tqdm as tqdm
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from vl_ret.metrics import compute_metrics
import sys
from vidchapters7m_dataloader import VidChapters7M_Dataset


def get_args_vidchap():
    # build args
    args = {
        "json_path": '/raid/1moritz/datasets/VidChapters-7M/chapters_dvc_test.json',
        "video_folder": '/raid/1moritz/datasets/VidChapters-7M/chapter_clips',
        "audio_folder": '/raid/1moritz/datasets/VidChapters-7M/chapter_audios',
        "asr_folder": '/raid/1moritz/datasets/VidChapters-7M/chapter_asr',
        "batch_size_val": 8,
        "num_thread_reader": 1,
        "cache_dir": '/raid/1moritz/models/languagebind/downloaded_weights',
    }
    args = EasyDict(args)
    return args

def run_eval(model:LanguageBind, tokenizer:LanguageBindImageTokenizer, dataloader:DataLoader, modality_transform: dict, device: torch.device):
    batch_sentences_embeddings, batch_videos_embeddings, batch_audios_embeddings, batch_asr_embeddings = [], [], [], []
    # Calculate embeddings
    for bid, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        sentences, video_paths, audio_paths, asr_texts = batch

        if not isinstance(sentences, list):
            sentences = list(sentences)
        if not isinstance(video_paths, list):
            video_paths = list(video_paths)
        if not isinstance(audio_paths, list):
            audio_paths = list(audio_paths)
        if not isinstance(asr_texts, list):
            asr_texts = list(asr_texts)

        # print(sentences)
        # print(type(sentences))
        # print(video_paths)
        # print(type(video_paths))
        # print(audio_paths)
        # print(type(audio_paths))
        # sys.exit()
        inputs = {
            'video': to_device(modality_transform['video'](video_paths), device),
            'audio': to_device(modality_transform['audio'](audio_paths), device)
        }
        inputs['language'] = to_device(tokenizer(sentences, max_length=77, padding='max_length',
                                            truncation=True, return_tensors='pt'), device)
        asr_inputs = {'language': to_device(tokenizer(asr_texts, max_length=77, padding='max_length',
                                            truncation=True, return_tensors='pt'), device)}
        
        with torch.no_grad():
            embeddings = model(inputs)
            asr_embeddings = model(asr_inputs)

        batch_sentences_embeddings.append(embeddings['language'])
        batch_audios_embeddings.append(embeddings['audio'])
        batch_videos_embeddings.append(embeddings['video'])
        batch_asr_embeddings.append(asr_embeddings['language'])

    # Create similarity matrix
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_videos_embeddings)

    # Log metrics Text-to-Video
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    tv_metrics = compute_metrics(sim_matrix)
    vt_metrics = compute_metrics(sim_matrix.T)
    print('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    print(f"VidChapters Text-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    print(f"VidChapters Video-to-Text:")
    print('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))
    
    # Log metrics Text-to-Audio
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_audios_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    ta_metrics = compute_metrics(sim_matrix)
    at_metrics = compute_metrics(sim_matrix.T)
    print(f"VidChapters Text-to-Audio:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(ta_metrics['R1'], ta_metrics['R5'], ta_metrics['R10'], ta_metrics['MR'], ta_metrics['MeanR']))
    print(f"VidChapters Audio-to-Text:")
    print('\t>>>  A2T$R@1: {:.1f} - A2T$R@5: {:.1f} - A2T$R@10: {:.1f} - A2T$Median R: {:.1f} - A2T$Mean R: {:.1f}'.
                format(at_metrics['R1'], at_metrics['R5'], at_metrics['R10'], at_metrics['MR'], at_metrics['MeanR']))
    
    # Log metrics Audio-to-Video
    sim_matrix = create_sim_matrix(batch_audios_embeddings, batch_videos_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    av_metrics = compute_metrics(sim_matrix)
    va_metrics = compute_metrics(sim_matrix.T)
    print(f"VidChapters Audio-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(av_metrics['R1'], av_metrics['R5'], av_metrics['R10'], av_metrics['MR'], av_metrics['MeanR']))
    print(f"VidChapters Video-to-Audio:")
    print('\t>>>  V2A$R@1: {:.1f} - V2A$R@5: {:.1f} - V2A$R@10: {:.1f} - V2A$Median R: {:.1f} - V2A$Mean R: {:.1f}'.
                format(va_metrics['R1'], va_metrics['R5'], va_metrics['R10'], va_metrics['MR'], va_metrics['MeanR']))
    
    # Log metrics Text-to-ASR
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_asr_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    ta_metrics = compute_metrics(sim_matrix)
    at_metrics = compute_metrics(sim_matrix.T)
    print(f"VidChapters Text-to-ASR:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(ta_metrics['R1'], ta_metrics['R5'], ta_metrics['R10'], ta_metrics['MR'], ta_metrics['MeanR']))
    print(f"VidChapters ASR-to-Text:")
    print('\t>>>  Asr2T$R@1: {:.1f} - Asr2T$R@5: {:.1f} - Asr2T$R@10: {:.1f} - Asr2T$Median R: {:.1f} - Asr2T$Mean R: {:.1f}'.
                format(at_metrics['R1'], at_metrics['R5'], at_metrics['R10'], at_metrics['MR'], at_metrics['MeanR']))
    
    # Log metrics ASR-to-Video
    sim_matrix = create_sim_matrix(batch_asr_embeddings, batch_videos_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    av_metrics = compute_metrics(sim_matrix)
    va_metrics = compute_metrics(sim_matrix.T)
    print(f"VidChapters ASR-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(av_metrics['R1'], av_metrics['R5'], av_metrics['R10'], av_metrics['MR'], av_metrics['MeanR']))
    print(f"VidChapters Video-to-ASR:")
    print('\t>>>  V2Asr$R@1: {:.1f} - V2Asr$R@5: {:.1f} - V2Asr$R@10: {:.1f} - V2Asr$Median R: {:.1f} - V2Asr$Mean R: {:.1f}'.
                format(va_metrics['R1'], va_metrics['R5'], va_metrics['R10'], va_metrics['MR'], va_metrics['MeanR']))

def create_sim_matrix(batch_sentences_embeddings, batch_videos_embeddings):
    """Calculate embedding vector product for similarity and download result to CPU
    
        Returns: 
            sim_matrix (Text X Video)
    """
    sim_matrix = []
    for idx1 in range(len(batch_sentences_embeddings)):
        sequence_output = batch_sentences_embeddings[idx1]
        each_row = []
        for idx2 in range(len(batch_videos_embeddings)):
            visual_output = batch_videos_embeddings[idx2]
            b1b2 =  sequence_output @ visual_output.T
            b1b2 = b1b2.cpu().detach().numpy()
            each_row.append(b1b2)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    return sim_matrix

def main():
    device = 'cuda:0'
    device = torch.device(device)
    clip_type = {
        'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
        'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
        'image': 'LanguageBind_Image',
    }
    args = get_args_vidchap()

    model = LanguageBind(clip_type=clip_type, cache_dir=args.cache_dir)
    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'LanguageBind/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir=osp.join(args.cache_dir, 'tokenizer_cache_dir'))
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    dataloader_vidchap = DataLoader(
        VidChapters7M_Dataset(
            json_path=args.json_path, 
            video_folder=args.video_folder, 
            audio_folder=args.audio_folder,
            asr_folder=args.asr_folder),
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    run_eval(model, tokenizer, dataloader_vidchap, modality_transform, device)


if __name__ == '__main__':
    main()
"""
VidChapters sim matrix size: 895, 895
	 Length-T: 895, Length-V:895
VidChapters Text-to-Video:
	>>>  R@1: 31.3 - R@5: 48.0 - R@10: 53.9 - Median R: 7.0 - Mean R: 110.8
VidChapters Video-to-Text:
	>>>  V2T$R@1: 11.2 - V2T$R@5: 19.7 - V2T$R@10: 22.4 - V2T$Median R: 185.0 - V2T$Mean R: 210.0
VidChapters sim matrix size: 895, 895
VidChapters Text-to-Audio:
	>>>  R@1: 0.2 - R@5: 1.3 - R@10: 2.7 - Median R: 339.0 - Mean R: 373.5
VidChapters Audio-to-Text:
	>>>  A2T$R@1: 0.1 - A2T$R@5: 0.5 - A2T$R@10: 1.1 - A2T$Median R: 216.0 - A2T$Mean R: 292.1
VidChapters sim matrix size: 895, 895
VidChapters Audio-to-Video:
	>>>  R@1: 0.7 - R@5: 2.8 - R@10: 5.8 - Median R: 217.0 - Mean R: 299.1
VidChapters Video-to-Audio:
	>>>  V2A$R@1: 1.7 - V2A$R@5: 6.7 - V2A$R@10: 10.1 - V2A$Median R: 150.0 - V2A$Mean R: 213.7
"""
"""
VidChapters sim matrix size: 895, 895
	 Length-T: 895, Length-V:895
VidChapters Text-to-Video:
	>>>  R@1: 31.4 - R@5: 47.4 - R@10: 53.7 - Median R: 7.0 - Mean R: 111.8
VidChapters Video-to-Text:
	>>>  V2T$R@1: 11.7 - V2T$R@5: 19.9 - V2T$R@10: 22.2 - V2T$Median R: 191.0 - V2T$Mean R: 213.4
VidChapters sim matrix size: 895, 895
VidChapters Text-to-Audio:
	>>>  R@1: 0.3 - R@5: 1.5 - R@10: 2.3 - Median R: 343.0 - Mean R: 375.3
VidChapters Audio-to-Text:
	>>>  A2T$R@1: 0.0 - A2T$R@5: 0.6 - A2T$R@10: 0.9 - A2T$Median R: 207.0 - A2T$Mean R: 290.7
VidChapters sim matrix size: 895, 895
VidChapters Audio-to-Video:
	>>>  R@1: 1.0 - R@5: 3.1 - R@10: 6.9 - Median R: 212.0 - Mean R: 297.1
VidChapters Video-to-Audio:
	>>>  V2A$R@1: 1.5 - V2A$R@5: 6.8 - V2A$R@10: 10.5 - V2A$Median R: 137.0 - V2A$Mean R: 210.5
VidChapters sim matrix size: 895, 895
VidChapters Text-to-ASR:
	>>>  R@1: 27.8 - R@5: 37.8 - R@10: 42.4 - Median R: 31.0 - Mean R: 138.6
VidChapters ASR-to-Text:
	>>>  Asr2T$R@1: 15.9 - Asr2T$R@5: 21.7 - Asr2T$R@10: 24.2 - Asr2T$Median R: 114.0 - Asr2T$Mean R: 164.5
VidChapters sim matrix size: 895, 895
VidChapters ASR-to-Video:
	>>>  R@1: 30.6 - R@5: 54.6 - R@10: 64.7 - Median R: 4.0 - Mean R: 68.2
VidChapters Video-to-ASR:
	>>>  V2Asr$R@1: 26.4 - V2Asr$R@5: 48.5 - V2Asr$R@10: 56.6 - V2Asr$Median R: 6.0 - V2Asr$Mean R: 53.0
"""