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
from msrvtt_dataloader import MSRVTT_Dataset


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

def run_eval(model:LanguageBind, tokenizer:LanguageBindImageTokenizer, dataloader:DataLoader, modality_transform: dict, device: torch.device):
    batch_sentences_embeddings, batch_videos_embeddings, batch_audios_embeddings, batch_asr_embeddings = [], [], [], []
    batch_summaries_embeddings = []
    # Calculate embeddings
    for bid, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        sentences, video_paths, audio_paths, asr_texts, summaries = batch

        if not isinstance(sentences, list):
            sentences = list(sentences)
        if not isinstance(video_paths, list):
            video_paths = list(video_paths)
        if not isinstance(audio_paths, list):
            audio_paths = list(audio_paths)
        if not isinstance(asr_texts, list):
            asr_texts = list(asr_texts)
        if not isinstance(summaries, list):
            summaries = list(summaries)

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
        summary_inputs = {'language': to_device(tokenizer(summaries, max_length=77, padding='max_length',
                                            truncation=True, return_tensors='pt'), device)}
        
        with torch.no_grad():
            embeddings = model(inputs)
            asr_embeddings = model(asr_inputs)
            summary_embeddings = model(summary_inputs)

        batch_sentences_embeddings.append(embeddings['language'])
        batch_audios_embeddings.append(embeddings['audio'])
        batch_videos_embeddings.append(embeddings['video'])
        batch_asr_embeddings.append(asr_embeddings['language'])
        batch_summaries_embeddings.append(summary_embeddings['language'])

    # Create similarity matrix
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_videos_embeddings)

    # Log metrics Text-to-Video
    print(f"MSR-VTT sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    tv_metrics = compute_metrics(sim_matrix)
    vt_metrics = compute_metrics(sim_matrix.T)
    print('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    print(f"MSR-VTT Text-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    print(f"MSR-VTT Video-to-Text:")
    print('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))
    
    # Log metrics Text-to-Audio
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_audios_embeddings)
    print(f"MSR-VTT sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    ta_metrics = compute_metrics(sim_matrix)
    at_metrics = compute_metrics(sim_matrix.T)
    print(f"MSR-VTT Text-to-Audio:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(ta_metrics['R1'], ta_metrics['R5'], ta_metrics['R10'], ta_metrics['MR'], ta_metrics['MeanR']))
    print(f"MSR-VTT Audio-to-Text:")
    print('\t>>>  A2T$R@1: {:.1f} - A2T$R@5: {:.1f} - A2T$R@10: {:.1f} - A2T$Median R: {:.1f} - A2T$Mean R: {:.1f}'.
                format(at_metrics['R1'], at_metrics['R5'], at_metrics['R10'], at_metrics['MR'], at_metrics['MeanR']))
    
    # Log metrics Audio-to-Video
    sim_matrix = create_sim_matrix(batch_audios_embeddings, batch_videos_embeddings)
    print(f"MSR-VTT sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    av_metrics = compute_metrics(sim_matrix)
    va_metrics = compute_metrics(sim_matrix.T)
    print(f"MSR-VTT Audio-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(av_metrics['R1'], av_metrics['R5'], av_metrics['R10'], av_metrics['MR'], av_metrics['MeanR']))
    print(f"MSR-VTT Video-to-Audio:")
    print('\t>>>  V2A$R@1: {:.1f} - V2A$R@5: {:.1f} - V2A$R@10: {:.1f} - V2A$Median R: {:.1f} - V2A$Mean R: {:.1f}'.
                format(va_metrics['R1'], va_metrics['R5'], va_metrics['R10'], va_metrics['MR'], va_metrics['MeanR']))
    
    # Log metrics Text-to-ASR
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_asr_embeddings)
    print(f"MSR-VTT sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    ta_metrics = compute_metrics(sim_matrix)
    at_metrics = compute_metrics(sim_matrix.T)
    print(f"MSR-VTT Text-to-ASR:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(ta_metrics['R1'], ta_metrics['R5'], ta_metrics['R10'], ta_metrics['MR'], ta_metrics['MeanR']))
    print(f"MSR-VTT ASR-to-Text:")
    print('\t>>>  Asr2T$R@1: {:.1f} - Asr2T$R@5: {:.1f} - Asr2T$R@10: {:.1f} - Asr2T$Median R: {:.1f} - Asr2T$Mean R: {:.1f}'.
                format(at_metrics['R1'], at_metrics['R5'], at_metrics['R10'], at_metrics['MR'], at_metrics['MeanR']))
    
    # Log metrics ASR-to-Video
    sim_matrix = create_sim_matrix(batch_asr_embeddings, batch_videos_embeddings)
    print(f"MSR-VTT sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    av_metrics = compute_metrics(sim_matrix)
    va_metrics = compute_metrics(sim_matrix.T)
    print(f"MSR-VTT ASR-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(av_metrics['R1'], av_metrics['R5'], av_metrics['R10'], av_metrics['MR'], av_metrics['MeanR']))
    print(f"MSR-VTT Video-to-ASR:")
    print('\t>>>  V2Asr$R@1: {:.1f} - V2Asr$R@5: {:.1f} - V2Asr$R@10: {:.1f} - V2Asr$Median R: {:.1f} - V2Asr$Mean R: {:.1f}'.
                format(va_metrics['R1'], va_metrics['R5'], va_metrics['R10'], va_metrics['MR'], va_metrics['MeanR']))
    
    # Log metrics Text-to-Summary
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_summaries_embeddings)
    print(f"MSR-VTT sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    ts_metrics = compute_metrics(sim_matrix)
    st_metrics = compute_metrics(sim_matrix.T)
    print(f"MSR-VTT Text-to-Summary_ASR:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(ts_metrics['R1'], ts_metrics['R5'], ts_metrics['R10'], ts_metrics['MR'], ts_metrics['MeanR']))
    print(f"MSR-VTT Summary_ASR-to-Text:")
    print('\t>>>  Sum2T$R@1: {:.1f} - Sum2T$R@5: {:.1f} - Sum2T$R@10: {:.1f} - Sum2T$Median R: {:.1f} - Sum2T$Mean R: {:.1f}'.
                format(st_metrics['R1'], st_metrics['R5'], st_metrics['R10'], st_metrics['MR'], st_metrics['MeanR']))
    
    # Log metrics Summary-to-Video
    sim_matrix = create_sim_matrix(batch_summaries_embeddings, batch_videos_embeddings)
    print(f"MSR-VTT sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    sv_metrics = compute_metrics(sim_matrix)
    vs_metrics = compute_metrics(sim_matrix.T)
    print(f"MSR-VTT Summary_ASR-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(sv_metrics['R1'], sv_metrics['R5'], sv_metrics['R10'], sv_metrics['MR'], sv_metrics['MeanR']))
    print(f"MSR-VTT Video-to-Summary_ASR:")
    print('\t>>>  V2Sum$R@1: {:.1f} - V2Sum$R@5: {:.1f} - V2Sum$R@10: {:.1f} - V2Sum$Median R: {:.1f} - V2Sum$Mean R: {:.1f}'.
                format(vs_metrics['R1'], vs_metrics['R5'], vs_metrics['R10'], vs_metrics['MR'], vs_metrics['MeanR']))

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
    args = get_args_msrvtt()

    model = LanguageBind(clip_type=clip_type, cache_dir=args.cache_dir)
    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'LanguageBind/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir=osp.join(args.cache_dir, 'tokenizer_cache_dir'))
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    dataloader_msrvtt = DataLoader(
        MSRVTT_Dataset(
            video_folder=args.video_folder, 
            audio_folder=args.audio_folder,
            asr_folder=args.asr_folder,
            summary_folder=args.summary_folder,
            lang_detect_json=args.language_detect_path,
            csv_path=args.csv_path),
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    run_eval(model, tokenizer, dataloader_msrvtt, modality_transform, device)


if __name__ == '__main__':
    main()
"""
MSR-VTT sim matrix size: 619, 619
	 Length-T: 619, Length-V:619
MSR-VTT Text-to-Video:
	>>>  R@1: 48.1 - R@5: 73.7 - R@10: 81.1 - Median R: 2.0 - Mean R: 13.5
MSR-VTT Video-to-Text:
	>>>  V2T$R@1: 46.4 - V2T$R@5: 71.8 - V2T$R@10: 80.9 - V2T$Median R: 2.0 - V2T$Mean R: 14.6
MSR-VTT sim matrix size: 619, 619
MSR-VTT Text-to-Audio:
	>>>  R@1: 2.4 - R@5: 8.6 - R@10: 12.9 - Median R: 102.0 - Mean R: 158.1
MSR-VTT Audio-to-Text:
	>>>  A2T$R@1: 1.3 - A2T$R@5: 5.1 - A2T$R@10: 8.8 - A2T$Median R: 137.0 - A2T$Mean R: 182.5
MSR-VTT sim matrix size: 619, 619
MSR-VTT Audio-to-Video:
	>>>  R@1: 2.9 - R@5: 11.0 - R@10: 15.5 - Median R: 119.0 - Mean R: 175.7
MSR-VTT Video-to-Audio:
	>>>  V2A$R@1: 5.3 - V2A$R@5: 11.8 - V2A$R@10: 17.9 - V2A$Median R: 86.0 - V2A$Mean R: 137.2
MSR-VTT sim matrix size: 619, 619
MSR-VTT Text-to-ASR:
	>>>  R@1: 17.9 - R@5: 28.5 - R@10: 33.3 - Median R: 45.0 - Mean R: 130.0
MSR-VTT ASR-to-Text:
	>>>  Asr2T$R@1: 15.0 - Asr2T$R@5: 23.8 - Asr2T$R@10: 27.4 - Asr2T$Median R: 108.0 - Asr2T$Mean R: 179.7
MSR-VTT sim matrix size: 619, 619
MSR-VTT ASR-to-Video:
	>>>  R@1: 33.8 - R@5: 47.7 - R@10: 54.8 - Median R: 7.0 - Mean R: 68.9
MSR-VTT Video-to-ASR:
	>>>  V2Asr$R@1: 31.0 - V2Asr$R@5: 47.3 - V2Asr$R@10: 53.7 - V2Asr$Median R: 7.5 - V2Asr$Mean R: 57.8
"""
"""
MSR-VTT sim matrix size: 619, 619
	 Length-T: 619, Length-V:619
MSR-VTT Text-to-Video:
	>>>  R@1: 47.7 - R@5: 72.5 - R@10: 81.6 - Median R: 2.0 - Mean R: 13.4
MSR-VTT Video-to-Text:
	>>>  V2T$R@1: 45.6 - V2T$R@5: 70.8 - V2T$R@10: 80.1 - V2T$Median R: 2.0 - V2T$Mean R: 14.5
MSR-VTT sim matrix size: 619, 619
MSR-VTT Text-to-Audio:
	>>>  R@1: 2.6 - R@5: 9.2 - R@10: 12.4 - Median R: 106.0 - Mean R: 158.2
MSR-VTT Audio-to-Text:
	>>>  A2T$R@1: 1.4 - A2T$R@5: 4.9 - A2T$R@10: 7.8 - A2T$Median R: 129.0 - A2T$Mean R: 180.8
MSR-VTT sim matrix size: 619, 619
MSR-VTT Audio-to-Video:
	>>>  R@1: 2.7 - R@5: 10.5 - R@10: 15.2 - Median R: 122.0 - Mean R: 174.8
MSR-VTT Video-to-Audio:
	>>>  V2A$R@1: 4.7 - V2A$R@5: 11.5 - V2A$R@10: 17.1 - V2A$Median R: 77.0 - V2A$Mean R: 138.1
MSR-VTT sim matrix size: 619, 619
MSR-VTT Text-to-ASR:
	>>>  R@1: 17.9 - R@5: 28.5 - R@10: 33.3 - Median R: 45.0 - Mean R: 130.0
MSR-VTT ASR-to-Text:
	>>>  Asr2T$R@1: 15.0 - Asr2T$R@5: 23.8 - Asr2T$R@10: 27.4 - Asr2T$Median R: 108.0 - Asr2T$Mean R: 179.7
MSR-VTT sim matrix size: 619, 619
MSR-VTT ASR-to-Video:
	>>>  R@1: 35.9 - R@5: 49.1 - R@10: 54.8 - Median R: 6.0 - Mean R: 68.9
MSR-VTT Video-to-ASR:
	>>>  V2Asr$R@1: 30.3 - V2Asr$R@5: 47.3 - V2Asr$R@10: 53.1 - V2Asr$Median R: 8.0 - V2Asr$Mean R: 57.3
MSR-VTT sim matrix size: 619, 619
MSR-VTT Text-to-Summary_ASR:
	>>>  R@1: 16.6 - R@5: 26.6 - R@10: 31.6 - Median R: 53.0 - Mean R: 134.2
MSR-VTT Summary_ASR-to-Text:
	>>>  Sum2T$R@1: 14.4 - Sum2T$R@5: 22.8 - Sum2T$R@10: 27.8 - Sum2T$Median R: 110.0 - Sum2T$Mean R: 182.3
MSR-VTT sim matrix size: 619, 619
MSR-VTT Summary_ASR-to-Video:
	>>>  R@1: 33.9 - R@5: 48.9 - R@10: 54.4 - Median R: 6.0 - Mean R: 70.9
MSR-VTT Video-to-Summary_ASR:
	>>>  V2Sum$R@1: 29.8 - V2Sum$R@5: 46.2 - V2Sum$R@10: 53.3 - V2Sum$Median R: 8.0 - V2Sum$Mean R: 59.5
"""