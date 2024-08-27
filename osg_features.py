import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from tqdm.auto import tqdm
import os.path as osp
import numpy as np
from easydict import EasyDict
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
import json


def get_config():
    config = {
        "features_folder": "/ltstorage/home/1moritz/storage/datasets/VidChapters-7M/osg_features",
        "sentence_json": "sentence2frame.json",
        "model_subfolder": "languagebind",
        "cache_dir": '/ltstorage/home/1moritz/storage/models/languagebind/downloaded_weights',
    }
    return EasyDict(config)


def extract_features(config, batch_size):
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    clip_type = {
        'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
        'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
        'image': 'LanguageBind_Image',
    }

    model = LanguageBind(clip_type=clip_type, cache_dir=config.cache_dir)
    model = model.to(device)
    model.eval()
    pretrained_ckpt = 'LanguageBind/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(
        pretrained_ckpt, cache_dir=osp.join(config.cache_dir, 'tokenizer_cache_dir')
    )
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    json_path = osp.join(config.features_folder, config.sentence_json)
    assert osp.exists(json_path)
    with open(json_path) as f:
        json_inputs = json.load(f)

    for video_id in tqdm(json_inputs.keys()):
        texts = [val["text"] for val in json_inputs[video_id].values()]
        frames = [val["frame_path"] for val in json_inputs[video_id].values()]
        assert len(texts) == len(frames)

        text_features = None
        frame_features = None
        for i in tqdm(range(0, len(texts), batch_size), leave=False, desc=video_id):
            texts_batch = texts[i: i + batch_size]
            frames_batch = frames[i: i + batch_size]

            inputs = {
                'language': to_device(tokenizer(texts_batch, max_length=77, padding='max_length',
                                                truncation=True, return_tensors='pt'), device),
                'image': to_device(modality_transform['image'](frames_batch), device)
            }
            with torch.no_grad():
                embeddings = model(inputs)

            if text_features is None or frame_features is None:
                # Initialize features with the first batch
                assert text_features is None
                assert frame_features is None
                text_features = embeddings['language']
                frame_features = embeddings['image']
            else:
                # Concatenate along the batch dimension
                text_features = torch.cat((text_features, embeddings['language']), dim=0)
                frame_features = torch.cat((frame_features, embeddings['image']), dim=0)

        # Define the output path
        feature_out_path = osp.join(
            config.features_folder, config.model_subfolder, f"features_{video_id}.npz"
        )
        if not os.path.exists(osp.dirname(feature_out_path)):
            os.makedirs(osp.dirname(feature_out_path))

        # Save the numpy array as a compressed .npz file
        np.savez_compressed(
            feature_out_path,
            text_features=text_features.cpu().numpy(),
            frame_features=frame_features.cpu().numpy()
        )


if __name__ == '__main__':
    config = get_config()
    extract_features(config, batch_size=64)
