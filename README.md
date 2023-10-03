# DALLE2-video

## Status

Something similar to Unet3D proposed in [Ho et al., Apr 2022](https://arxiv.org/abs/2204.03458) is implemented and working.

- Sep. 29
  - ms4-5にて，4 subsetsでdecoder trainingを開始．
- Sep. 20
  - CLIP trainingが良さそうだったので切り上げた．Decoder trainingがdeepspeed level3, batch_size=1でも走らないので要改善．
- Sep. 8
  - FSDPのdecoder trainingスクリプトを実装．CLIP trainingがおわったら実行．
- Sep. 7
  - CLIP trainingはshardingなしでできたので，とりあえず走らせた（1エポック5時間程度）．
  - Shardingなしのdecoder訓練スクリプトを実装．

## Training on CelebV-Text dataset

Download CelebV-Text dataset from their [GitHub](https://github.com/celebv-text/CelebV-Text/issues/8).

- Dataset (70,000 videos) is separated into 70 zip files. You don't need to download all of them.

- After untaring the dataset, the folder structure should look like this:

```
├── texts
│   ├── action_dur
│   ├── emotion
│   ├── face40_details_new
│   ├── light_colot_temp
│   ├── light_dir
│   └── light_intensity
└── videos
    ├── celebvtext_6
    └── ...
```

Run preprocessing.

```bash
python preprocess.py
```

- Tokenizes text and save as a `.pt` file (saved under `texts/`).

- Preprocesses videos and save as a `.h5` file (saved under `videos/`).

  - Videos are usually tens of seconds long. This preprocessing takes only the first `args.seq_len` seconds.

  - When training, `CelebVTextDataset` only reads pointers to the videos. The videos are loaded in the collate function. This is to avoid loading all videos into memory at once.

Configure DeepSpeed

```bash
accelerate config
```

- You should answer yes the question that asks if you want to use a config file for DeepSpeed to use the example json file that I provided `configs/deepspeed_config.json`

  - Basic Accelerate config file will be saved to `~/.cache/huggingface/accelerate/default_config.yaml`

  - When using `deepspeed_config_file` (json), variable `gradient_accumulation_steps` is ignored, so we need to specify that from train config file (yaml).

  - For more details refer to [this page](https://huggingface.co/docs/accelerate/usage_guides/deepspeed#deepspeed-config-file).

Run CLIP training

```bash
nohup python train_clip.py use_wandb=**** > logs/****.log &
```

Run decoder training

```bash
nohup accelerate launch train_decoder.py train_name=**** use_wandb=**** > logs/****.log &
```

## TODOs

- [x] Unet3D proposed in [Ho et al., Apr 2022](https://arxiv.org/abs/2204.03458)
- [ ] Support ZeRO stage 3 training with [DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed#deepspeed-config-file)
- [ ] Temporal super-resolution proposed in [Ho et al., Oct 2022](https://arxiv.org/abs/2210.02303)
- [ ] Learning variance for video diffusion significantly unstabilized the training. Is there a way to stably learn variance?

## References

- [DALLE2-pytorch](https://github.com/lucidrains/DALLE2-pytorch)

- Video Diffusion Models ([Ho et al., Apr 2022](https://arxiv.org/abs/2204.03458))

- Imagen Video: High Definition Video Generation with Diffusion Models ([Ho et al., Oct 2022](https://arxiv.org/abs/2210.02303))