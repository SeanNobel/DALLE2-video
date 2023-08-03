from dalle2_pytorch.trainer import *
from dalle2_video.dalle2_video import VideoDecoder


class VideoDecoderTrainer(nn.Module):
    def __init__(
        self,
        decoder,
        accum_grad: bool = False,
        accelerator=None,
        dataloaders=None,
        use_ema=True,
        lr=1e-4,
        wd=1e-2,
        eps=1e-8,
        warmup_steps=None,
        cosine_decay_max_steps=None,
        max_grad_norm=0.5,
        amp=False,
        group_wd_params=True,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(decoder, VideoDecoder)
        ema_kwargs, kwargs = groupby_prefix_and_trim("ema_", kwargs)

        self.accum_grad = accum_grad

        self.accelerator = default(accelerator, Accelerator)

        self.num_unets = len(decoder.unets)

        self.use_ema = use_ema
        self.ema_unets = nn.ModuleList([])

        self.amp = amp

        # be able to finely customize learning rate, weight decay
        # per unet

        lr, wd, eps, warmup_steps, cosine_decay_max_steps = map(
            partial(cast_tuple, length=self.num_unets),
            (lr, wd, eps, warmup_steps, cosine_decay_max_steps),
        )

        assert all(
            [unet_lr <= 1e-2 for unet_lr in lr]
        ), "your learning rate is too high, recommend sticking with 1e-4, at most 5e-4"

        optimizers = []
        schedulers = []
        warmup_schedulers = []

        for (
            unet,
            unet_lr,
            unet_wd,
            unet_eps,
            unet_warmup_steps,
            unet_cosine_decay_max_steps,
        ) in zip(decoder.unets, lr, wd, eps, warmup_steps, cosine_decay_max_steps):
            if isinstance(unet, nn.Identity):
                optimizers.append(None)
                schedulers.append(None)
                warmup_schedulers.append(None)
            else:
                optimizer = get_optimizer(
                    unet.parameters(),
                    lr=unet_lr,
                    wd=unet_wd,
                    eps=unet_eps,
                    group_wd_params=group_wd_params,
                    **kwargs,
                )

                optimizers.append(optimizer)

                if exists(unet_cosine_decay_max_steps):
                    scheduler = CosineAnnealingLR(
                        optimizer, T_max=unet_cosine_decay_max_steps
                    )
                else:
                    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

                warmup_scheduler = (
                    warmup.LinearWarmup(optimizer, warmup_period=unet_warmup_steps)
                    if exists(unet_warmup_steps)
                    else None
                )
                warmup_schedulers.append(warmup_scheduler)

                schedulers.append(scheduler)

            if self.use_ema:
                self.ema_unets.append(EMA(unet, **ema_kwargs))

        # gradient clipping if needed

        self.max_grad_norm = max_grad_norm

        self.register_buffer("steps", torch.tensor([0] * self.num_unets))

        if (
            self.accelerator.distributed_type == DistributedType.DEEPSPEED
            and decoder.clip is not None
        ):
            # Then we need to make sure clip is using the correct precision or else deepspeed will error
            cast_type_map = {
                "fp16": torch.half,
                "bf16": torch.bfloat16,
                "no": torch.float,
            }
            precision_type = cast_type_map[accelerator.mixed_precision]
            assert (
                precision_type == torch.float
            ), "DeepSpeed currently only supports float32 precision when using on the fly embedding generation from clip"
            clip = decoder.clip
            clip.to(precision_type)

        decoder, *optimizers = list(self.accelerator.prepare(decoder, *optimizers))

        self.decoder = decoder

        # prepare dataloaders

        train_loader = val_loader = None
        if exists(dataloaders):
            train_loader, val_loader = self.accelerator.prepare(
                dataloaders["train"], dataloaders["val"]
            )

        self.train_loader = train_loader
        self.val_loader = val_loader

        # store optimizers

        for opt_ind, optimizer in zip(range(len(optimizers)), optimizers):
            setattr(self, f"optim{opt_ind}", optimizer)

        # store schedulers

        for sched_ind, scheduler in zip(range(len(schedulers)), schedulers):
            setattr(self, f"sched{sched_ind}", scheduler)

        # store warmup schedulers

        self.warmup_schedulers = warmup_schedulers

    def validate_and_return_unet_number(self, unet_number=None):
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        assert exists(unet_number) and 1 <= unet_number <= self.num_unets
        return unet_number

    def num_steps_taken(self, unet_number=None):
        unet_number = self.validate_and_return_unet_number(unet_number)
        return self.steps[unet_number - 1].item()

    def save(self, path, overwrite=True, **kwargs):
        path = Path(path)
        assert not (path.exists() and not overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_obj = dict(
            model=self.accelerator.unwrap_model(self.decoder).state_dict(),
            version=__version__,
            steps=self.steps.cpu(),
            **kwargs,
        )

        for ind in range(0, self.num_unets):
            optimizer_key = f"optim{ind}"
            scheduler_key = f"sched{ind}"

            optimizer = getattr(self, optimizer_key)
            scheduler = getattr(self, scheduler_key)

            optimizer_state_dict = optimizer.state_dict() if exists(optimizer) else None
            scheduler_state_dict = scheduler.state_dict() if exists(scheduler) else None

            save_obj = {
                **save_obj,
                optimizer_key: optimizer_state_dict,
                scheduler_key: scheduler_state_dict,
            }

        if self.use_ema:
            save_obj = {**save_obj, "ema": self.ema_unets.state_dict()}

        self.accelerator.save(save_obj, str(path))

    def load_state_dict(self, loaded_obj, only_model=False, strict=True):
        if version.parse(__version__) != version.parse(loaded_obj["version"]):
            self.accelerator.print(
                f'loading saved decoder at version {loaded_obj["version"]}, but current package version is {__version__}'
            )

        self.accelerator.unwrap_model(self.decoder).load_state_dict(
            loaded_obj["model"], strict=strict
        )
        self.steps.copy_(loaded_obj["steps"])

        if only_model:
            return loaded_obj

        for ind, last_step in zip(range(0, self.num_unets), self.steps.tolist()):
            optimizer_key = f"optim{ind}"
            optimizer = getattr(self, optimizer_key)

            scheduler_key = f"sched{ind}"
            scheduler = getattr(self, scheduler_key)

            warmup_scheduler = self.warmup_schedulers[ind]

            if exists(optimizer):
                optimizer.load_state_dict(loaded_obj[optimizer_key])

            if exists(scheduler):
                scheduler.load_state_dict(loaded_obj[scheduler_key])

            if exists(warmup_scheduler):
                warmup_scheduler.last_step = last_step

        if self.use_ema:
            assert "ema" in loaded_obj
            self.ema_unets.load_state_dict(loaded_obj["ema"], strict=strict)

    def load(self, path, only_model=False, strict=True):
        path = Path(path)
        assert path.exists()

        loaded_obj = torch.load(str(path), map_location="cpu")

        self.load_state_dict(loaded_obj, only_model=only_model, strict=strict)

        return loaded_obj

    @property
    def unets(self):
        return nn.ModuleList([ema.ema_model for ema in self.ema_unets])

    def increment_step(self, unet_number):
        assert 1 <= unet_number <= self.num_unets

        unet_index_tensor = torch.tensor(unet_number - 1, device=self.steps.device)
        self.steps += F.one_hot(unet_index_tensor, num_classes=len(self.steps))

    def update(self, unet_number=None):
        unet_number = self.validate_and_return_unet_number(unet_number)
        index = unet_number - 1

        optimizer = getattr(self, f"optim{index}")
        scheduler = getattr(self, f"sched{index}")

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(
                self.decoder.parameters(), self.max_grad_norm
            )  # Automatically unscales gradients

        optimizer.step()
        optimizer.zero_grad()

        warmup_scheduler = self.warmup_schedulers[index]
        scheduler_context = (
            warmup_scheduler.dampening if exists(warmup_scheduler) else nullcontext
        )

        with scheduler_context():
            scheduler.step()

        if self.use_ema:
            ema_unet = self.ema_unets[index]
            ema_unet.update()

        self.increment_step(unet_number)

    @torch.no_grad()
    @cast_torch_tensor
    @decoder_sample_in_chunks
    def sample(self, *args, **kwargs):
        distributed = self.accelerator.num_processes > 1
        base_decoder = self.accelerator.unwrap_model(self.decoder)

        was_training = base_decoder.training
        base_decoder.eval()

        if kwargs.pop("use_non_ema", False) or not self.use_ema:
            out = base_decoder.sample(*args, **kwargs, distributed=distributed)
            base_decoder.train(was_training)
            return out

        trainable_unets = self.accelerator.unwrap_model(self.decoder).unets
        base_decoder.unets = (
            self.unets
        )  # swap in exponential moving averaged unets for sampling

        output = base_decoder.sample(*args, **kwargs, distributed=distributed)

        base_decoder.unets = trainable_unets  # restore original training unets

        # cast the ema_model unets back to original device
        for ema in self.ema_unets:
            ema.restore_ema_model_device()

        base_decoder.train(was_training)
        return output

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def embed_text(self, *args, **kwargs):
        return self.accelerator.unwrap_model(self.decoder).clip.embed_text(
            *args, **kwargs
        )

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def embed_image(self, *args, **kwargs):
        return self.accelerator.unwrap_model(self.decoder).clip.embed_image(
            *args, **kwargs
        )

    @cast_torch_tensor
    def forward(
        self,
        *args,
        unet_number=None,
        max_batch_size=None,
        return_lowres_cond_video=False,
        **kwargs,
    ):
        unet_number = self.validate_and_return_unet_number(unet_number)

        total_loss = []
        cond_videos = []
        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(
            *args, split_size=max_batch_size, **kwargs
        ):
            with self.accelerator.autocast():
                loss_obj = self.decoder(
                    *chunked_args,
                    unet_number=unet_number,
                    return_lowres_cond_video=return_lowres_cond_video,
                    **chunked_kwargs,
                )
                # loss_obj may be a tuple with loss and cond_image
                if return_lowres_cond_video:
                    loss, cond_video = loss_obj
                else:
                    loss = loss_obj
                    cond_video = None

                loss = loss * chunk_size_frac

                if cond_video is not None:
                    cond_videos.append(cond_video)

            total_loss.append(loss)

            if self.training and not self.accum_grad:
                self.accelerator.backward(loss)

        total_loss = torch.cat(total_loss).mean()

        if self.training and self.accum_grad:
            self.accelerator.backward(total_loss)

        if return_lowres_cond_video:
            return total_loss.item(), torch.stack(cond_videos)
        else:
            return total_loss.item()
