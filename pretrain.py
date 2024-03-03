"""Script for a pretraining run."""

import torch
from torch import nn
import torch.nn.functional as F
import hydra

import numpy as np

import os
import time
import datetime
import logging
from collections import defaultdict

import cramming

from cramming.backend import _load_optimizer

import up, random, wandb, gc, math, copy, tqdm

from up.util import d, sample, gradient_norm, tic, toc
from up.data import load_data, cas

from tqdm import trange
from collections import Counter

# Logs
log = logging.getLogger(__name__)

# Different logs
LOG2E = math.log2(math.e)
LOGE2 = math.log(2.0)

def mask_batch(inputs=None, num_tokens=32768, special_tokens_mask=None, mlm_probability=.15, use_80_20_rule=True,
               mask_token=4,
            ):
        """
        -- Modified from backed/utils to remove the OO/dataloader parts.

        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        The ratios in this version are always fixed so that the number of masks is never dynamic!

        Also special_tokens_masks are disregarded in this flavor

        According to timeit this is not slower than the old approach (with was fast enough)
        """
        labels = inputs.clone() # prediction target, the unmasked input
        # -- NB non-manipulated tokens are masked out below (by setting the target to -100).

        number_of_masks = round(mlm_probability * inputs.shape[1])
        mask_locations = torch.argsort(torch.randint_like(inputs, inputs.shape[1]))[:, :number_of_masks]
        # this was slightly fudged to be faster. A draw of torch.rand would be more random, but take slightly longer to sort

        masked_indices = torch.zeros_like(inputs, dtype=torch.bool)
        masked_indices.scatter_(1, mask_locations, 1)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # -- Note that -100 is the default ignore_index in the CrossEntropyLoss
        #    https: // pytorch.org / docs / stable / generated / torch.nn.CrossEntropyLoss.html

        if use_80_20_rule:
            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            first_80percent_mask_locations = mask_locations[:, : round(0.8 * number_of_masks)]

            indices_replaced = torch.zeros_like(inputs, dtype=torch.bool)
            indices_replaced.scatter_(1, first_80percent_mask_locations, 1)
            inputs[indices_replaced] = mask_token

            # 10% of the time, we replace masked input tokens with random word
            next_10percent_mask_locations = mask_locations[:, round(0.8 * number_of_masks) : round(0.9 * number_of_masks)]

            indices_random = torch.zeros_like(inputs, dtype=torch.bool)
            indices_random.scatter_(1, next_10percent_mask_locations, 1)

            random_words = torch.randint(num_tokens, labels.shape, dtype=inputs.dtype, device=d())
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            # -- Note that these are different from the unmasked tokens in that we _do_ compute a loss over them.
            pass
        else:
            # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            inputs[masked_indices] = mask_token

        # This is not used in the default setting
        # if self.token_drop > 0:
        #     inputs, labels = self._drop_tokens(inputs, labels)

        return inputs, labels

def set_lr(lr, opt):
    for g in opt.param_groups:
        g['lr'] = lr
        g['initial_lr'] = lr

def data_generator(num_tokens, cfg):
    """
    A generator for batches of random data.
    :param cfg:
    :return:
    """
    distill = cfg.up.transfer == 'distill'
    context = cfg.arch.embedding.max_seq_length

    if cfg.up.source_mode.startswith('nn'):
        # randomness source model

        source = up.GTransformer(
            emb=cfg.arch.embedding.embedding_dim,
            heads=cfg.arch.attention.num_attention_heads,
            depth=cfg.up.source_layers,
            seq_length=cfg.data.seq_length,
            num_tokens=num_tokens,
            nl=up.util.nl(cfg.up.nonlinearity),
            mask_channel=False,
            autoregressive=not cfg.up.bid_source
        )

        if torch.cuda.is_available():
            source.cuda()
        if cfg.up.dp:
            source = torch.nn.DataParallel(source)

        print('-- source model:')
        print(source)
        print()

        # Add one output channel to the source model for the masking.
        # i, o = source.decoder.in_features, source.decoder.out_features
        # source.decoder = nn.Linear(i, o + 1)

    if cfg.up.source_mode == 'nn':
        buffer = \
            torch.randn(size=(cfg.up.buffer_size, context, num_tokens)) if distill else \
            torch.randint(low=0, high=num_tokens, size=(cfg.up.buffer_size, context))
        # -- In distill mode, the buffer stores all logits that the source model produced. Otherwise, we just store a
        #    sample of the tokens.


    num = 0

    while True:
        with (torch.no_grad()):

            # Process the buffer
            if cfg.up.source_mode == 'nn':
                tic()

                # Re-initialize the parameters of source (i.e. sample a random source)
                if cfg.up.init_mode == 'default':
                    up.weights_init(source, init_mult_max=cfg.up.init_mult_max, mask_prob_max=cfg.up.mask_prob_max)
                elif cfg.up.init_mode == 'plain':
                    up.weights_init_plain(source, init_mult_max=cfg.up.init_mult_max, mask_prob_max=cfg.up.mask_prob_max)
                elif cfg.up.init_mode == 'minimal':
                    up.weights_init_minimal(source, init_mult_max=cfg.up.init_mult_max)
                else:
                    raise

                # Slice a random selection of rows from the buffer (without replacement)
                iz = random.sample(range(buffer.size(0)), cfg.up.sample_batch_size)
                z = buffer[iz].to(d())

                # Replace some random instances with uniform random characters, or random logits in distillation mode
                rows = torch.bernoulli(torch.full(size=(cfg.up.sample_batch_size,), fill_value=cfg.up.reset_prob)).to(torch.bool)
                mask = \
                    rows[:, None, None].expand(cfg.up.sample_batch_size, context, num_tokens) if distill else \
                    rows[:, None].expand(cfg.up.sample_batch_size, context)

                noise = \
                    torch.randn(size=(cfg.up.sample_batch_size, context, num_tokens), device=d()) if distill else \
                    torch.randint(low=0, high=num_tokens, size=(cfg.up.sample_batch_size, context), device=d())
                # torch.randint(low=0, high=num_tokens, size=(cfg.up.sample_batch_size, context), device=d())

                z[mask] = noise[mask]
                if distill:
                    z = sample(z, temperature=cfg.up.temperature)

                # Pass it through the source
                output = source(z)

                # output = source(z)['outputs'].view(cfg.up.sample_batch_size, context, -1)
                # chars, mask = output[:, :, :-1], output[:, :, -1]

                if not distill:
                    output = sample(output, temperature=cfg.up.temperature)

                # mask = torch.sigmoid(mask).to(torch.bool)
                #
                # z[mask] = chars[mask] # replace the masked part of the input by the output samples

                buffer[iz, :] = output.to('cpu')     # replace the inputs in the buffer

                # -- Note that the samples are in full precision. These often require large weights, so mixed precision
                #    leads to nans and infs and whatnot.

        num += 1
        if num > cfg.up.spinup:

            batch = None
            if cfg.up.source_mode == 'aut':
                # Sample from a probabilistic automaton
                batch = [up.data.gen_autseq(length=cfg.data.seq_length, vocab=cfg.data.vocab_size) for _ in
                         range(cfg.up.batch_size)]
                batch = torch.tensor(batch)

            elif cfg.up.source_mode == 'nn':
                # Perform a training step on batches sampled from the buffer
                # Sample a batch from the buffer
                iz = random.sample(range(cfg.up.buffer_size), cfg.up.batch_size)

                if distill:
                    logits = buffer[iz, :].detach().to(d())
                    batch = sample(logits, temperature=cfg.up.temperature)
                else:
                    batch = buffer[iz, :].detach().to(d())

            if cfg.up.source_mode == 'nnsimple':

                # We pick a weight multiplier uniformly in log-space
                # logwm = random.random() * math.log(cfg.up.init_mult_max) + 1
                up.weights_init_minimal(source, cfg.up.init_mult_max)

                input = torch.randint(low=0, high=num_tokens, size=(cfg.up.batch_size, context), device=d())

                logits = source(input)
                batch = sample(logits, temperature=cfg.up.temperature)


            yield batch


def pretrain(cfg, setup):

    print('Start universal pretraining.')

    # Load the datasets for ood evaluation
    if cfg.up.eval_ood_every > 0:
        print('Loading data.')
        datasets = {
            'dyck'  : torch.tensor(load_data('dyck', char_offset=10), dtype=torch.long),
            'ndfa'  : torch.tensor(load_data('ndfa', char_offset=10), dtype=torch.long),
            'toy'   : torch.tensor(load_data('toy', char_offset=10),  dtype=torch.long),
            'bits'  : torch.tensor(load_data('bits', char_offset=10), dtype=torch.long),
            'champ' : torch.tensor(load_data('champ', char_offset=10), dtype=torch.long),
            'wp'    : torch.tensor(load_data('wp-val', char_offset=10), dtype=torch.long)
        }
        # -- We offset the indices by 10 so that the tokens used don't overlap with the special tokens (in particular the
        #    masking tokens). Other than that, it doesn't really matter which tokens are used.

    scaler = torch.cuda.amp.GradScaler()

    # pre-training target model
    model = cramming.construct_model(cfg.arch, cfg.data.vocab_size)
    num_tokens = model.encoder.embedding.word_embedding.num_embeddings

    datagen = data_generator(num_tokens, cfg)

    if cfg.up.reuse_opt:
        opt, _ = _load_optimizer(model, cfg.train, cfg.impl, initial_time=0)
        # -- `initial_time` is only used for the scheduler (second return value), which we discard.

        # Reset the learning rate to the correct UP learning rate.
        print('setting learning rate to ', cfg.up.lr)
        for g in opt.param_groups:
            g['lr'] = cfg.up.lr
            g['initial_lr'] = cfg.up.lr
            g['weight_decay'] = cfg.up.weight_decay
            g['betas'] = cfg.up.betas

        print(opt)
    else:
        opt = torch.optim.Adam(lr=cfg.up.lr, params=model.parameters())

    if cfg.up.warmup > 0:
        lr = 0.0
        set_lr(lr, opt)
        lr_delta = cfg.up.lr / cfg.up.warmup # -- By how much to increase the lr per instance

    if cfg.up.cooldown > 0:
        cd_delta = cfg.up.lr / cfg.up.cooldown
        # -- By how much to cool down the lr (per instance) after the peak is reached
        cd_start = (cfg.up.num_batches * cfg.up.batch_size) - cfg.up.cooldown

    if cfg.up.acc_warmup > 0:
        acc = 1.0 # the macrobatch size
        acc_delta = (cfg.up.accumulate - 1) / cfg.up.acc_warmup
        # -- By how much to increase the warmup per instance
    else:
        acc = cfg.up.accumulate
    mbatch_size = 0 # size of the current macrobatch

    context = cfg.arch.embedding.max_seq_length

    if torch.cuda.is_available():
        model.cuda()

    if cfg.up.dp:
        model = torch.nn.DataParallel(model)

    # mean and variance of the gradient norm
    gnm, gnv = 0, 0
    seen = 0
    batch = None

    # Launch training
    for i in (bar := trange(cfg.up.num_batches)):

        tic()
        batch = next(datagen); sampletime = toc()

        if cfg.up.print_every > 0 and i % cfg.up.print_every == 0:
            for i in range(5):
                seq = batch[i].tolist()
                print('target', i)

                print(up.util.remap(seq))
                print()

        tic()
        if torch.cuda.is_available():
            batch = batch.cuda()

        # We use the MLM loss to train.
        inputs, targets = mask_batch(batch, mask_token=cfg.up.mask_token, mlm_probability=cfg.up.mlm_probability,
                                            use_80_20_rule=cfg.up.use_80_20_rule)

        with (torch.cuda.amp.autocast()):
            output = model(inputs)['outputs'].view(cfg.up.batch_size, context, -1)

            if cfg.up.transfer == 'discrete':
                loss = F.cross_entropy(output.transpose(2, 1), targets)
                # -- This looks like the loss is computed for all tokens, but the non-manipulated ones are set to
                #    -100 in 'targets', so that they get masked out.
            elif cfg.up.transfer == 'distill':
                assert cfg.up.source_mode == 'nnsimple' or cfg.up.source_mode == 'nn'

                # Compute the distill loss
                loss = F.cross_entropy(output.transpose(2, 1), F.softmax(logits.detach(), dim=-1).transpose(2, 1), reduction='none')

                # zero out the loss for the entries that were not manipulated in `mask_batch`.
                tomask = (targets == -100)
                assert tomask.size() == loss.size()
                loss[tomask] *= cfg.up.loss_mask_scale
                loss = loss.mean()
            else:
                raise

        scaler.scale(loss).backward()

        # Adaptive gradient clipping. We keep an exponential moving estimate of the mean and variance of the gradient
        # norm, and if the current norm is more than `cfg.up.gc` standard deviations above the mean, we clip it to
        # that value.
        gn = gradient_norm(model)
        lim = gnm + math.sqrt(gnv) * cfg.up.gc
        if i > 10 and gn > lim:
            nn.utils.clip_grad_norm_(model.parameters(), lim)

        gnm, gnv = em_meanvar(gn, gnm, gnv)

        mbatch_size += 1

        if mbatch_size > int(acc):  # perform a step

            scaler.step(opt)
            scaler.update()

            opt.zero_grad()
            set_lr(lr=min(lr, cfg.up.lr), opt=opt)

            mbatch_size = 0

        if cfg.up.acc_warmup and int(acc) < cfg.up.accumulate:
            acc += acc_delta * batch.size(0)
        if seen <= cfg.up.warmup: # warm up the learning rate
            lr  += lr_delta * batch.size(0)
        if cfg.up.cooldown > 0 and seen > cd_start: # cool down the learning rate
            lr  -= cd_delta * batch.size(0)

        seen += batch.size(0)
        traintime = toc()

        if cfg.wandb.enabled:
            wandb.log({
                'loss': loss,
                'learning_rate': opt.param_groups[0]['lr'],
                'gradient_norm': gn,
                'sample_time': sampletime,
                'train_time': traintime,
                'pre-training': 1.0,
                'ema_gn': gnm,
                'em_std_gn': math.sqrt(gnv),
                'clip': 1.0 if gn > lim else 0.0,
                'acc': acc
            })
        bar.set_postfix({'loss': f'{loss:.02}'})

        if cfg.up.eval_ood_every > 0 and (i - cfg.up.spinup) % cfg.up.eval_ood_every == 0:

            for name, data in datasets.items():
                print(f'evaluating {name}')

                with torch.no_grad():
                    est = estimate_compression(
                        model=model,
                        data=data,
                        nsamples=cfg.up.eval_samples,
                        context=cfg.data.seq_length,
                        batch_size=int(cfg.up.batch_size * 2.0)
                    )

                wandb.log({f'ood/val-{name}': est})

    opt.zero_grad()
    optimizer_to(opt, 'cpu')
    # -- We send the optimizer to the CPU. This avoids (?) issues with the optimizer states on GPU not being cleared,
    #    leading to OOM.

    if cfg.up.snapshot_file is not None:
        print(f'Saving snapshot to {cfg.up.snapshot_file}')
        torch.save({
            'model': model.state_dict(),
            'opt': opt.state_dict()
        }, cfg.up.snapshot_file)

    return model, opt


def estimate_compression(model, data, nsamples, context, batch_size, verbose=False, model_produces_logits=True, mask_token=4):
    """
    Estimates the averages bits/token for the masked out tokens in a sequence.

    :param model: An MLM style sequence-to-sequence model that takes as input a (sub) sequence of integer.
    :param data: A singe list of integers representing the data
    :return: The result of the computation in "bits per byte". That is, how many bits does the compressed representation
    spend on each byte (=ASCII character) of the raw data.
    """

    bits, tot = 0.0, 0
    batch = []

    # indices of target characters in the data
    gtargets = random.sample(range(data.size(0)), k=nsamples)

    # Buffer, every time it fills up, we run it through the model
    # -- After we pass the batch through the model, we look at only the probabilities predicted for the final token
    #    (for which the input is masked).
    target_indices = []

    for i, current in enumerate(tqdm.tqdm(gtargets) if verbose else gtargets):
        # current is the character to be predicted

        fr = max(0, current - context + 1)
        to = current + 1

        instance = data[fr:to].to(torch.long) # the subsequence of the data to add to the batch
        # -- slice out an instance of size context + 1 (or shorter at the start of the data)

        target_indices.append(instance.size(0) - 1) # index of the last element of the context

        if instance.size(0) < context:
            # the index in the output tensor of the character we want to predict
            # -- It's context + 1, because we clip off the last token as a target

            pad = torch.full(fill_value=mask_token, size=(context - instance.size(0),), dtype=torch.long)
            instance = torch.cat([instance, pad], dim=0)
            # -- the first tokens don't have enough tokens preceding them, so we pad them to the right size.

            assert instance.size(0) == context # all instances should be `context` long fater padding

        if torch.cuda.is_available():
            instance = instance.cuda()

        batch.append(instance[None, :])
        # -- We add a singleton dimension to concatenate along later.

        if len(batch) == batch_size or i == len(gtargets) - 1:
            # batch is full, or we are at the last instance, run it through the model

            b = len(batch)

            inputs = torch.cat(batch, dim=0)

            # mask out the last token
            targets = inputs[torch.arange(b, device=d()), target_indices]
            inputs[torch.arange(b, device=d()), target_indices] = mask_token

            assert targets.size() == (b,), f'{targets.size()=} should be {(b, )}'

            with torch.no_grad():
                if torch.cuda.is_available():
                    inputs = inputs.cuda()

                output = model(inputs)['outputs'].view(b, context, -1)

                if model_produces_logits:
                    output = F.log_softmax(output, dim=-1)

            assert output.size()[:2] == (b, context), f'was: {output.size()}, should be {(b, context)}'

            lnprobs = output[torch.arange(b, device=d()), target_indices, targets]
            log2probs = lnprobs * LOG2E
            # -- The model produces natural logarithms of probabilities, but we need base-2 logarithms of the
            #    probabilities, since these give us bits.

            bits += - log2probs.sum() # Add the bits for each character (the negative log_2 probabilties) to the running total
            batch, target_indices = [], []  # clear the buffer

    return bits.item() / nsamples # total nr of bits used

def main_training_process(cfg, setup):
    """This function controls the central training loop."""
    local_time = time.time()

    opt_sd = None
    if cfg.up.enabled:
        if cfg.up.snapshot is None:
            model, opt = pretrain(cfg, setup)
            opt_sd = opt.state_dict()

        else:
            print(f'Loading UP snapshot from file {cfg.up.snapshot}')
            dct = torch.load(cfg.up.snapshot)
            model_sd = dct['model']
            opt_sd = dct['opt']

            model = cramming.construct_model(cfg.arch, cfg.data.vocab_size)
            model.load_state_dict(model_sd)

        if cfg.up.up_mix > 0.0:
            # Preload a buffer of samples from the UP generator
            num_tokens = model.encoder.embedding.word_embedding.num_embeddings
            datagen = data_generator(num_tokens, cfg)

            print('Preloading rehearsal data.'); tic()
            rbatches = [next(datagen) for _ in range(cfg.up.nrehearsal)]
            rbuffer = torch.cat(rbatches, dim=0)
            print(f'Done. ({toc():.2}s).')

            # -- We preload the rehearsal data, rather than generating it on the fly, so that it doesn't cut into
            #    our DP training budget.
            rmix = cfg.up.up_mix

    else:
            model = cramming.construct_model(cfg.arch, cfg.data.vocab_size)

    dataset, tokenizer = cramming.load_pretraining_corpus(cfg.data, cfg.impl)
    checkpoint_rendevous = os.path.join(cfg.base_dir, cfg.name, "intermediate_state.pth")

    if cfg.impl.resume_run_after_preempt and os.path.isfile(checkpoint_rendevous):
        try:
            metadata = torch.load(checkpoint_rendevous, map_location=torch.device("cpu"))["metadata"]
            initial_step, elapsed_time = metadata["step"], metadata["elapsed"]
        except RuntimeError:
            log.info("Checkpoint file unreadable or corrupted.")
            os.remove(checkpoint_rendevous)
            initial_step, elapsed_time = 0, 0.0
    else:
        initial_step, elapsed_time = 0, 0.0

    model_engine, _, _, dataloader = cramming.load_backend(model, dataset, tokenizer, cfg.train, cfg.impl, elapsed_time, setup=setup)
    if cfg.impl.resume_run_after_preempt and os.path.isfile(checkpoint_rendevous):
        log.info(f"Loading intermediate checkpoint from previous run onto device {cfg.impl.local_rank}...")
        model_engine.load_training_checkpoint(checkpoint_rendevous)

    if cfg.up.enabled and cfg.up.reuse_opt:
        with torch.no_grad():
            assert opt_sd is not None

            print('state dict:')

            if cfg.up.opt_mult > 0.0:
                for val in opt_sd['state'].values():
                    val['exp_avg'] *= cfg.up.opt_mult
                    val['exp_avg_sq'] *= cfg.up.opt_mult

                    print('    min/max for exp. avg', val['exp_avg'].min(), val['exp_avg'].max())

            print()
            # -- Apply a multiplier to the exp moving average and the second moment. This can be seen as a convex
            #    combination of the fresh optimizer state (which is zero) and the optimizer state inherited from the
            #    universal pretraining.

            model_engine.optimizer.load_state_dict(opt_sd)

            # -- reuse the optimizer from the UP training

    model_engine.train(cfg.train.pretrain_in_train_mode)
    stats = defaultdict(list)

    # Start the clocks now:
    wallclock_timer = time.time() - elapsed_time
    train_time = time.time()
    training_allowed, no_recovery_necessary = True, True
    loss_vals = []

    # Launch training
    for step, batch in enumerate(dataloader, initial_step + 1):


        if rmix > 0.0:
            b, l = batch['input_ids'].size()
            k = int(rmix * b)

            if k > 0:
                bufferidx = random.sample(k=k, population=range(rbuffer.size(0)))
                batchidx  = random.sample(k=k, population=range(b))

                batch['input_ids'][batchidx] = rbuffer[bufferidx]

            rmix -= cfg.up.up_mix_decay

        # Heavy lifting is moved to engines
        device_batch = model_engine.to_device(batch)
        loss = model_engine.step(device_batch)
        loss_vals.append(loss.detach())

        if cfg.wandb.enabled:
            wandb.log({
                'dp-loss': loss.item(),
                'dp-lr': model_engine.optimizer.param_groups[0]['lr'],
                'rehearsal proportion': rmix,
            })

        # Check stopping criteria
        if check_deadline(wallclock_timer, cfg.budget) or step == cfg.train.steps:
            training_allowed = False
            log.info("Reached deadline. Stopping training ...")

        # Collect stats and print to console and upload to wandb
        if step % cfg.impl.print_loss_every_nth_step == 0:
            loss_vals, train_time = collect_stats(step, loss_vals, train_time, stats, model_engine, dataloader, cfg)
            if check_early_termination(wallclock_timer, stats["loss"][-1], cfg.impl.early_termination):
                training_allowed = False
                log.info("Loss higher than allowed threshold. Stopping training early...")

        # Checkpointing is triggered from stopping criteria and normal intervals
        if cfg.impl.save_intermediate_checkpoints and step % cfg.impl.save_every_nth_step == 0:
            if loss.detach().isfinite() and cramming.utils.is_main_process() and not cfg.dryrun:
                model_engine.save_training_checkpoint(checkpoint_rendevous, metadata=dict(step=step, elapsed=time.time() - wallclock_timer))

        if not loss.detach().isfinite():
            training_allowed, no_recovery_necessary = engage_troubleshooting(
                model_engine, step, training_allowed, no_recovery_necessary, cfg
            )

        communicate_flags(training_allowed, no_recovery_necessary)

        if (cfg.dryrun and step > 2) or not training_allowed:
            break

        if not no_recovery_necessary:  # synced across devices
            log.info(f"Attempting reload of checkpoint on device {cfg.impl.local_rank}.")
            model_engine.load_training_checkpoint(checkpoint_rendevous)
            no_recovery_necessary = True

    # Save to summary:
    cramming.utils.save_summary("pretrain", cfg, stats, time.time() - local_time, setup)
    if cramming.utils.is_main_process():
        # Save final checkpoint? Might have to recover the latest checkpoint first
        if not loss.detach().isfinite() and cfg.impl.save_intermediate_checkpoints:
            model_engine.load_training_checkpoint(checkpoint_rendevous)
            loss = torch.as_tensor(16.0)  # fake value for model file name
        if loss.detach().isfinite():
            now = datetime.datetime.now()
            long_checkpoint_id = f"{''.join(cfg.arch.architectures)}_{now.strftime('%Y-%m-%d')}_{loss:2.4f}"
            model_engine.save_final_model(os.path.join(cfg.base_dir, cfg.name), long_checkpoint_id, tokenizer, cfg.arch, cfg.dryrun)

            if cfg.impl.push_to_huggingface_hub:
                model_engine.push_to_hub(tokenizer, cfg, dryrun=cfg.dryrun)
    metrics = dict(num_params=sum([p.numel() for p in model.parameters()]))
    return metrics


def check_deadline(launch_time, hour_limit):
    """These measurements are deliberately wall-clock based."""
    current_time = time.time()
    return True if (current_time - launch_time) / 3600 > hour_limit else False


def check_early_termination(launch_time, loss, early_termination):
    """Early termination based on terrible loss."""
    if early_termination.enabled and loss > early_termination.loss_threshold:
        current_time = time.time()
        return True if (current_time - launch_time) / 3600 > early_termination.budget else False
    else:
        return False


def collect_stats(step, loss_vals, train_time, stats, model_engine, dataloader, cfg):
    stats["step"] += [step]
    stats["epoch"] += [dataloader.epoch_counter]

    tokens_per_step = model_engine.record_tokens_per_step()
    stats["tokens"] += [step * tokens_per_step]
    stats["loss"] += [torch.stack(loss_vals).mean().item()]  # Averaged loss

    current_lr = model_engine.optimizer.param_groups[0].get("lr", float("NaN"))
    log_msg = f"Train loss {loss_vals[-1].item():2.4f} at step {step} with lr {current_lr:.5f}. "
    log_msg += f"[Avg: {stats['loss'][-1]:2.4f}] "
    if step > 0:
        stats["train_time"] += [(time.time() - train_time) / cfg.impl.print_loss_every_nth_step]
        estimated_train_finish = str(datetime.timedelta(seconds=stats["train_time"][-1] * cfg.train.steps))
        tokens_per_second = tokens_per_step / stats["train_time"][-1]
        stats["tok/sec"] += [int(tokens_per_second)]
        log_msg += f" Perf: {stats['train_time'][-1]:2.4f}s per step ({tokens_per_second:.0f}t/s). "
        log_msg += f"Estimated Total Train: {estimated_train_finish}."

    # Adaptive optim stats
    stats["lr"] += [current_lr]
    stats["batch_size"] += [model_engine.record_batch_size()]
    stats["seq_length"] = [model_engine.current_seq_length]

    # Publish
    cramming.utils.wandb_log(stats, cfg)
    log.info(log_msg)

    # Clear:
    loss_vals = []
    train_time = time.time()
    return loss_vals, train_time


def engage_troubleshooting(model_engine, step, training_allowed, no_recovery_necessary, cfg):
    log.info(f"Non-finite loss in step {step} on device {cfg.impl.local_rank}.")

    is_finite_grad = [torch.isfinite(p.grad).all() for p in model_engine.model.parameters() if p.grad is not None]
    has_finite_gradients = torch.stack(is_finite_grad).all() if len(is_finite_grad) > 0 else True
    if not has_finite_gradients:
        if "dump_nan_grads" in cfg.impl.troubleshoot_strategy:
            log.info(f"Non-finite gradients in step {step} on device {cfg.impl.local_rank}, dumping...")
            model_engine.optimizer.zero_grad()
        else:
            if "recover_checkpoint" in cfg.impl.troubleshoot_strategy:
                no_recovery_necessary = False
            else:
                training_allowed = False
                log.info(f"Stopping training due to non-finite grads in step {step} on device {cfg.impl.local_rank}.")

    has_finite_parameters = torch.stack([torch.isfinite(p).all() for p in model_engine.model.parameters()]).all()
    if not has_finite_parameters:
        if "recover_checkpoint" in cfg.impl.troubleshoot_strategy:
            no_recovery_necessary = False
        else:
            training_allowed = False
            log.info(f"Stopping training due to non-finite parameters in step {step} on device {cfg.impl.local_rank}.")
    return training_allowed, no_recovery_necessary


def communicate_flags(training_allowed, no_recovery_necessary):
    """A quick and dirty communication through the comm protocol. Should not be a major burden."""
    if torch.distributed.is_initialized():
        comm_tensor_allowed = torch.as_tensor([training_allowed, no_recovery_necessary])
        comm_tensor_allowed = comm_tensor_allowed.cuda() if torch.cuda.is_available() else comm_tensor_allowed.float()
        torch.distributed.all_reduce(comm_tensor_allowed, torch.distributed.ReduceOp.MIN, async_op=False)
        if comm_tensor_allowed[0] >= 1:  # training indeed allowed on all devices
            return True, comm_tensor_allowed[1] > 0
        else:
            return False, True
    else:
        return training_allowed, no_recovery_necessary

def em_meanvar(x, mean=0, variance=0, alpha=0.5):
    """
    Computes exp. moving average and variance.

    ```
    mean, variance = 0, 0
    for x in values:
       mean, variance = em_meanvar(x, mean, variance, alpha=0.5)
    ```

    source: Incremental calculation of weighted mean and variance (Tony Finch, 2009)
    """
    diff = x - mean
    incr = alpha * diff

    newmean = mean + incr
    newvar = (1 - alpha) * (variance + diff * incr)
    if not (np.isfinite(newmean) and np.isfinite(newvar)):
        # skip x if it causes infs or nans
        return mean, variance

    return newmean, newvar

@hydra.main(config_path="cramming/config", config_name="cfg_pretrain", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_training_process, job_name="pretraining")

def optimizer_to(optim, device):
    """
    Move optimizer to a given device. From https://github.com/pytorch/pytorch/issues/7415#issuecomment-693424574
    :param optim:
    :param device:
    :return: nothing, modifies the optimizer in place.
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

if __name__ == "__main__":
    launch()