"""Script for a pretraining run."""

import torch
from torch import nn
import torch.nn.functional as F
import hydra

import numpy as np

import os, re
import time
import datetime
import logging

from collections import defaultdict, deque
from collections.abc import Iterable

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

LOSS_EMA_START = 10.0 # start value for the loss EMA
LOSS_EMA_GAMMA = 0.9  # mixture parameter for the loss EMA

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

            random_words = torch.randint(num_tokens, labels.shape, dtype=inputs.dtype, device=inputs.device)
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

def lstm_scale(lstm : nn.LSTM, weight_mult=1.0, bias_mult=1.0):

    l = lstm.num_layers

    for k in range(l):
        for wlist in lstm.all_weights:
            for w in wlist:
                w.data *= weight_mult

        for b in getattr(lstm, 'bias_ih_l'+ str(k)), getattr(lstm, 'bias_hh_l'+ str(k)):
            b.data *= bias_mult

def rand_batch(length, con, ran, num_tokens):
    """
    Generate a batch of constant and random instances.

    :param const: # of constant instances
    :param rand: # of random instances
    :param length: Length of the instances
    :return:
    """
    crows = torch.randint(low=0, high=num_tokens, size=(con, 1))
    crows = crows.tile((1, length))
    rrows = torch.randint(low=0, high=num_tokens, size=(ran, length))

    rows = torch.cat((crows, rrows), dim=0)

    return rows

def data_generator(num_tokens, cfg):
    """
    A generator for batches of random data.
    :param cfg:
    :return:
    """

    # -- TODO: LSTMs

    distill = cfg.up.transfer == 'distill'
    context = cfg.arch.embedding.max_seq_length

    if cfg.up.source_mode.startswith('lstm'):

        source = up.LSTMGen(cfg.up.lstmemb, mask_channel=False, num_tokens=num_tokens, layers=cfg.up.lstmlayers)

        lstmdev = 'cuda' if torch.cuda.is_available() else 'cpu'
        source.to(lstmdev)

        buffer = torch.randint(low=0, high= num_tokens, size=(cfg.up.buffer_size, 1), device=lstmdev)
        buffer = buffer.tile((1, context))

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
            if cfg.up.source_mode == 'lstm':

                # replace some random rows in the buffer with constant and random sequences
                con, ran = cfg.up.lstmreset

                rows = rand_batch(context, con, ran, num_tokens).to('cuda')

                idx = random.sample(range(cfg.up.buffer_size), rows.size(0))
                buffer[idx] = rows

                # Re-initialize the source
                source.reset_parameters()
                mult_sample = np.random.uniform(*cfg.up.lstmmult)

                # print(f'mult {mult_sample:.4} \t temp {np.log10(temp_sample):.4}')
                lstm_scale(source.lstm, mult_sample)

                source.token_embedding.weight.data *= cfg.up.lstmembmult
                source.toprobs.weight.data *= cfg.up.lstmlinmult

                # slice a random selection of rows from the buffer (without replacement)
                iseeds = random.sample(range(buffer.size(0)), cfg.up.sample_batch_size)
                iconds = random.sample(range(buffer.size(0)), cfg.up.sample_batch_size)

                s = random.randrange(0, context - cfg.up.lstmseed)
                seeds = buffer[iseeds, s:s + cfg.up.lstmseed]
                conds = buffer[iconds, :]

                chars = source.sample_sequence(seed=seeds,
                                                max_context=context, num_tokens=num_tokens,
                                                length=context - seeds.size(1), temperature=cfg.up.lstmtemp,
                                                conditional=conds)

                buffer[iconds, :] = chars

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

            if cfg.up.source_mode == 'lstm':

                ibatch = random.sample(range(cfg.up.buffer_size), cfg.up.batch_size)
                batch = buffer[ibatch, :]

            yield batch

class AdWrap(nn.Module):

    def __init__(self, adapter):
        super().__init__()

        self.ad = adapter
        self.mult = nn.Parameter(torch.tensor(0.0))

    def forward(self, states, mask):
        return states + self.ad(states) * self.mult

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

    # Take out certain layers to be re-inserted as adapters after UP
    adapters, layers = deque(), []
    for mode, layer in zip(cfg.up.pattern, model.encoder.layers):
        if mode == 'u':
            layers.append(layer) # becomes a layer in the pre-training model
        elif mode == 'a':
            adapters.append(layer.to('cpu')) # store for later

    model.encoder.layers = nn.ModuleList(layers)

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
        # TODO: Remove this and always use cramming opt

    # for i, layer in enumerate(model.encoder.layers):
    #     if i in cfg.up.freeze_layers:
    #         print('Freezing layer', i)
    #         for parm in list(layer.attn.parameters()) + list(layer.ffn.parameters()):
    #             parm.requires_grad = False

    if cfg.up.warmup > 0:
        lr = 0.0
        set_lr(lr, opt)
        lr_delta = cfg.up.lr / cfg.up.warmup # -- By how much to increase the lr per instance

    if cfg.up.cooldown > -1:

        cooldown_rate = 0.5 ** (1 / cfg.up.cooldown)
        last_cooldown = cfg.up.warmup

        # cd_delta = cfg.up.lr / cfg.up.cooldown
        # # -- By how much to cool down the lr (per instance) after the peak is reached
        # cd_start = (cfg.up.num_batches * cfg.up.batch_size) - cfg.up.cooldown

    accumulated = 0 # nr of instances accumulated currently
    # cfg.up.accumulate = macrobatch_size
    if cfg.up.acc_warmup > 0:
        mbraw =  cfg.up.batch_size # the macrobatch size (starts eq to the microbatch size)
    else:
        mbraw = cfg.up.accumulate # macrobatch size

    accumulated = 0 # nr of instances accumulated currently

    context = cfg.arch.embedding.max_seq_length

    if torch.cuda.is_available():
        model.cuda()

    if cfg.up.dp:
        model = torch.nn.DataParallel(model)

    seen = 0
    batch = None

    # Launch training
    for i in (bar := trange(cfg.up.num_batches)):

        tic()
        batch = next(datagen); sampletime = toc()

        if cfg.up.print_every > 0 and i % cfg.up.print_every == 0:
            for i in range(min(5, batch.size(0))):
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

                rloss = F.cross_entropy(output.transpose(2, 1), targets, reduction='sum')
                # -- This looks like the loss is computed for all tokens, but the non-manipulated ones are set to
                #    -100 in 'targets', so that they get masked out.

                loss = (rloss / inputs.size(1))
                # -- We divide out the time, but sum over the instances

            elif cfg.up.transfer == 'distill':
                assert cfg.up.source_mode == 'nnsimple' or cfg.up.source_mode == 'nn'

                # Compute the distill loss
                loss = F.cross_entropy(output.transpose(2, 1), F.softmax(logits.detach(), dim=-1).transpose(2, 1), reduction='none')

                # zero out the loss for the entries that were not manipulated in `mask_batch`.
                tomask = (targets == -100)
                assert tomask.size() == loss.size()
                loss[tomask] *= cfg.up.loss_mask_scale
                loss = loss.mean()

                # -- Distillation works ok, but it's too expensive compared to data-only

            else:
                raise

        scaler.scale(loss).backward()
        accumulated += inputs.size(0)

        if accumulated >= mbraw: # perform a step

            scaler.unscale_(opt)

            # scale the gradients to average over the macrobatch
            # -- here we divide out the instances
            for parm in model.parameters():
                if parm.grad is not None:
                    parm.grad /= accumulated

            gn = gradient_norm(model)
            if cfg.up.gc > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.up.gc)

            if cfg.wandb.enabled:
                wandb.log({
                    'gradient_norm': gn,
                    'accumulated': accumulated # Sanity check.
                }, step=seen)

            scaler.step(opt)
            scaler.update()

            opt.zero_grad()

            accumulated = 0

        seen += batch.size(0)
        traintime = toc()

        # Admin

        # Set accumulation target
        if seen <= cfg.up.acc_start:
            mbraw = cfg.up.batch_size
        elif cfg.up.acc_start <= seen < cfg.up.acc_warmup + cfg.up.acc_start:
            prop = (seen - cfg.up.acc_start) / cfg.up.acc_warmup
            mbraw = cfg.up.batch_size + (cfg.up.accumulate - cfg.up.batch_size) * prop
        else:
            assert seen >= cfg.up.acc_warmup + cfg.up.acc_start
            mbraw = cfg.up.accumulate

        # Set LR
        if cfg.up.warmup > 0 and seen <= cfg.up.warmup:
            set_lr(lr=lr_delta * seen, opt=opt)

        else:
            if cfg.up.cooldown > 0:
                since_wu = seen - cfg.up.warmup
                set_lr(lr=cfg.up.lr * cooldown_rate ** since_wu, opt=opt)

        # if cfg.up.acc_warmup and int(acc) < cfg.up.accumulate:
        #     acc += acc_delta * batch.size(0)
        # if seen <= cfg.up.warmup: # warm up the learning rate
        #     lr  += lr_delta * batch.size(0)
        # if cfg.up.cooldown > 0 and seen > cd_start: # cool down the learning rate
        #     lr  -= cd_delta * batch.size(0)

        # Logging
        if cfg.wandb.enabled:
            wandb.log({
                'loss': loss,
                'learning_rate': opt.param_groups[0]['lr'],
                'sample_time': sampletime,
                'train_time': traintime,
                'pre-training': 1.0,
                'acc': mbraw
            }, step=seen)

        bar.set_postfix({'loss': f'{loss:.02}'})

        # Eval
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
    # optimizer_to(opt, 'cpu')
    # # -- We send the optimizer to the CPU. This avoids (?) issues with the optimizer states on GPU not being cleared,
    # #    leading to OOM.

    if cfg.up.snapshot_file is not None:
        print(f'Saving snapshot to {cfg.up.snapshot_file}')
        torch.save({
            'model': model.state_dict(),
            'opt': opt.state_dict()
        }, cfg.up.snapshot_file)

    if cfg.up.up_only:
        print('Stopping process after UP phase.')
        exit()
        # -- not pretty, but whatever.

    # Restore the untrained (adapter) layers
    current = 0 # index of the 'u' layer before which we're inserting
    newparms = []
    for mode in cfg.up.pattern:
        if mode == 'u':
            current += 1
        elif mode == 'a':
            # get the adapter from the store, move to cuda
            adapter = adapters.popleft().to('cuda')
            adapter = AdWrap(adapter)

            model.encoder.layers.insert(current, adapter)
            newparms.extend(adapter.parameters())

            current += 1
        else:
            raise

    print('After inserting Adapters.')
    print(model)

    opt.add_param_group({'params': newparms})

    # # Unfreeze layers
    # for i, layer in enumerate(model.encoder.layers):
    #     if i in cfg.up.freeze_layers:
    #         print('Unfreezing layer', i)
    #         for parm in list(layer.attn.parameters()) + list(layer.ffn.parameters()):
    #             parm.requires_grad = True

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
    """
    This function controls the central training loop.
    """

    context = cfg.arch.embedding.max_seq_length

    # Log env vars to wandb
    for key in os.environ:
        if re.match(r"^NCCL|CUDA|PATH|^LD|USER|PWD|SLURM", key):
            wandb.config[key] = os.getenv(key)

    local_time = time.time()

    opt_sd = None
    rmix = -1.0
    use_alpha = False

    if cfg.up.enabled:

        if cfg.up.snapshot is None:
            print(f'Pretraining UP model')

            upmodel, opt = pretrain(cfg, setup)

            # opt_sd = opt.state_dict()

        else:
            print(f'Loading UP snapshot from file {cfg.up.snapshot}')
            dct = torch.load(cfg.up.snapshot)
            model_sd = dct['model']
            opt_sd = dct['opt']

            upmodel = cramming.construct_model(cfg.arch, cfg.data.vocab_size)
            upmodel.load_state_dict(model_sd)

        if cfg.up.up_mix > 0.0: # rehearsal data
            # Preload a buffer of samples from the UP generator
            num_tokens = upmodel.encoder.embedding.word_embedding.num_embeddings
            datagen = data_generator(num_tokens, cfg)

            print('Preloading rehearsal data.'); tic()
            rbatches = [next(datagen) for _ in range(cfg.up.nrehearsal)]
            rbuffer = torch.cat(rbatches, dim=0).to('cpu')
            print(f'Done. ({toc():.2}s).')

            # -- We preload the rehearsal data, rather than generating it on the fly, so that it doesn't cut into
            #    our DP training budget.
            rmix = cfg.up.up_mix

            # if cfg.up.snapshot is not None:
            #     REPS, BATCH = 500, 32
            #     # Check the loss of the snapshot on the researsal data
            #     loss = 0.0
            #
            #     model.to('cuda')
            #
            #     with torch.no_grad():
            #         for _ in trange(REPS):
            #             # sample a batch
            #             bidx = random.sample(k=BATCH, population=range(rbuffer.size(0)))
            #             batch = rbuffer[bidx].to(d())
            #
            #             inputs, targets = mask_batch(batch, mask_token=cfg.up.mask_token,
            #                                          mlm_probability=cfg.up.mlm_probability,
            #                                          use_80_20_rule=cfg.up.use_80_20_rule)
            #
            #             inputs, targets = inputs.to(d()), targets.to(d())
            #
            #             output = model(inputs)['outputs'].view(BATCH, context, -1)
            #
            #             loss += F.cross_entropy(output.transpose(2, 1), targets).item()
            #
            #         loss /= REPS
            #         print(f'Estimated model loss on rehearsal buffer: {loss:.4} nats/token.')

        if cfg.up.mode == 'none':
            model = cramming.construct_model(cfg.arch, cfg.data.vocab_size)
            del upmodel

            use_alpha = False

        elif cfg.up.mode == 'norm':

            # # Freeze the UP model
            # for parm in upmodel.parameters():
            #     parm.requires_grad = False
            # upmodel.to(d())
            #
            # model = cramming.construct_model(cfg.arch, cfg.data.vocab_size)
            # use_alpha = True
            raise # -- too slow

        elif cfg.up.mode == 'init':

            if cfg.up.aux_alpha <= 0.0:
                model = upmodel
                print("Model set.")
            else:
                model = cramming.construct_model(cfg.arch, cfg.data.vocab_size)

                # Pick a halfway point between the raw initialized model and the pretrained one
                a = cfg.up.aux_alpha
                for mparm, uparm in zip(model.parameters(), upmodel.parameters()):
                    mparm.data = mparm.data * (1 - a) + uparm.data * a

            use_alpha = False

        elif cfg.up.mode == 'distill':
            # print('Using distillation mode.')
            #
            # model = cramming.construct_model(cfg.arch, cfg.data.vocab_size)
            # upmodel.to(d())
            #
            # if cfg.impl.compile_torch:
            #     upmodel = torch.compile(upmodel)
            #
            # use_alpha = True
            raise # -- too slow

        else:
            raise ValueError(f'Transfer mode {cfg.up.mode} not recognized.')

    else:
        model = cramming.construct_model(cfg.arch, cfg.data.vocab_size)

    dataset, tokenizer = cramming.load_pretraining_corpus(cfg.data, cfg.impl, cfg.up.data_path)
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

    model_engine.wandb = wandb
    model_engine.which_aux = cfg.up.which_aux

    if cfg.impl.resume_run_after_preempt and os.path.isfile(checkpoint_rendevous):
        log.info(f"Loading intermediate checkpoint from previous run onto device {cfg.impl.local_rank}...")
        model_engine.load_training_checkpoint(checkpoint_rendevous)

    loss_ema = LOSS_EMA_START

    if cfg.up.enabled and cfg.up.reuse_opt: # transfer opt state
        with torch.no_grad():
            # assert opt_sd is not None
            #
            # print('state dict:')
            #
            # if cfg.up.opt_mult > 0.0:
            #     for val in opt_sd['state'].values():
            #         val['exp_avg'] *= cfg.up.opt_mult
            #         val['exp_avg_sq'] *= cfg.up.opt_mult
            #
            #         print('    min/max for exp. avg', val['exp_avg'].min(), val['exp_avg'].max())
            #
            # print()
            # # -- Apply a multiplier to the exp moving average and the second moment. This can be seen as a convex
            # #    combination of the fresh optimizer state (which is zero) and the optimizer state inherited from the
            # #    universal pretraining.

            # print(len(cfg.train.limited_decay_keys))
            # print('target state dict')
            # print(model_engine.optimizer.state_dict()['param_groups'])
            # print()
            # print('source state dict')
            # print(opt_sd['param_groups'])

            # # Reuse the optimizer from the UP training
            # model_engine.optimizer.load_state_dict(opt_sd)
            # # -- If this fails, it may be due to the weight decay being set to 0.0 for this run but not for the
            # #    pretraining run (or vice versa). This seems to result in different numbers of parameter groups.

            model_engine.optimizer = opt # Does this work?

    # Reset betas to the value given in the CL params
    if cfg.up.reset_betas:
        model_engine.optimizer.betas = cfg.up.betas
        for g in model_engine.optimizer.param_groups:
            g['betas'] = cfg.up.betas

    # Reset weight decay to the value given in the cramming params
    if cfg.up.reset_wd:
        model_engine.optimizer.weight_decay = cfg.train.optim.weight_decay
        for g in model_engine.optimizer.param_groups:
            g['weight_decay'] = cfg.train.optim.weight_decay

    print('optimizer:')
    for i, g in enumerate(model_engine.optimizer.param_groups):
        print('    group', i)
        for k,v in g.items():
            if type(v) is not torch.Tensor:
                print(k, v)
        #
        # print('    lr', g['lr'])
        # print('    initial lr', g['initial_lr'])
        # print('    weight_decay', g['weight_decay'])
        # print('    betas', g['betas'])

    model_engine.train(cfg.train.pretrain_in_train_mode)
    stats = defaultdict(list)

    # Start the clocks now:
    wallclock_timer = time.time() - elapsed_time
    train_time = time.time()
    training_allowed, no_recovery_necessary = True, True
    loss_vals = []

    # Alpha warmup
    if isinstance(cfg.up.alpha_warmup, Iterable): # start and end point
        awu_from, awu_to = cfg.up.alpha_warmup
        awu = True
    elif cfg.up.alpha_warmup > 0: # end point only
        awu_from, awu_to = 0.0, cfg.up.alpha_warmup
        awu = True
    else:
        awu = False

    # Alpha cooldown
    if isinstance(cfg.up.alpha_cooldown, Iterable): # start and end point
        acd_from, acd_to = cfg.up.alpha_cooldown
        acd = True
    elif cfg.up.alpha_cooldown > 0: # start point only
        print(type(cfg.up.alpha_cooldown))
        acd_from, acd_to = cfg.up.alpha_cooldown, 1.0
        acd = True
    else:
        acd = False

    # Launch training
    for step, batch in enumerate(dataloader, initial_step + 1):

        b, l = batch['input_ids'].size()

        batchmodded = False

        # Mix in rehearsal data
        if rmix > 0.0:

            # Select some random ids in the batch
            idx = torch.bernoulli(torch.full(fill_value=rmix, size=(b, )))
            k = int(idx.sum().item())

            if k > 0:
                idx = idx.nonzero() # convert to indices
                # idx = idx[:, None].to(torch.bool).expand(b, l)

                # And the same number of random ids in the buffer
                # bidx = torch.full(fill_value=0.0, size=(rbuffer.size(0), ))
                bidx = torch.tensor(random.sample(k=k, population=range(rbuffer.size(0))))
                # bidx = bidx[:, None].to(torch.bool).expand(rbuffer.size(0), l)

                upbatch = rbuffer[bidx, :]
                upinputs, uptargets = mask_batch(upbatch, mask_token=cfg.up.mask_token, mlm_probability=cfg.up.mlm_probability,
                                             use_80_20_rule=cfg.up.use_80_20_rule)

                # print(batch['input_ids'][idx].size(), upinputs.size())

                batch['input_ids'][idx] = upinputs[:, None, :]
                batch['labels'][idx] = uptargets[:, None, :]

                batchmodded = True

            rmix -= cfg.up.up_mix_decay

        # Heavy lifting is moved to engines
        device_batch = model_engine.to_device(batch)

        prop = timeprop(wallclock_timer, cfg.budget)
        alphamult = 1.0

        if awu: # alpha warmup
            if prop <= awu_from:
                alphamult = cfg.up.log_alpha_min if cfg.up.use_log_alpha else 0.0
            if awu_from < prop <= awu_to:
                alphamult = (prop - awu_from) / (awu_to - awu_from)
            if prop > awu_to:
                alphamult = 1.0

        if acd: # alpha cooldown
            if prop <= acd_from:
                alphamult = 1.0
            if acd_from < prop <= acd_to:
                alphamult = (acd_to - prop) / (acd_to - acd_from)
            if prop > acd_to:
                alphamult = cfg.up.log_alpha_min if cfg.up.use_log_alpha else 0.0

        if cfg.up.use_log_alpha:
            minexp = np.log10(cfg.up.log_alpha_min)
            alphamult = 10.0 ** (0 * alphamult + minexp * (1 - alphamult))

        guide = None

        if cfg.up.enabled:
            if cfg.up.mode == 'norm':
                guide = upmodel
            elif cfg.up.mode == 'distill':
                with torch.no_grad():
                    output = upmodel(batch['input_ids'].to(d()))['outputs']
                    output = output.reshape(b, l, -1) # Hope this is right ...

                    guide = output.softmax(dim=-1)

        loss = model_engine.step(device_batch,
                                 guide=guide,
                                 alpha=0.0 if not use_alpha else alphamult * cfg.up.aux_alpha,
                                 mode=cfg.up.mode)

        # -- Includes both the forward and the backward.
        # -- Note the above relies on the fact that exactly 25% of tokens are masked. The loss is then computed sparsely
        #    over just these tokens to speed up processing.

        with torch.no_grad():
            sl = int(l * .25)
            loss = loss.reshape(b, sl)

            up_loss = loss[idx, :].sum() / (k * sl) if batchmodded else torch.tensor(0.0)
            pile_loss = (loss.sum() - loss[idx, :].sum()) / ((b - k) * sl) if batchmodded else loss.mean()
            # -- Extract the loss only over the UP part of the data and only over the pile part of the data.

        loss = loss.mean()

        loss_ema = loss_ema * LOSS_EMA_GAMMA + loss.item() * (1.0 - LOSS_EMA_GAMMA)

        loss_vals.append(loss.detach())

        if cfg.wandb.enabled:
            wandb.log({
                'dp-loss': loss.item(),
                'dp-loss-ema': loss_ema,
                'dp-gn': gradient_norm(model),
                'dp-lr': model_engine.optimizer.param_groups[0]['lr'],
                'rehearsal proportion': rmix,
                'alpha_mult': alphamult
            })

            if rmix > 0.0:
                wandb.log({
                    'dp-loss-up': up_loss.item(),
                    'dp-loss-pile': pile_loss.item(),
                })

        if cfg.up.early_stop > 0.0 and loss_ema < cfg.up.early_stop:
            training_allowed = False
            log.info(f"Reached early stopping threshold (loss {loss}, ema {loss_ema}). Stopping training ...")

        # Check stopping criteria
        if check_deadline(wallclock_timer, cfg.budget) or step == cfg.train.steps:
            training_allowed = False
            log.info("Reached deadline. Stopping training ...")

        # Collect stats and print to console and upload to wandb
        if not cfg.up.manual:
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

        if not cfg.up.manual:
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


def timeprop(launch_time, hour_limit):
    """The passed time, as a proportion of the total budget."""
    current_time = time.time()
    return  ((current_time - launch_time) / 3600) / hour_limit


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