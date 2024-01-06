"""Script for a pretraining run."""

import torch
from torch import nn
import torch.nn.functional as F
import hydra

import os
import time
import datetime
import logging
from collections import defaultdict

import cramming

from cramming.backend import _load_optimizer

import up, random, wandb, gc

from up.util import d, sample, gradient_norm, tic, toc

from tqdm import trange

log = logging.getLogger(__name__)

def mask_batch(inputs=None, num_tokens=32768, special_tokens_mask=None, mlm_probability=.15, use_80_20_rule=True, mask_token=4):
        """
        -- Modified from backed/utils to remove the OO/dataloader parts.

        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        The ratios in this version are always fixed so that the number of masks is never dynamic!

        Also special_tokens_masks are disregarded in this flavor

        According to timeit this is not slower than the old approach (with was fast enough)
        """
        labels = inputs.clone() # prediction target, the unmasked input

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

def pretrain(cfg, setup):

    print('Start universal pretraining.')

    scaler = torch.cuda.amp.GradScaler()

    # randomness source model
    source = cramming.construct_model(cfg.arch, cfg.data.vocab_size)

    # Add one output channel to the source model for the masking.
    i, o = source.decoder.in_features, source.decoder.out_features
    source.decoder = nn.Linear(i, o + 1)

    # pre-training target model
    model = cramming.construct_model(cfg.arch, cfg.data.vocab_size)

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

    # opt = torch.optim.Adam(lr=cfg.up.lr, params=model.parameters())

    if cfg.up.warmup > 0:
        warmup = cfg.up.warmup / cfg.up.accumulate
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (warmup / cfg.up.batch_size), 1.0))

    num_tokens = model.encoder.embedding.word_embedding.num_embeddings
    context = cfg.arch.embedding.max_seq_length

    if torch.cuda.is_available():
        model.cuda()
        source.cuda()

    if cfg.up.dp:
        model = torch.nn.DataParallel(model)
        source = torch.nn.DataParallel(model)

    buffer = torch.randint(low=0, high=num_tokens, size=(cfg.up.buffer_size, context), device=d())

    # Launch training
    for i in (bar := trange(cfg.up.num_batches)):

        # Sample a batch on the fly
        with torch.no_grad():

            tic()
            # Re-initialize the parameters of source (i.e. sample a random source)
            up.weights_init(source, init_mult_max=cfg.up.init_mult_max, mask_prob_max=cfg.up.mask_prob_max)

            # Slice a random selection of rows from the buffer (without replacement)
            iz = random.sample(range(buffer.size(0)), cfg.up.sample_batch_size)
            z = buffer[iz, :]

            # Replace some random rows with uniform random characters (reset)
            rows = torch.bernoulli(torch.full(size=(cfg.up.sample_batch_size, 1), fill_value=cfg.up.reset_prob))
            mask = rows.expand(cfg.up.sample_batch_size, context).to(torch.bool)

            uniform = torch.randint(low=0, high=num_tokens, size=(cfg.up.sample_batch_size, context), device=d())
            z[mask] = uniform[mask]

            # Pass it through the source
            # -- In non-sequential mode, pass the input through the model, and then mix the input and output together.
            #    The model itself produces the mask, functioning as a kind of gate on the input. This increase the
            #    probability that the model retains some of the complexity of the input, while also allowing the option
            #    that the input is entirely ignored.

            output = source(z)['outputs'].view(cfg.up.sample_batch_size, context, -1)
            chars, mask = output[:, :, :-1], output[:, :, -1]

            chars = sample(chars, temperature=cfg.up.temperature)
            mask = torch.sigmoid(mask).to(torch.bool)

            z[mask] = chars[mask] # replace the masked part of the input by the output samples
            buffer[iz, :] = z     # replace the inputs in the buffer

            # -- Note that the samples are in full precision. These often require large weights, so mixed precision
            #    leads to nans and infs and whatnot.
            sampletime = toc()

        # Perform a training step on batches sampled from the buffer

        tic()
        # Sample a batch from the buffer
        iz = random.sample(range(cfg.up.buffer_size), cfg.up.batch_size)

        batch = buffer[iz, :]
        if torch.cuda.is_available():
            batch = batch.cuda()

        # We use the MLM loss to train.
        inputs, targets = mask_batch(batch, mask_token=cfg.up.mask_token, mlm_probability=cfg.up.mlm_probability)

        with torch.cuda.amp.autocast():
            output = model(inputs)['outputs'].view(cfg.up.batch_size, context, -1)
            loss = F.cross_entropy(output.transpose(2, 1), targets)

        scaler.scale(loss).backward()

        gn = gradient_norm(model)
        if cfg.up.gc > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.up.gc)


        if i % cfg.up.accumulate == 0:  # perform a step

            scaler.step(opt)
            scaler.update()

            opt.zero_grad()

            if cfg.up.warmup > 0:
                sch.step()

        traintime = toc()

        if cfg.wandb.enabled:
            wandb.log({
                'loss': loss,
                'learning_rate': opt.param_groups[0]['lr'],
                'gradient_norm': gn,
                'sample_time': sampletime,
                'train_time': traintime,
                'pre-training': 1.0
            })
        bar.set_postfix({'loss': f'{loss:.02}'})

        # if cfg.up.print_every > 0 and cfg.up.i % print_every == 0:
        #     print('target')
        #     print_batch(batch[:4, :], ascii_only)
        #
        #     print('model output')
        #
        #     seed = torch.randint(low=0, high=NUM_TOKENS, size=(4, 1), device=d())
        #     output = sample_sequence(model, seed, context, num_tokens=NUM_TOKENS, length=context,
        #                              temperature=temperature)
        #     print_batch(output, ascii_only)

    opt.zero_grad()
    optimizer_to(opt, 'cpu')
    # -- We send the optimizer to the CPU. This avoids (?) issues with the optimizer states on GPU not being cleared,
    #    leading to OOM.

    return model, opt

def main_training_process(cfg, setup):
    """This function controls the central training loop."""
    local_time = time.time()

    if cfg.up.enabled:
        model, opt = pretrain(cfg, setup)
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

    if cfg.up.reuse_opt:
        with torch.no_grad():
            sd = opt.state_dict()

            if cfg.up.opt_mult > 0.0:
                for val in sd['state'].values():
                    val['exp_avg'] *= cfg.up.opt_mult
                    val['exp_avg_sq'] *= cfg.up.opt_mult
                # -- Apply a multiplier to the exp moving average and the second moment. This can be seen as a convex
                #    combination of the fresh optimizer state (which is zero) and the optimizer state inherited from the
                #    universal pretraining.

            model_engine.optimizer.load_state_dict(sd)
            # -- reuse the optimizer from the UP training

    # -- Force some garbage collecting
    # del sd, opt
    # gc.collect()
    # with torch.no_grad():
    #     torch.cuda.empty_cache()

    model_engine.train(cfg.train.pretrain_in_train_mode)
    stats = defaultdict(list)

    # Start the clocks now:
    wallclock_timer = time.time() - elapsed_time
    train_time = time.time()
    training_allowed, no_recovery_necessary = True, True
    loss_vals = []

    # Launch training
    for step, batch in enumerate(dataloader, initial_step + 1):

        # Heavy lifting is moved to engines
        device_batch = model_engine.to_device(batch)
        loss = model_engine.step(device_batch)
        loss_vals.append(loss.detach())

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
