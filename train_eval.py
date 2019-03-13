import gc
from collections import OrderedDict
from dataclasses import dataclass
import logging

import numpy as np
import torch
from pytorch_pretrained_bert import BertAdam
from torch.utils.data import SequentialSampler, DistributedSampler, DataLoader as TorchDataLoader, RandomSampler
from tqdm import tqdm, trange

from bert_erp_datasets import collate_fn, max_example_sequence_length, PreparedDataDataset, PreparedData
from bert_erp_modeling import make_loss_handler, BertMultiHead
from bert_erp_settings import Settings
from result_output import write_predictions


logger = logging.getLogger(__name__)


__all__ = ['evaluate', 'train', 'TaskResult', 'TaskResults']


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def evaluate(settings, model, loss_handlers, device, global_step, eval_results, eval_data_set, return_detailed=False):

    if settings.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data_set)
    else:
        eval_sampler = DistributedSampler(eval_data_set)
    eval_data_loader = TorchDataLoader(
        eval_data_set, sampler=eval_sampler, batch_size=settings.predict_batch_size, collate_fn=collate_fn)

    model.eval()
    all_results = OrderedDict()
    logger.info("Start evaluating")

    if settings.show_step_progress:
        batch_iterator = tqdm(eval_data_loader, desc="Evaluating")
    else:
        batch_iterator = eval_data_loader

    total_loss = 0
    total_count = 0
    for batch in batch_iterator:
        # if len(all_results) % 1000 == 0:
        #     logger.info("Processing example: %d" % (len(all_results)))
        for k in batch:
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            predictions = model(batch)
            loss_result = OrderedDict(
                (h.field,
                 (h.weight, h(batch, predictions, return_detailed=return_detailed, apply_weight=False, as_numpy=True)))
                for h in loss_handlers)
            if return_detailed:
                loss_dict = OrderedDict()
                for k in loss_result:
                    weight, (summary, detailed) = loss_result[k]
                    loss_dict[k] = weight, summary
                    if k not in all_results:
                        all_results[k] = list()
                    all_results[k].extend(detailed)
            else:
                loss_dict = loss_result
            loss = None
            for data_key in loss_dict:
                weight, data_loss = loss_dict[data_key]
                no_valid_inputs = isinstance(data_loss, str) and data_loss == 'no_valid_inputs'
                if no_valid_inputs:
                    current = np.nan
                else:
                    current = data_loss
                if data_key in settings.loss_tasks and not no_valid_inputs:
                    if loss is None:
                        loss = weight * current
                    else:
                        loss += weight * current
                eval_results.add_result(data_key, global_step, current)
            if loss is not None:
                total_loss += loss
                total_count += len(batch['unique_id'])

    if total_count > 0:
        logger.info('eval: {}'.format(total_loss / total_count))
    else:
        logger.info('eval: {}'.format(np.nan))

    if return_detailed:
        return all_results


def _loss_weights(loss_count_dict):
    keys = [k for k in loss_count_dict]
    counts = np.array([loss_count_dict[k] for k in keys])
    loss_weights = 1. / (np.sum(1. / counts) * counts)
    return dict(zip(keys, [w.item() for w in loss_weights]))


def train(settings: Settings, output_validation_path, output_test_path, data: PreparedData, n_gpu, device):

    max_sequence_length = max_example_sequence_length(data)

    train_data_set = PreparedDataDataset(max_sequence_length, data, which='train')
    validation_data_set = PreparedDataDataset(max_sequence_length, data, which='validation')

    num_train_steps = int(
        len(train_data_set) /
        settings.train_batch_size /
        settings.gradient_accumulation_steps * settings.num_train_epochs)

    token_level_prediction_shapes = OrderedDict()
    pooled_prediction_shapes = OrderedDict()

    loss_example_counts = dict()
    loss_handlers = list()

    for k in settings.loss_tasks:
        if k not in train_data_set.fields:
            raise ValueError('loss_task is not present as a field: {}'.format(k))

    for k in train_data_set.fields:
        if k in settings.loss_tasks or k in settings.non_response_outputs or train_data_set.is_response_data(k):
            data_key = train_data_set.data_set_key_for_field(k)
            if k in settings.loss_tasks:
                if data_key is None:
                    loss_example_counts[k] = len(train_data_set)
                else:
                    loss_example_counts[k] = train_data_set.num_examples_for_data_key(data_key)
            critic_type = 'mse'
            critic_kwargs = None
            if k in settings.task_settings:
                critic_type = settings.task_settings[k].critic_type
                critic_kwargs = settings.task_settings[k].critic_kwargs
            else:
                if data_key is not None and data_key in settings.task_settings:
                    critic_type = settings.task_settings[data_key].critic_type
                    critic_kwargs = settings.task_settings[data_key].critic_kwargs
            handler = make_loss_handler(k, critic_type, critic_kwargs)
            loss_handlers.append(handler)
            prediction_shape = handler.shape_adjust(train_data_set.value_shape(k))
            if train_data_set.is_sequence(k):
                token_level_prediction_shapes[k] = prediction_shape
            else:
                pooled_prediction_shapes[k] = prediction_shape

    loss_weights = _loss_weights(loss_example_counts)
    for loss_handler in loss_handlers:
        if loss_handler.field in loss_weights:
            loss_handler.weight = loss_weights[loss_handler.field]

    token_level_input_key_to_shape = OrderedDict()
    pooled_input_key_to_shape = OrderedDict()

    for k in train_data_set.fields:
        if k in settings.additional_input_fields:
            if train_data_set.is_sequence(k):
                token_level_input_key_to_shape[k] = train_data_set.value_shape(k)
            else:
                pooled_input_key_to_shape[k] = train_data_set.value_shape(k)

    # Prepare model
    model = BertMultiHead.from_pretrained(
        settings.bert_model,
        token_prediction_key_to_shape=token_level_prediction_shapes,
        token_level_input_key_to_shape=token_level_input_key_to_shape,
        pooled_prediction_key_to_shape=pooled_prediction_shapes,
        pooled_input_key_to_shape=pooled_input_key_to_shape)

    if settings.fp16:
        model.half()
    model.to(device)
    if settings.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[settings.local_rank], output_device=settings.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if settings.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_())
                           for n, param in model.named_parameters()]
    elif settings.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_())
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=settings.learning_rate,
                         warmup=settings.warmup_proportion,
                         t_total=num_train_steps)

    global_step = 0
    train_results = TaskResults()
    validation_results = TaskResults()
    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(train_data_set))
    # for now we set max_sequence_length so these are never split
    logger.info("  Num split examples = %d", len(train_data_set))
    logger.info("  Batch size = %d", settings.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    if settings.local_rank == -1:
        train_sampler = RandomSampler(train_data_set)
    else:
        train_sampler = DistributedSampler(train_data_set)
    train_data_loader = TorchDataLoader(
        train_data_set, sampler=train_sampler, batch_size=settings.train_batch_size, collate_fn=collate_fn)

    if settings.show_epoch_progress:
        epoch_range = trange(int(settings.num_train_epochs), desc="Epoch")
    else:
        epoch_range = range(int(settings.num_train_epochs))

    for _ in epoch_range:

        model.train()

        if settings.show_step_progress:
            batch_iterator = tqdm(train_data_loader, desc="Iteration")
        else:
            batch_iterator = train_data_loader

        for step, batch in enumerate(batch_iterator):
            if n_gpu == 1:
                for k in batch:
                    batch[k] = batch[k].to(device)
            predictions = model(batch)
            loss_dict = OrderedDict(
                (h.field, (h.weight, h(batch, predictions, apply_weight=False))) for h in loss_handlers)
            batch_size = len(batch['unique_id'])

            # free up memory
            del predictions
            del batch

            loss = None
            for data_key in loss_dict:
                weight, data_loss = loss_dict[data_key]
                no_valid_inputs = isinstance(data_loss, str) and data_loss == 'no_valid_inputs'
                if data_key in settings.loss_tasks and not no_valid_inputs:
                    current = weight * data_loss
                    if loss is None:
                        loss = current
                    else:
                        loss += current
                train_result = np.nan if no_valid_inputs else data_loss.detach().cpu().numpy().item()
                train_results.add_result(
                    data_key,
                    global_step,
                    train_result)

            del loss_dict

            if loss is not None:
                logger.info('train: {}'.format(loss.item() / batch_size))
                if n_gpu > 1:  # hmm - not sure how this is supposed to work
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if settings.fp16 and settings.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * settings.loss_scale
                if settings.gradient_accumulation_steps > 1:
                    loss = loss / settings.gradient_accumulation_steps
                loss.backward()

            if (step + 1) % settings.gradient_accumulation_steps == 0:
                if settings.fp16 or settings.optimize_on_cpu:
                    if settings.fp16 and settings.loss_scale != 1.0:
                        # scale down gradients for fp16 training
                        for param in model.parameters():
                            param.grad.data = param.grad.data / settings.loss_scale
                    is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                    if is_nan:
                        logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                        settings.loss_scale = settings.loss_scale / 2
                        model.zero_grad()
                        continue
                    optimizer.step()
                    copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                else:
                    optimizer.step()
                model.zero_grad()
                global_step += 1

            # we're being super aggressive about releasing memory here because
            # we're right on the edge of fitting in gpu
            del loss
            gc.collect()
            torch.cuda.empty_cache()

        if len(validation_data_set) > 0:
            evaluate(settings, model, loss_handlers, device, global_step, validation_results, validation_data_set)

    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(validation_data_set))
    logger.info("  Num split examples = %d", len(validation_data_set))
    logger.info("  Batch size = %d", settings.predict_batch_size)

    if len(validation_data_set) > 0:
        all_validation = evaluate(
            settings, model, loss_handlers, device, global_step, validation_results, validation_data_set,
            return_detailed=True)
    else:
        all_validation = {}

    test_data_set = PreparedDataDataset(max_sequence_length, data, which='test')

    test_results = TaskResults()
    if len(test_data_set) > 0:
        all_test = evaluate(
            settings, model, loss_handlers, device, global_step, test_results, test_data_set, return_detailed=True)
    else:
        all_test = {}

    write_predictions(output_validation_path, all_validation, validation_data_set, settings)
    write_predictions(output_test_path, all_test, test_data_set, settings)

    # clean up after we're done to try to release CUDA resources to other people when there are no more tasks
    gc.collect()
    torch.cuda.empty_cache()


@dataclass
class TaskResult:
    step: int
    value: float


class TaskResults:

    def __init__(self):
        self.results = OrderedDict()

    def add_result(self, name, step, value):
        if name not in self.results:
            self.results[name] = list()
        self.results[name].append(TaskResult(step, value))
