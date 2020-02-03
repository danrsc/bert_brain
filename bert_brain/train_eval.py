import os
import gc
from collections import OrderedDict
from dataclasses import dataclass
from copy import deepcopy
from typing import Optional
import logging

import numpy as np
import torch
import torch.nn
from torch.utils.data import SequentialSampler, DataLoader as TorchDataLoader
from torch.optim.lr_scheduler import LambdaLR

from .data_sets import collate_fn, DataIdMultiDataset, BatchOneTaskSequentialSampler
from .modeling import KeyedLinear, BertMultiPredictionHead
from .settings import Settings
from .result_output import write_predictions, write_loss_curve


logger = logging.getLogger(__name__)


__all__ = ['evaluate', 'train', 'TaskResult', 'TaskResults', 'setup_prediction_heads_and_losses']


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
                param_opti.grad = torch.nn.Parameter(
                    param_opti.data.new().resize_(*param_opti.data.size()), requires_grad=True)
            # noinspection PyUnresolvedReferences
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def restore_model_parameters_and_set_meta_gradient(
        model, model_state_to_restore, model_state_for_gradient, num_inner_steps):
    model.zero_grad()
    target_state = deepcopy(model.state_dict())
    model.load_state_dict(model_state_to_restore)
    model.zero_grad()
    for key, p in model.named_parameters():
        if key not in target_state or key not in model_state_for_gradient:
            raise ValueError('Inconsistent state dictionaries')
        if p.requires_grad:
            if p.grad is None:
                p.grad = torch.nn.Parameter(p.data.new().resize_(*p.data.size()), requires_grad=True)
            # noinspection PyUnresolvedReferences
            p.grad.data.copy_((model_state_for_gradient[key].detach() - target_state[key].detach()) / num_inner_steps)


def evaluate(
        settings: Settings,
        model,
        loss_handlers,
        train_samplers,
        device,
        epoch,
        global_step,
        eval_results,
        eval_data_set,
        return_detailed=False):

    eval_sampler = None
    batch_sampler = None
    if settings.use_sequential_sampling_on_evaluate or return_detailed:
        if settings.sampler_factory.is_one_task_at_a_time_sampler():
            batch_sampler = BatchOneTaskSequentialSampler(
                eval_data_set, settings.optimization_settings.predict_batch_size)
        else:
            eval_sampler = SequentialSampler(eval_data_set)
    else:
        sampler_ = settings.sampler_factory.make_sampler(
            eval_data_set, settings.optimization_settings.predict_batch_size)
        if hasattr(sampler_, 'update_from'):
            sampler_.update_from(train_samplers)
        if settings.sampler_factory.is_batch_sampler():
            batch_sampler = sampler_
        else:
            eval_sampler = sampler_

    if batch_sampler is None:
        eval_data_loader = TorchDataLoader(
            eval_data_set,
            sampler=eval_sampler,
            batch_size=settings.optimization_settings.predict_batch_size,
            collate_fn=collate_fn,
            num_workers=settings.optimization_settings.num_loader_workers)
    else:
        eval_data_loader = TorchDataLoader(
            eval_data_set, batch_sampler=batch_sampler, collate_fn=collate_fn,
            num_workers=settings.optimization_settings.num_loader_workers)

    model.eval()
    all_results = OrderedDict()
    logger.info("Start evaluating")

    total_loss = 0
    total_count = 0
    losses_to_write = OrderedDict()
    losses_to_write_counts = OrderedDict()
    for batch in eval_data_loader:
        # if len(all_results) % 1000 == 0:
        #     logger.info("Processing example: %d" % (len(all_results)))
        for k in batch:
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            predictions = model(batch, eval_data_set)
            eval_data_set.just_in_time_targets(batch, predictions)
            loss_result = OrderedDict(
                (h.field,
                 (h.weight,
                  h(True, epoch, global_step, batch, predictions,
                    return_detailed=return_detailed, apply_weight=False, as_numpy=True, reduction='none')))
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
            for data_key in loss_dict:
                weight, (data_loss, data_valid_count) = loss_dict[data_key]
                if data_key not in losses_to_write:
                    losses_to_write[data_key] = 0
                    losses_to_write_counts[data_key] = 0
                if data_valid_count == 0:
                    current = np.nan
                else:
                    current = np.sum(data_loss)
                    losses_to_write[data_key] += current
                    losses_to_write_counts[data_key] += data_valid_count

                if data_valid_count > 0:
                    kind = eval_data_set.response_data_kind(data_key)
                    if data_key in settings.all_loss_tasks or kind in settings.all_loss_tasks:
                        total_loss += current
                        total_count += data_valid_count

    for h in loss_handlers:
        if hasattr(h, 'after_eval_batches'):
            h.after_eval_batches(epoch, global_step)

    for k in losses_to_write:
        if losses_to_write_counts[k] == 0:
            losses_to_write[k] = np.nan
        else:
            losses_to_write[k] /= losses_to_write_counts[k]
        eval_results.add_result(k, epoch, global_step, losses_to_write[k])

    if total_count > 0:
        if len(losses_to_write) < 4:
            logger.info('eval:  {:<#8.6}, '.format(total_loss / total_count) + ', '.join(
                ['{}: {:<#8.6}'.format(k, losses_to_write[k]) for k in losses_to_write]))
        else:
            logger.info('eval:  {}'.format(total_loss / total_count))
    else:
        if len(losses_to_write) < 4:
            logger.info('eval:  {:<#8.6}, '.format(total_loss / total_count) + ', '.join(
                ['{}: {:<#8.6}'.format(k, losses_to_write[k]) for k in losses_to_write]))
        else:
            logger.info('eval:  {}'.format(np.nan))

    if return_detailed:
        return all_results


def _loss_weights(loss_count_dict):
    keys = [k for k in loss_count_dict]
    counts = np.array([loss_count_dict[k] for k in keys])
    loss_weights = 1. / (np.sum(1. / counts) * counts)
    return dict(zip(keys, [w.item() for w in loss_weights]))


def _raise_if_head_settings_inconsistent(head_a, head_b):
    if head_a.head_type != head_b.head_type:
        raise ValueError('Inconsistent types in prediction head settings with same key: {}'.format(head_b.key))
    if len(head_a.kwargs) != len(head_b.kwargs):
        raise ValueError('Inconsistent kwargs in prediction head settings with same key: {}'.format(head_b.key))
    for kwarg in head_a.kwargs:
        if kwarg not in head_b.kwargs \
                or head_a.kwargs[kwarg] != head_b.kwargs[kwarg]:
            raise ValueError('Inconsistent kwargs in prediction head settings with same key: {}'.format(head_b.key))


def setup_prediction_heads_and_losses(settings: Settings, data_set):

    loss_example_counts = dict()
    loss_handlers = list()

    all_kinds = set([data_set.response_data_kind(k) for k in data_set.fields
                     if data_set.response_data_kind(k) is not None])

    for k in settings.all_loss_tasks:
        if k not in all_kinds and k not in data_set.fields:
            raise ValueError('loss_task is not present as a field: {}'.format(k))

    placeholder_name_to_fields = dict()
    prediction_shapes = dict()
    for k in data_set.fields:
        kind = data_set.response_data_kind(k) if data_set.is_response_data(k) else None
        corpus_key = data_set.data_set_key_for_field(k)
        if k in settings.all_loss_tasks \
                or kind in settings.all_loss_tasks \
                or k in settings.non_response_outputs \
                or data_set.is_response_data(k):
            if k in settings.all_loss_tasks or kind in settings.all_loss_tasks:
                loss_example_counts[k] = data_set.num_examples_for_field(k)
            handler = settings.get_critic(k, data_set)
            loss_handlers.append(handler)

            prediction_shape = handler.shape_adjust(data_set.value_shape(k))
            prediction_shapes[k] = prediction_shape

            if kind is not None:
                if kind not in placeholder_name_to_fields:
                    placeholder_name_to_fields[kind] = [k]
                else:
                    placeholder_name_to_fields[kind].append(k)
            if corpus_key is not None:
                if corpus_key not in placeholder_name_to_fields:
                    placeholder_name_to_fields[corpus_key] = [k]
                else:
                    placeholder_name_to_fields[corpus_key].append(k)

    if settings.weight_losses_by_inverse_example_counts:
        loss_weights = _loss_weights(loss_example_counts)
        for loss_handler in loss_handlers:
            if loss_handler.field in loss_weights:
                loss_handler.weight = loss_weights[loss_handler.field]

    graph_parts = OrderedDict()
    if settings.common_graph_parts is not None:
        for k in settings.common_graph_parts:
            settings.common_graph_parts[k].resolve_placeholders(
                placeholder_name_to_fields, prediction_shapes, len(data_set.response_fields))
            graph_parts[k] = settings.common_graph_parts[k]

    default_sequence_head = None
    default_pooled_head = None
    for k in prediction_shapes:
        kind = data_set.response_data_kind(k) if data_set.is_response_data(k) else None
        corpus_key = data_set.data_set_key_for_field(k)
        prediction_head_parts = None
        if k in settings.head_graph_parts:
            prediction_head_parts = settings.head_graph_parts[k]
        elif kind in settings.head_graph_parts:
            prediction_head_parts = settings.head_graph_parts[kind]
        elif corpus_key in settings.head_graph_parts:
            prediction_head_parts = settings.head_graph_parts[corpus_key]

        if prediction_head_parts is None:
            if data_set.is_sequence(k):
                if default_sequence_head is None:
                    default_sequence_head = OrderedDict(default_sequence_linear=KeyedLinear(
                        settings.default_sequence_source, True))
                prediction_head_parts = default_sequence_head
                prediction_head_parts['default_sequence_linear'].output_key_to_shape[k] = prediction_shapes[k]
                prediction_head_parts['default_sequence_linear'].apply_at_most_one_data_id[k] = \
                    'if_no_target' if data_set.is_just_in_time_field(k) else False
            else:
                if default_pooled_head is None:
                    default_pooled_head = OrderedDict(
                        default_pooled_linear=KeyedLinear(
                            settings.default_pooled_source, False, apply_at_most_one_data_id=dict()))
                prediction_head_parts = default_pooled_head
                prediction_head_parts['default_pooled_linear'].output_key_to_shape[k] = prediction_shapes[k]
                prediction_head_parts['default_pooled_linear'].apply_at_most_one_data_id[k] = \
                    'if_no_target' if data_set.is_just_in_time_field(k) else False

        for key in prediction_head_parts:
            if key not in graph_parts:
                graph_parts[key] = prediction_head_parts[key]
                graph_parts[key].resolve_placeholders(
                    placeholder_name_to_fields, prediction_shapes, len(data_set.response_fields))
            else:
                if id(graph_parts[key]) != id(prediction_head_parts[key]):
                    raise ValueError('Duplicate graph_part name: {}'.format(key))

    token_supplemental_key_to_shape = OrderedDict()
    pooled_supplemental_key_to_shape = OrderedDict()

    for k in data_set.fields:
        if k in settings.supplemental_fields:
            if data_set.is_sequence(k):
                token_supplemental_key_to_shape[k] = data_set.value_shape(k)
            else:
                pooled_supplemental_key_to_shape[k] = data_set.value_shape(k)

    return graph_parts, token_supplemental_key_to_shape, pooled_supplemental_key_to_shape, loss_handlers


@dataclass
class MetaLearnScheduler:
    outer_scheduler: LambdaLR
    inner_scheduler: LambdaLR

    def step(self, epoch: Optional[int] = None):
        self.outer_scheduler.step(epoch)
        self.inner_scheduler.step(epoch)


def _train_step(
        settings: Settings,
        loss_tasks: set,
        train_data_set: DataIdMultiDataset,
        device,
        model,
        batch,
        step,
        index_epoch,
        global_step,
        loss_handlers,
        train_results,
        param_optimizer,
        optimizer,
        scheduler,
        sampler_to_update,
        log_tag='train'):

    for k in batch:
        batch[k] = batch[k].to(device)
    predictions = model(batch, train_data_set)
    if sampler_to_update is not None and hasattr(sampler_to_update, 'update'):
        sampler_to_update.update(batch, predictions, loss_handlers)
    train_data_set.just_in_time_targets(batch, predictions)
    loss_dict = OrderedDict(
        (h.field,
         (h.weight,
          h(False, index_epoch, global_step, batch, predictions, apply_weight=False))) for h in loss_handlers)

    loss = None
    losses_to_write = OrderedDict()
    for data_key in loss_dict:
        weight, data_loss = loss_dict[data_key]
        no_valid_inputs = isinstance(data_loss, str) and data_loss == 'no_valid_inputs'
        kind = train_data_set.response_data_kind(data_key)
        if (data_key in loss_tasks or kind in loss_tasks) and not no_valid_inputs:
            current = weight * data_loss
            losses_to_write[data_key] = np.nan if no_valid_inputs else data_loss.detach().cpu().numpy().item()
            if loss is None:
                loss = current
            else:
                loss += current
        train_result = np.nan if no_valid_inputs else data_loss.detach().cpu().numpy().item()
        train_results.add_result(
            data_key,
            index_epoch,
            global_step,
            train_result)

    if loss is not None:
        if len(losses_to_write) < 4:
            # noinspection PyUnresolvedReferences
            logger.info('{}: {:<#8.6}, '.format(log_tag, loss.item()) + ', '.join(
                ['{}: {:<#8.6}'.format(k, losses_to_write[k]) for k in losses_to_write]))
        else:
            # noinspection PyUnresolvedReferences
            logger.info('{}: {}'.format(log_tag, loss.item()))
        if settings.optimization_settings.fp16 and settings.loss_scale != 1.0:
            # rescale loss for fp16 training
            # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
            loss = loss * settings.loss_scale
        if settings.optimization_settings.gradient_accumulation_steps > 1:
            loss = loss / settings.optimization_settings.gradient_accumulation_steps
        loss.backward()

    if (step + 1) % settings.optimization_settings.gradient_accumulation_steps == 0:
        if settings.optimization_settings.fp16 or settings.optimization_settings.optimize_on_cpu:
            if settings.optimization_settings.fp16 and settings.loss_scale != 1.0:
                # scale down gradients for fp16 training
                for param in model.parameters():
                    param.grad.data = param.grad.data / settings.loss_scale
            is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
            if is_nan:
                logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                settings.loss_scale = settings.loss_scale / 2
                model.zero_grad()
                return False
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
        else:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        model.zero_grad()

        for group in optimizer.param_groups:
            assert(group['lr'] >= 0)

    return True


def iterate_and_update(update_fn, iterable):
    for item in iterable:
        yield item
        update_fn()


def train(
        settings: Settings,
        output_dir: str,
        completion_file_path: str,
        output_model_path: str,
        train_data_set: DataIdMultiDataset,
        validation_data_set: DataIdMultiDataset,
        test_data_set: DataIdMultiDataset,
        device: torch.device,
        progress_update,
        load_from_path: str = None):

    if len(test_data_set) == 0:
        test_data_set = None

    output_train_curve_path = os.path.join(output_dir, 'train_curve.npz')
    output_validation_curve_path = os.path.join(output_dir, 'validation_curve.npz')

    is_meta_learn_active = (
        (settings.meta_learn_no_gradient_loss_tasks is not None and len(settings.meta_learn_no_gradient_loss_tasks) > 0)
        or (settings.meta_learn_gradient_loss_tasks is not None and len(settings.meta_learn_gradient_loss_tasks) > 0))

    if is_meta_learn_active:
        if settings.loss_tasks is not None and len(settings.loss_tasks) > 0:
            raise ValueError('When meta learning is active, settings.loss_tasks must be empty')
        if settings.meta_learn_gradient_loss_tasks is None or len(settings.meta_learn_gradient_loss_tasks) == 0:
            raise ValueError(
                'If settings.meta_learn_no_gradient_loss_tasks is non-empty, '
                'then settings.meta_learn_gradient_loss_tasks must be non-empty')
        if settings.num_meta_learn_gradient_samples < 1:
            raise ValueError('settings.num_meta_learn_gradient_samples must be >= 1 when meta learning is active')
        if settings.num_meta_learn_no_gradient_samples < 0:
            raise ValueError('settings.num_meta_learn_no_gradient_samples must be >= 0 when meta learning is active')
        if not settings.sampler_factory.is_one_task_at_a_time_sampler():
            raise ValueError('settings.sampler_factory must be a one-task-at-a-time variety for meta learning. '
                             'Current factory: {}'.format(settings.sampler_factory))

    train_sampler = None
    batch_sampler = None
    no_gradient_meta_learn_sampler = None
    gradient_meta_learn_sampler = None
    train_samplers = list()
    if is_meta_learn_active:
        if not settings.sampler_factory.is_batch_sampler() \
                or not settings.sampler_factory.is_one_task_at_a_time_sampler():
            raise ValueError('Unsupported sampler_factory for meta learning: {}'.format(
                type(settings.sampler_factory)))
        gradient_meta_learn_sampler = settings.sampler_factory.make_sampler(
            train_data_set,
            settings.optimization_settings.train_batch_size,
            settings.meta_learn_gradient_loss_tasks)
        train_samplers.append(gradient_meta_learn_sampler)
        if (settings.meta_learn_no_gradient_loss_tasks is not None
                and len(settings.meta_learn_no_gradient_loss_tasks) > 0):
            no_gradient_meta_learn_sampler = settings.sampler_factory.make_sampler(
                train_data_set,
                settings.optimization_settings.train_batch_size,
                settings.meta_learn_no_gradient_loss_tasks)
            train_samplers.append(no_gradient_meta_learn_sampler)
    else:
        sampler_ = settings.sampler_factory.make_sampler(
            train_data_set, settings.optimization_settings.train_batch_size)
        train_samplers.append(sampler_)
        if settings.sampler_factory.is_batch_sampler():
            batch_sampler = sampler_
        else:
            train_sampler = sampler_

    if is_meta_learn_active:
        no_gradient_meta_learn_loader = None if no_gradient_meta_learn_sampler is None else TorchDataLoader(
            train_data_set, batch_sampler=no_gradient_meta_learn_sampler, collate_fn=collate_fn,
            num_workers=settings.optimization_settings.num_loader_workers)
        gradient_meta_learn_loader = TorchDataLoader(
            train_data_set, batch_sampler=gradient_meta_learn_sampler, collate_fn=collate_fn,
            num_workers=settings.optimization_settings.num_loader_workers)
        len_train = int(np.round(gradient_meta_learn_sampler.true_div_len() * gradient_meta_learn_sampler.batch_size))
        train_data_loader = None
    else:
        no_gradient_meta_learn_loader = None
        gradient_meta_learn_loader = None
        if batch_sampler is None:
            train_data_loader = TorchDataLoader(
                train_data_set,
                sampler=train_sampler,
                batch_size=settings.optimization_settings.train_batch_size,
                collate_fn=collate_fn,
                num_workers=settings.optimization_settings.num_loader_workers)
        else:
            train_data_loader = TorchDataLoader(
                train_data_set, batch_sampler=batch_sampler, collate_fn=collate_fn,
                num_workers=settings.optimization_settings.num_loader_workers)
        len_train = train_data_set.length(task_filter=settings.meta_learn_gradient_loss_tasks) \
            if is_meta_learn_active else len(train_data_set)

    num_train_steps_per_epoch = int(np.ceil(
        len_train /
        settings.optimization_settings.train_batch_size /
        settings.optimization_settings.gradient_accumulation_steps))

    num_total_batches = num_train_steps_per_epoch * settings.optimization_settings.num_train_epochs
    progress_update.update_batch_total(num_total_batches)

    num_train_steps_prediction_heads = num_train_steps_per_epoch * settings.optimization_settings.num_train_epochs
    num_train_steps_other = num_train_steps_per_epoch * (
        settings.optimization_settings.num_train_epochs
        - settings.optimization_settings.num_epochs_train_prediction_heads_only
        - settings.optimization_settings.num_final_epochs_train_prediction_heads_only)

    num_epochs_prediction_head_only_train = settings.optimization_settings.num_epochs_train_prediction_heads_only
    if num_epochs_prediction_head_only_train < 0:
        num_epochs_prediction_head_only_train = settings.optimization_settings.num_train_epochs
    start_final_epochs_prediction_head_only_train = int(
        settings.optimization_settings.num_train_epochs
        - settings.optimization_settings.num_final_epochs_train_prediction_heads_only)

    graph_parts, token_supplemental_key_to_shape, pooled_supplemental_key_to_shape, loss_handlers = \
        setup_prediction_heads_and_losses(settings, train_data_set)

    # Prepare model
    model = BertMultiPredictionHead.from_pretrained(
        load_from_path if load_from_path is not None else settings.bert_model,
        head_graph_parts=graph_parts,
        token_supplemental_key_to_shape=token_supplemental_key_to_shape,
        pooled_supplemental_key_to_shape=pooled_supplemental_key_to_shape)

    if settings.optimization_settings.fp16:
        model.half()
    model.to(device)

    # Prepare optimizer
    if settings.optimization_settings.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_())
                           for n, param in model.named_parameters()]
    elif settings.optimization_settings.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_())
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())

    def is_no_decay(param_name):
        name_parts = param_name.split('.')
        if name_parts[-1] in ['bias', 'gamma', 'beta']:
            return True
        if len(name_parts) > 1 and name_parts[-2] == 'LayerNorm' and name_parts[-1] == 'weight':
            return True
        return False

    non_prediction_head_parameters = None
    if num_epochs_prediction_head_only_train > 0 or start_final_epochs_prediction_head_only_train:
        non_prediction_head_parameters = [p for n, p in param_optimizer if not n.startswith('prediction_head.')]
        for p in non_prediction_head_parameters:
            p.requires_grad = False

    prediction_head_weight_decay_group = {
        'params': [p for n, p in param_optimizer if not is_no_decay(n) and n.startswith('prediction_head.')],
        'weight_decay': 0.01,
        't_total': num_train_steps_prediction_heads}
    prediction_head_no_decay_group = {
        'params': [p for n, p in param_optimizer if is_no_decay(n) and n.startswith('prediction_head.')],
        'weight_decay': 0.0,
        't_total': num_train_steps_prediction_heads}

    if settings.optimization_settings.learning_rate_head is not None:
        prediction_head_weight_decay_group['lr'] = settings.optimization_settings.learning_rate_head
        prediction_head_no_decay_group['lr'] = settings.optimization_settings.learning_rate_head

    # noinspection PyUnresolvedReferences
    optimizer_grouped_parameters = [
        # weight decay, prediction head
        prediction_head_weight_decay_group,
        # no weight decay, prediction head
        prediction_head_no_decay_group,
        # weight decay, non-prediction head
        {'params': [p for n, p in param_optimizer if not is_no_decay(n) and not n.startswith('prediction_head.')],
         'weight_decay': 0.01,
         't_total': num_train_steps_other},
        # no weight decay, prediction head
        {'params': [p for n, p in param_optimizer if is_no_decay(n) and not n.startswith('prediction_head.')],
         'weight_decay': 0.0,
         't_total': num_train_steps_other}]

    inner_optimizer_grouped_parameters = None
    if is_meta_learn_active:
        inner_optimizer_grouped_parameters = list(dict(g) for g in optimizer_grouped_parameters)
        for g in inner_optimizer_grouped_parameters:
            g['t_total'] *= settings.num_meta_learn_no_gradient_samples + settings.num_meta_learn_gradient_samples

    optimizer = settings.optimization_settings.make_optimizer(
        optimizer_grouped_parameters,
        lr=settings.optimization_settings.learning_rate)
    scheduler = settings.optimization_settings.learning_rate_schedule.get_schedule(
        optimizer, optimizer_grouped_parameters)

    inner_meta_learn_optimizer = None
    if is_meta_learn_active:
        inner_meta_learn_optimizer = settings.optimization_settings.make_inner_meta_learn_optimizer(
            optimizer,
            inner_optimizer_grouped_parameters,
            lr=settings.optimization_settings.learning_rate)
        inner_meta_learn_scheduler = LambdaLR(inner_meta_learn_optimizer, scheduler.lr_lambdas, last_epoch=-1)
        scheduler = MetaLearnScheduler(scheduler, inner_meta_learn_scheduler)

    global_step = 0
    train_results = TaskResults()
    validation_results = TaskResults()
    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(train_data_set))
    # for now we set max_sequence_length so these are never split
    logger.info("  Num split examples = %d", len(train_data_set))
    logger.info("  Batch size = %d", settings.optimization_settings.train_batch_size)
    logger.info("  Num steps (heads) = %d", num_train_steps_prediction_heads)
    logger.info("  Num steps (all param) = %d", num_train_steps_other)

    for index_epoch in iterate_and_update(
            progress_update.update_epoch, range(int(settings.optimization_settings.num_train_epochs))):

        logger.info('Starting epoch {}'.format(index_epoch))

        model.train()

        if index_epoch == start_final_epochs_prediction_head_only_train:
            for p in non_prediction_head_parameters:
                p.requires_grad = False
        elif index_epoch == num_epochs_prediction_head_only_train:
            for p in non_prediction_head_parameters:
                p.requires_grad = True

        if is_meta_learn_active:
            no_gradient_meta_learn_iterator = iter(no_gradient_meta_learn_loader) \
                if no_gradient_meta_learn_loader is not None else None
            gradient_meta_learn_iterator = iter(gradient_meta_learn_loader)

            for step in iterate_and_update(
                    progress_update.update_batch,
                    range(num_train_steps_prediction_heads // settings.optimization_settings.num_train_epochs)):
                restore_state = deepcopy(model.state_dict())
                for key, p in model.named_parameters():
                    restore_state[key].requires_grad = p.requires_grad
                gradient_state = None
                for index_inner_step in range(
                        settings.num_meta_learn_no_gradient_samples + settings.num_meta_learn_gradient_samples):
                    if index_inner_step < settings.num_meta_learn_no_gradient_samples:
                        try:
                            inner_batch = next(no_gradient_meta_learn_iterator)
                        except StopIteration:
                            no_gradient_meta_learn_iterator = iter(no_gradient_meta_learn_loader)
                            inner_batch = next(no_gradient_meta_learn_iterator)
                        current_loss_tasks = settings.meta_learn_no_gradient_loss_tasks
                        sampler_to_update = no_gradient_meta_learn_sampler
                        log_tag = 'inner_no_grad'
                    else:
                        if index_inner_step == settings.num_meta_learn_no_gradient_samples:
                            if index_inner_step == 0:
                                # special case, when there are no no_gradient steps use the restore_state
                                # to avoid extra memory
                                gradient_state = restore_state
                            else:
                                gradient_state = deepcopy(model.state_dict())
                                for key, p in model.named_parameters():
                                    gradient_state[key].requires_grad = p.requires_grad
                        try:
                            inner_batch = next(gradient_meta_learn_iterator)
                        except StopIteration:
                            gradient_meta_learn_iterator = iter(gradient_meta_learn_loader)
                            inner_batch = next(gradient_meta_learn_iterator)
                        current_loss_tasks = settings.meta_learn_gradient_loss_tasks
                        sampler_to_update = gradient_meta_learn_sampler
                        log_tag = 'inner_grad'
                    _train_step(
                        settings,
                        current_loss_tasks,
                        train_data_set, device, model, inner_batch, step, index_epoch, global_step,
                        loss_handlers, train_results, param_optimizer,
                        inner_meta_learn_optimizer, None, sampler_to_update, log_tag=log_tag)
                logger.info('meta epoch {}, step {}'.format(index_epoch, step))
                restore_model_parameters_and_set_meta_gradient(
                    model, restore_state, gradient_state, settings.num_meta_learn_gradient_samples)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
        else:
            sampler_to_update = batch_sampler if batch_sampler is not None else train_sampler

            for step, batch in iterate_and_update(progress_update.update_batch, enumerate(train_data_loader)):
                if _train_step(
                        settings,
                        settings.loss_tasks,
                        train_data_set, device, model, batch, step, index_epoch, global_step,
                        loss_handlers, train_results, param_optimizer, optimizer, scheduler, sampler_to_update):
                    global_step += 1

        write_loss_curve(output_train_curve_path, train_results)
        if len(validation_data_set) > 0:
            evaluate(settings, model, loss_handlers, train_samplers, device, index_epoch,
                     global_step, validation_results, validation_data_set)
            write_loss_curve(output_validation_curve_path, validation_results)

    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(validation_data_set))
    logger.info("  Num split examples = %d", len(validation_data_set))
    logger.info("  Batch size = %d", settings.optimization_settings.predict_batch_size)

    if len(validation_data_set) > 0:
        all_validation = evaluate(
            settings, model, loss_handlers, train_samplers, device, settings.optimization_settings.num_train_epochs - 1,
            global_step, TaskResults(), validation_data_set, return_detailed=True)
    else:
        all_validation = {}

    if test_data_set is not None and len(test_data_set) > 0:
        all_test = evaluate(
            settings, model, loss_handlers, train_samplers, device, settings.optimization_settings.num_train_epochs - 1,
            global_step, TaskResults(), test_data_set, return_detailed=True)
    else:
        all_test = {}

    write_predictions(os.path.join(output_dir, 'validation_predictions'), all_validation, validation_data_set, settings)
    write_predictions(os.path.join(output_dir, 'test_predictions'), all_test, test_data_set, settings)

    # Save a trained model and the associated configuration
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    model.save_pretrained(output_model_path)

    with open(completion_file_path, 'wt') as completion_file:
        completion_file.write('We did it!\n')
        completion_file.write('batches:\t{}\n'.format(num_total_batches))

    # clean up after we're done to try to release CUDA resources to other people when there are no more tasks
    gc.collect()
    torch.cuda.empty_cache()


@dataclass
class TaskResult:
    epoch: int
    step: int
    value: float


class TaskResults:

    def __init__(self):
        self.results = OrderedDict()

    def add_result(self, name, epoch, step, value):
        if name not in self.results:
            self.results[name] = list()
        self.results[name].append(TaskResult(epoch, step, value))
