import os
from collections import OrderedDict
import logging
import pickle
import inspect
import tarfile
import tempfile
import shutil
import itertools

import numpy as np
import torch
from torch import nn
from pytorch_pretrained_bert.modeling import BertConfig, \
    BertPreTrainedModel, BertModel, CONFIG_NAME, WEIGHTS_NAME, PRETRAINED_MODEL_ARCHIVE_MAP, TF_WEIGHTS_NAME, \
    cached_path, load_tf_weights_in_bert, gelu, BertLayerNorm

from bert_erp_modeling.utility_modules import GroupPool, at_most_one_data_id

logger = logging.getLogger(__name__)


__all__ = ['BertMultiPredictionHead', 'KeyedLinear', 'KeyedGroupPooledLinear', 'MultiPredictionHead',
           'BertOutputSupplement', 'KeyedCombinedLinear']


class KeyedBase(torch.nn.Module):

    def __init__(self, prediction_key_to_shape):
        super().__init__()
        self.prediction_key_to_shape = OrderedDict(prediction_key_to_shape)
        self.splits = [int(np.prod(self.prediction_key_to_shape[k])) for k in self.prediction_key_to_shape]

    def forward(self, sequence_output, pooled_output, batch):
        raise NotImplementedError('{} does not implement forward'.format(type(self)))

    def update_state_dict(self, prefix, state_dict, old_prediction_key_to_shape):
        old_splits = np.cumsum([int(np.prod(old_prediction_key_to_shape[k])) for k in old_prediction_key_to_shape])
        old_splits = dict((k, (0 if i == 0 else old_splits[i] - 1, old_splits[i]))
                          for i, k in enumerate(old_prediction_key_to_shape))
        ranges = [old_splits[k] if k in old_splits else None for k in self.prediction_key_to_shape]
        for idx, k in enumerate(self.prediction_key_to_shape):
            if ranges[idx] is not None and int(np.prod(self.prediction_key_to_shape[k])) != ranges[idx]:
                raise ValueError('Inconsistent number of targets for prediction key: {}'.format(k))
        current_splits = [0] + np.cumsum(self.splits).tolist()
        total = current_splits[-1]
        current_splits = current_splits[:-1]

        def update(module, prefix_=''):
            for name, tensor in itertools.chain(
                    module.named_buffers(prefix_[:-1], False), module.named_parameters(prefix_[:-1], False)):
                if name in state_dict:
                    state = state_dict[name]
                    updated_state = tensor.clone()
                    for idx in range(len(current_splits)):
                        if ranges[idx] is not None:
                            end = current_splits[idx + 1] if idx + 1 < len(current_splits) else total
                            if len(state.size()) < 3:
                                updated_state[idx:end] = state[ranges[idx][0]:ranges[idx][1]]
                            else:
                                raise ValueError('Unexpected state size: {}'.format(len(state.size())))
                    state_dict[name] = updated_state

            for name, child in module.named_children():
                if child is not None:
                    update(child, prefix_ + name + '.')

        update(self, prefix)


class _HiddenLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, activation_function=gelu):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.activation_function = activation_function
        self.layer_norm = BertLayerNorm(out_channels, eps=1e-12)

    def forward(self, x):
        x = self.linear(x)
        if self.activation_function is not None:
            x = self.activation_function(x)
        return self.layer_norm(x)


class KeyedLinear(KeyedBase):

    def __init__(
            self,
            is_sequence,
            in_sequence_channels,
            in_pooled_channels,
            prediction_key_to_shape,
            hidden_sizes=None,
            hidden_activation=gelu):
        super().__init__(prediction_key_to_shape)
        self.is_sequence = is_sequence
        self.hidden = None
        in_channels = in_sequence_channels if self.is_sequence else in_pooled_channels
        if hidden_sizes is not None:
            if np.isscalar(hidden_sizes):
                hidden_sizes = [hidden_sizes]
            hidden_modules = list()
            for index_hidden in range(len(hidden_sizes)):
                current_in = in_channels if index_hidden == 0 else hidden_sizes[index_hidden - 1]
                hidden_modules.append(_HiddenLayer(current_in, hidden_sizes[index_hidden], hidden_activation))
            self.hidden = torch.nn.Sequential(*hidden_modules)
        if hidden_sizes is not None:
            in_channels = hidden_sizes[-1]
        self.linear = nn.Linear(in_channels, sum(self.splits))

    def forward(self, sequence_output, pooled_output, batch):
        x = sequence_output if self.is_sequence else pooled_output
        if self.hidden is not None:
            x = self.hidden(x)
        predictions = self.linear(x)
        predictions = torch.split(predictions, self.splits, dim=-1)
        result = OrderedDict()
        assert(len(self.prediction_key_to_shape) == len(predictions))
        for k, p in zip(self.prediction_key_to_shape, predictions):
            if self.is_sequence:
                p = p.view(p.size()[:2] + self.prediction_key_to_shape[k])
            else:
                p = p.view(p.size()[:1] + self.prediction_key_to_shape[k])
            result[k] = p
            if not self.is_sequence and k not in batch:
                # we are in data_ids mode, there must be at most one valid data_id per example
                data_ids = at_most_one_data_id(batch[(k, 'data_ids')])
                indicator_valid = data_ids >= 0
                result[k] = result[k][indicator_valid]
                result[(k, 'data_ids')] = data_ids[indicator_valid]
                result[(k, 'example_ids')] = torch.arange(len(data_ids), device=data_ids.device)[indicator_valid]

        return result


class KeyedGroupPooledLinear(torch.nn.Module):

    def __init__(self, in_sequence_channels, in_pooled_channels, prediction_key_to_shape):
        super().__init__()
        self.group_pool = GroupPool()
        self.linear = KeyedLinear(
            is_sequence=False,
            in_sequence_channels=in_sequence_channels,
            in_pooled_channels=in_pooled_channels,
            prediction_key_to_shape=prediction_key_to_shape)

    def forward(self, sequence_output, pooled_output, batch):
        all_data_ids = [batch[(k, 'data_ids')] for k in self.prediction_key_to_shape]
        for idx in range(1, len(all_data_ids)):
            if not torch.equal(all_data_ids[0], all_data_ids[idx]):
                raise ValueError('Inconsistent data_ids cannot be used within the same instance of FMRIHead')
        data_ids = all_data_ids[0]
        pooled, groups, example_ids = self.group_pool(sequence_output, data_ids)
        result = self.linear(None, pooled, batch)
        keys = [k for k in result]
        for k in keys:
            result[(k, 'data_ids')] = groups
            result[(k, 'example_ids')] = example_ids
        return result


class KeyedCombinedLinear(KeyedBase):

    def __init__(self, in_sequence_channels, in_pooled_channels, prediction_key_to_shape):
        super().__init__(prediction_key_to_shape)
        self.sequence_linear = nn.Linear(in_sequence_channels, sum(self.splits))
        self.pooled_linear = nn.Linear(in_pooled_channels, sum(self.splits))

    def forward(self, sequence_output, pooled_output, batch):
        pooled_predictions = self.pooled_linear(pooled_output)
        predictions = self.sequence_linear(sequence_output) + torch.unsqueeze(pooled_predictions, 1)
        predictions = torch.split(predictions, self.splits, dim=-1)
        result = OrderedDict()
        assert(len(self.prediction_key_to_shape) == len(predictions))
        for k, p in zip(self.prediction_key_to_shape, predictions):
            p = p.view(p.size()[:2] + self.prediction_key_to_shape[k])
            result[k] = p

        return result


class BertOutputSupplement(torch.nn.Module):

    def __init__(
            self, in_channels, supplemental_dropout_prob, is_sequence_supplement, supplement_key_to_shape,
            skip_dropout_keys=None):
        super().__init__()
        self.is_sequence_supplement = is_sequence_supplement
        self.in_channels = in_channels
        self.dropout = torch.nn.Dropout(supplemental_dropout_prob)
        self.supplement_key_to_shape = OrderedDict()
        if supplement_key_to_shape is not None:
            self.supplement_key_to_shape.update(supplement_key_to_shape)
        self.skip_dropout_keys = set()
        if skip_dropout_keys is not None:
            self.skip_dropout_keys.update(skip_dropout_keys)

    def supplement_channels(self):
        return sum(int(np.prod(self.supplement_key_to_shape[k])) for k in self.supplement_key_to_shape)

    def out_channels(self):
        return self.in_channels + self.supplement_channels()

    def forward(self, x, batch):
        # we expect that dropout has already been applied to sequence_output / pooled_output
        all_values = [x]
        for key in self.supplement_key_to_shape:
            values = batch[key]
            shape_part = values.size()[:2] if self.is_sequence_supplement else values.size()[:1]
            values = values.view(
                shape_part + (int(np.prod(self.supplement_key_to_shape[key])),)).type(all_values[0].dtype)
            if key not in self.skip_dropout_keys:
                values = self.dropout(values)
            all_values.append(values)
        return torch.cat(all_values, dim=2 if self.is_sequence_supplement else 1)


class MultiPredictionHead(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            supplemental_dropout_prob,
            prediction_head_settings,
            token_supplemental_key_to_shape=None,
            token_supplemental_skip_dropout_keys=None,
            pooled_supplemental_key_to_shape=None,
            pooled_supplemental_skip_dropout_keys=None):

        super().__init__()

        def _maybe_copy(x):
            return type(x)(x) if x is not None else None

        self._save_kwargs = dict(
            in_channels=in_channels,
            supplemental_dropout_prob=supplemental_dropout_prob,
            prediction_head_settings=list(prediction_head_settings),
            token_supplemental_key_to_shape=_maybe_copy(token_supplemental_key_to_shape),
            token_supplemental_skip_dropout_keys=_maybe_copy(token_supplemental_skip_dropout_keys),
            pooled_supplemental_key_to_shape=_maybe_copy(pooled_supplemental_key_to_shape),
            pooled_supplemental_skip_dropout_keys=_maybe_copy(pooled_supplemental_skip_dropout_keys))

        in_sequence_channels = in_channels
        in_pooled_channels = in_channels
        self.token_supplement = None
        if token_supplemental_key_to_shape is not None and len(token_supplemental_key_to_shape) > 0:
            self.token_supplement = BertOutputSupplement(
                in_channels,
                supplemental_dropout_prob=supplemental_dropout_prob,
                is_sequence_supplement=True,
                supplement_key_to_shape=token_supplemental_key_to_shape,
                skip_dropout_keys=token_supplemental_skip_dropout_keys)
            in_sequence_channels = self.token_supplement.out_channels()
        self.pooled_supplement = None
        if pooled_supplemental_key_to_shape is not None and len(pooled_supplemental_key_to_shape) > 0:
            self.pooled_supplement = BertOutputSupplement(
                in_channels,
                supplemental_dropout_prob=supplemental_dropout_prob,
                is_sequence_supplement=False,
                supplement_key_to_shape=pooled_supplemental_key_to_shape,
                skip_dropout_keys=pooled_supplemental_skip_dropout_keys)
            in_pooled_channels = self.pooled_supplement.out_channels()

        self.prediction_heads = torch.nn.ModuleDict(modules=[(
            ph[0].key,
            ph[0].head_type(
                in_sequence_channels=in_sequence_channels,
                in_pooled_channels=in_pooled_channels,
                prediction_key_to_shape=ph[1],
                **ph[0].kwargs))
            for ph in prediction_head_settings])

    def forward(self, sequence_output, pooled_output, batch, dataset):
        if self.token_supplement is not None:
            sequence_output = self.token_supplement(sequence_output, batch)
        if self.pooled_supplement is not None:
            pooled_output = self.pooled_supplement(pooled_output, batch)

        predictions = OrderedDict()
        for name in self.prediction_heads:
            head = self.prediction_heads[name]
            head_predictions = head(sequence_output, pooled_output, batch)
            for k in head_predictions:
                if k in predictions:
                    raise ValueError('multiple predictions made for key: {}'.format(k))
                else:
                    predictions[k] = head_predictions[k]

        # fetch the data that was too expensive to put in batch as padded
        for k in predictions:
            if isinstance(k, tuple) and len(k) == 2 and k[1] == 'data_ids':
                group_data = dataset.get_data_for_data_ids(k[0], predictions[k].cpu().numpy())
                batch[k[0]] = group_data.to(predictions[k].device)

        return predictions

    def save_kwargs(self, output_model_path):
        with open(os.path.join(output_model_path, 'kwargs.pkl'), 'wb') as f:
            pickle.dump(self._save_kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_kwargs(model_path):
        with open(os.path.join(model_path, 'kwargs.pkl'), 'rb') as f:
            return pickle.load(f)

    def update_state_dict(self, prefix, state_dict, saved_kwargs):
        saved_prediction_head_settings = dict((s[0].key, s[1]) for s in saved_kwargs['prediction_head_settings'])
        for name in self.prediction_heads:
            if name in saved_prediction_head_settings:
                self.prediction_heads[name].update_state_dict(
                    prefix + 'prediction_heads.' + name + '.', state_dict, saved_prediction_head_settings[name])
        return state_dict


class BertMultiPredictionHead(BertPreTrainedModel):

    def __init__(
            self,
            config,
            prediction_head_settings,
            token_supplemental_key_to_shape=None,
            token_supplemental_skip_dropout_keys=None,
            pooled_supplemental_key_to_shape=None,
            pooled_supplemental_skip_dropout_keys=None):

        super(BertMultiPredictionHead, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.prediction_head = MultiPredictionHead(
            config.hidden_size,
            config.hidden_dropout_prob,
            prediction_head_settings,
            token_supplemental_key_to_shape,
            token_supplemental_skip_dropout_keys,
            pooled_supplemental_key_to_shape,
            pooled_supplemental_skip_dropout_keys)
        self.apply(self.init_bert_weights)

    def save(self, output_model_path):
        output_model_file = os.path.join(output_model_path, WEIGHTS_NAME)
        torch.save(self.state_dict(), output_model_file)
        output_config_file = os.path.join(output_model_path, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(self.config.to_json_string())
        self.prediction_head.save_kwargs(output_model_path)

    @classmethod
    def load(cls, model_path, map_location='default_map_location'):
        config = BertConfig(os.path.join(model_path, CONFIG_NAME))
        kwargs = MultiPredictionHead.load_kwargs(model_path)
        signature = inspect.signature(cls.__init__)
        bad_keys = [k for k in kwargs if k not in signature.parameters]
        for k in bad_keys:
            del kwargs[k]
        bound = signature.bind_partial(**kwargs)
        model = cls(config, **bound.kwargs)

        if map_location == 'default_map_location':
            map_location = 'cpu' if not torch.cuda.is_available() else None

        state_dict = torch.load(os.path.join(model_path, WEIGHTS_NAME), map_location=map_location)

        model.load_state_dict(state_dict)

        return model

    @classmethod
    def from_fine_tuned(cls, model_path, map_location='default_map_location', *inputs, **kwargs):
        config = BertConfig(os.path.join(model_path, CONFIG_NAME))
        model = cls(config, *inputs, **kwargs)
        saved_kwargs = MultiPredictionHead.load_kwargs(model_path)
        if map_location == 'default_map_location':
            map_location = 'cpu' if not torch.cuda.is_available() else None
        state_dict = torch.load(os.path.join(model_path, WEIGHTS_NAME), map_location=map_location)
        model.prediction_head.update_state_dict('prediction_head.', state_dict, saved_kwargs)
        model.load_state_dict(state_dict, strict=False)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None,
                        from_tf=False, map_location='default_map_location', *inputs, **kwargs):
        """
        Copied from pytorch_pretrained_bert modeling.py so we can pass a map_location argument
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            if map_location == 'default_map_location':
                map_location = 'cpu' if not torch.cuda.is_available() else None
            state_dict = torch.load(weights_path, map_location)
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))
        return model

    def forward(self, batch, dataset):
        sequence_output, pooled_output = self.bert(
            batch['token_ids'],
            token_type_ids=batch['type_ids'] if 'type_ids' in batch else None,
            attention_mask=batch['mask'] if 'mask' in batch else None,
            output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        return self.prediction_head(sequence_output, pooled_output, batch, dataset)
