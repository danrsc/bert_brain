import inspect
import logging
import os
import pickle
import shutil
import tarfile
import tempfile
from collections import OrderedDict
from copy import deepcopy
from inspect import signature

import numpy as np
import torch
import torch.nn
from pytorch_pretrained_bert import BertModel, BertConfig, cached_path, load_tf_weights_in_bert
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, WEIGHTS_NAME, CONFIG_NAME, \
    PRETRAINED_MODEL_ARCHIVE_MAP, TF_WEIGHTS_NAME

__all__ = ['MultiPredictionHead', 'BertMultiPredictionHead', 'BertOutputSupplement', 'LazyBertOutputBatch',
           'LazyBertOutputNumChannels']


logger = logging.getLogger(__name__)


class MultiPredictionHead(torch.nn.Module):

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        sig = signature(cls.__init__)
        bound_arguments = sig.bind_partial(*args, **kwargs)
        bound_arguments.apply_defaults()
        obj._bound_arguments = deepcopy(bound_arguments.arguments)
        return obj

    def __init__(
            self,
            in_channels,
            num_in_layers,
            dropout_prob,
            head_graph_parts,
            token_supplemental_key_to_shape=None,
            token_supplemental_skip_dropout_keys=None,
            pooled_supplemental_key_to_shape=None,
            pooled_supplemental_skip_dropout_keys=None):

        super().__init__()

        self.dropout = torch.nn.Dropout(dropout_prob)

        self.token_supplement = None
        if token_supplemental_key_to_shape is not None and len(token_supplemental_key_to_shape) > 0:
            self.token_supplement = BertOutputSupplement(
                in_channels,
                supplemental_dropout_prob=dropout_prob,
                is_sequence_supplement=True,
                supplement_key_to_shape=token_supplemental_key_to_shape,
                skip_dropout_keys=token_supplemental_skip_dropout_keys)
        self.pooled_supplement = None
        if pooled_supplemental_key_to_shape is not None and len(pooled_supplemental_key_to_shape) > 0:
            self.pooled_supplement = BertOutputSupplement(
                in_channels,
                supplemental_dropout_prob=dropout_prob,
                is_sequence_supplement=False,
                supplement_key_to_shape=pooled_supplemental_key_to_shape,
                skip_dropout_keys=pooled_supplemental_skip_dropout_keys)

        name_to_num_channels = LazyBertOutputNumChannels(
            self.token_supplement, self.pooled_supplement, in_channels, num_in_layers)
        for key in head_graph_parts:
            graph_part_num_channels = head_graph_parts[key].instantiate(name_to_num_channels.copy())
            for k in graph_part_num_channels:
                if k in name_to_num_channels:
                    raise ValueError('Duplicate output: {}'.format(k))
                name_to_num_channels[k] = graph_part_num_channels[k]

        # noinspection PyTypeChecker
        self.head_graph_parts = torch.nn.ModuleDict(modules=[(k, head_graph_parts[k]) for k in head_graph_parts])

    def forward(self, sequence_output, pooled_output, batch, dataset):
        batch_inputs = LazyBertOutputBatch(
            sequence_output,
            pooled_output,
            batch,
            self.dropout,
            self.token_supplement,
            self.pooled_supplement)

        outputs = OrderedDict()
        # noinspection PyTypeChecker
        for name in self.head_graph_parts:
            # noinspection PyUnresolvedReferences
            head = self.head_graph_parts[name]
            head_outputs = head(batch_inputs)

            for k in head_outputs:
                if k in outputs:  # don't allow two graph parts to output the same key
                    raise ValueError('multiple predictions made for key: {}'.format(k))
                else:
                    outputs[k] = head_outputs[k]
                    # overwrite batch_inputs[k] if it exists
                    batch_inputs[k] = head_outputs[k]

        # pass through anything that was not output by a graph part
        for k in batch:
            if k not in outputs:
                outputs[k] = batch[k]

        # set the outputs as the batch, which now includes passed through keys
        batch = outputs

        return batch

    def save_kwargs(self, output_model_path):
        with open(os.path.join(output_model_path, 'kwargs.pkl'), 'wb') as f:
            pickle.dump(self._bound_arguments, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_kwargs(model_path):
        with open(os.path.join(model_path, 'kwargs.pkl'), 'rb') as f:
            result = pickle.load(f)
            if 'supplemental_dropout_prob' in result:  # backwards compatible for now
                result['dropout_prob'] = result['supplemental_dropout_prob']
                del result['supplemental_dropout_prob']
            return result

    def update_state_dict(self, prefix, state_dict, saved_kwargs):
        saved_prediction_head_settings = dict((s[0].key, s[1]) for s in saved_kwargs['prediction_head_settings'])
        for name in self.prediction_heads:
            if name in saved_prediction_head_settings:
                self.prediction_heads[name].update_state_dict(
                    prefix + 'prediction_heads.' + name + '.', state_dict, saved_prediction_head_settings[name])
        return state_dict


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


class BertMultiPredictionHead(BertPreTrainedModel):

    def __init__(
            self,
            config,
            head_graph_parts,
            token_supplemental_key_to_shape=None,
            token_supplemental_skip_dropout_keys=None,
            pooled_supplemental_key_to_shape=None,
            pooled_supplemental_skip_dropout_keys=None):

        super(BertMultiPredictionHead, self).__init__(config)
        self.bert = BertModel(config)
        # noinspection PyUnresolvedReferences
        self.prediction_head = MultiPredictionHead(
            config.hidden_size,
            len(self.bert.encoder.layer),
            config.hidden_dropout_prob,
            head_graph_parts,
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
        # noinspection PyUnresolvedReferences
        self.prediction_head.save_kwargs(output_model_path)

    @classmethod
    def load(cls, model_path, map_location='default_map_location'):
        config = BertConfig(os.path.join(model_path, CONFIG_NAME))
        kwargs = MultiPredictionHead.load_kwargs(model_path)
        sig = inspect.signature(cls.__init__)
        bad_keys = [k for k in kwargs if k not in sig.parameters]
        for k in bad_keys:
            del kwargs[k]
        bound = sig.bind_partial(**kwargs)
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
        # noinspection PyUnresolvedReferences
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
                - a path or url to a pre-trained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pre-trained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionary (collections.OrderedDict object) to use instead of Google
                pre-trained models
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
            # noinspection PyProtectedMember
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            # noinspection PyProtectedMember
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
            output_all_encoded_layers=True)
        # noinspection PyCallingNonCallable
        return self.prediction_head(sequence_output, pooled_output, batch, dataset)

    def to(self, *args, **kwargs):

        # noinspection PyProtectedMember, PyUnresolvedReferences
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

        if dtype is not None:
            if not dtype.is_floating_point:
                raise TypeError('nn.Module.to only accepts floating point '
                                'dtypes, but got desired dtype={}'.format(dtype))

        forced_cpu = list()

        def set_forced_cpu(module):
            for child in module.children():
                set_forced_cpu(child)
            force_cpu = getattr(module, 'force_cpu', False)
            if force_cpu:
                def set_forced_cpu_tensor(t):
                    forced_cpu.append(t)
                    return t

                # noinspection PyProtectedMember
                module._apply(set_forced_cpu_tensor)

        set_forced_cpu(self)

        def is_forced_cpu(t):
            for have in forced_cpu:
                if have.is_set_to(t):
                    return True
            return False

        def convert(t):
            if is_forced_cpu(t):
                return t.to(torch.device('cpu'), dtype if t.is_floating_point() else None, non_blocking)
            else:
                return t.to(device, dtype if t.is_floating_point() else None, non_blocking)

        self._apply(convert)

    def cuda(self, device=None):
        forced_cpu = list()

        def set_forced_cpu(module):
            for child in module.children():
                set_forced_cpu(child)
            force_cpu = getattr(module, 'force_cpu', False)
            if force_cpu:
                def set_forced_cpu_tensor(t):
                    forced_cpu.append(t)
                    return t

                # noinspection PyProtectedMember
                module._apply(set_forced_cpu_tensor)

        set_forced_cpu(self)

        def is_forced_cpu(t):
            for have in forced_cpu:
                if have.is_set_to(t):
                    return True
            return False

        def convert(t):
            if is_forced_cpu(t):
                return t.cpu()
            return t.cuda(device)
        return self._apply(convert)


class LazyBertOutputNumChannels:

    def __init__(self, sequence_supplement, pooled_supplement, in_channels, num_layers):
        self.sequence_supplement = sequence_supplement
        self.pooled_supplement = pooled_supplement
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.name_to_num_channels = OrderedDict()

    def copy(self):
        result = LazyBertOutputNumChannels(
            self.sequence_supplement, self.pooled_supplement, self.in_channels, self.num_layers)
        result.name_to_num_channels = OrderedDict(self.name_to_num_channels)
        return result

    def __delitem__(self, key):
        del self.name_to_num_channels[key]

    def __iter__(self):
        return iter(self.name_to_num_channels)

    def __len__(self):
        return len(self.name_to_num_channels)

    def __setitem__(self, key, value):
        self.name_to_num_channels[key] = value

    def __getitem__(self, item):
        if item in self.name_to_num_channels:
            return self.name_to_num_channels[item]

        if isinstance(item, (tuple, list)):
            if item[0] == 'bert':
                kind = item[1]
                layer = item[2] if len(item) > 2 else -1
                if kind == 'sequence' or kind == 'untransformed_pooled':
                    if layer == 'all':
                        num_layers = self.num_layers
                    elif np.ndim(layer) == 0:
                        num_layers = 1
                    else:
                        num_layers = len(layer)
                    if kind == 'untransformed_pooled':
                        supplement = self.pooled_supplement
                    else:
                        supplement = self.sequence_supplement
                elif kind == 'pooled':
                    if np.ndim(layer) != 0 or not isinstance(layer, int):
                        raise KeyError('Cannot get pooled result from a layer other than the last layer. '
                                       'Requested layer: {}. Did you mean untransformed_pooled?'.format(layer))
                    if layer < 0:
                        layer += self.num_layers
                    if layer != self.num_layers - 1:
                        raise KeyError('Cannot get pooled result from a layer other than the last layer. '
                                       'Requested layer: {}. Did you mean untransformed_pooled?'.format(item[2]))
                    num_layers = 1
                    supplement = self.pooled_supplement
                else:
                    raise KeyError('Unrecognized kind: {}'.format(kind))

                self.name_to_num_channels[item] = self.in_channels * num_layers + (
                    supplement.supplement_channels() if supplement is not None else 0)

        return self.name_to_num_channels[item]


class LazyBertOutputBatch:

    def __init__(
            self,
            sequence_output,
            pooled_output,
            batch,
            dropout_layer,
            sequence_supplement,
            pooled_supplement):
        self.pooled_output = pooled_output
        self.sequence_output = sequence_output
        self.batch = OrderedDict(batch)
        self._dropped_out = dict()
        self.dropout_layer = dropout_layer
        self.sequence_supplement = sequence_supplement
        self.pooled_supplement = pooled_supplement

    def __delitem__(self, key):
        del self.batch[key]

    def __iter__(self):
        return iter(self.batch)

    def __len__(self):
        return len(self.batch)

    def __setitem__(self, key, value):
        self.batch[key] = value

    def __contains__(self, item):
        try:
            _ = self[item]
            return True
        except KeyError:
            return False

    def __getitem__(self, item):
        if item in self.batch:
            return self.batch[item]

        if isinstance(item, (tuple, list)):
            if item[0] == 'bert':
                kind = item[1]
                layer = item[2] if len(item) > 2 else -1
                if kind == 'sequence' or kind == 'untransformed_pooled':
                    if layer == 'all':
                        indices = range(len(self.sequence_output))
                    elif np.ndim(layer) == 0:
                        indices = [layer]
                    else:
                        indices = layer
                    x = list()
                    for layer in indices:
                        if layer < 0:
                            layer += len(self.sequence_output)
                            if layer < 0 or layer >= len(self.sequence_output):
                                raise KeyError('Invalid layer requested: {}'.format(item[2]))
                        if ('sequence', layer) not in self._dropped_out:
                            self._dropped_out[('sequence', layer)] = self.dropout_layer(self.sequence_output[layer])
                        x.append(self._dropped_out[('sequence', layer)])
                    if kind == 'untransformed_pooled':
                        x = [x_[:, 0] for x_ in x]
                        supplement = self.pooled_supplement
                    else:
                        supplement = self.sequence_supplement
                elif kind == 'pooled':
                    if np.ndim(layer) != 0 or not isinstance(layer, int):
                        raise KeyError('Cannot get pooled result from a layer other than the last layer. '
                                       'Requested layer: {}. Did you mean untransformed_pooled?'.format(layer))
                    if layer < 0:
                        layer += len(self.sequence_output)
                    if layer != len(self.sequence_output) - 1:
                        raise KeyError('Cannot get pooled result from a layer other than the last layer. '
                                       'Requested layer: {}. Did you mean untransformed_pooled?'.format(item[2]))
                    if (kind, layer) not in self._dropped_out:
                        self._dropped_out[(kind, layer)] = self.dropout_layer(self.pooled_output)
                    x = [self._dropped_out[(kind, layer)]]
                    supplement = self.pooled_supplement
                else:
                    raise KeyError('Unrecognized kind: {}'.format(kind))

                if len(x) == 1:
                    x = x[0]
                else:
                    x = torch.cat(x, dim=-1)
                if supplement is not None:
                    self.batch[item] = supplement(x, self.batch)
                else:
                    self.batch[item] = x

        return self.batch[item]
