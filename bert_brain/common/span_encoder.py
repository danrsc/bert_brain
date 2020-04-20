from collections import OrderedDict

import numpy as np
import torch


__all__ = ['NamedSpanEncoder']


class NamedSpanEncoder:

    def __init__(self, names_in_order=None, frozen_names=False):
        if frozen_names and (names_in_order is None or len(names_in_order) == 0):
            raise ValueError('frozen_names is True, but no names given')
        if names_in_order is not None:
            self.names = OrderedDict((n, i) for i, n in enumerate(names_in_order))
        else:
            self.names = OrderedDict()
        self.frozen_names = frozen_names

    def encode(self, active_names):
        if np.ndim(active_names) == 0:
            active_names = [active_names]
        result = 0
        for name in active_names:
            if name not in self.names:
                if self.frozen_names:
                    raise ValueError('Invalid name: {}'.format(name))
                self.names[name] = len(self.names)
            span_id = self.names[name]
            result += 1 << span_id
        return result

    def decode(self, encoded_span_ids):
        active_names = list()
        round_trip = 0
        for name in self.names:
            span_id = self.names[name]
            mask = 1 << span_id
            if mask & encoded_span_ids == mask:
                active_names.append(name)
                round_trip += mask
        if round_trip != encoded_span_ids:
            raise ValueError('Bad value for encoded_span_ids: {}'.format(encoded_span_ids))
        return active_names

    def masks(self):
        return OrderedDict((n, 1 << self.names[n]) for n in self.names)

    def torch_span_indicators(self, multi_encoded_span_ids):
        result = OrderedDict()
        round_trips = torch.zeros_like(multi_encoded_span_ids)
        for name in self.names:
            span_id = self.names[name]
            mask = (1 << span_id)
            masked = multi_encoded_span_ids & mask
            result[name] = masked == mask
            round_trips += masked
        if not torch.equal(round_trips, multi_encoded_span_ids):
            return result, False
        return result, True
