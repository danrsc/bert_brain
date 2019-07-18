from collections import OrderedDict


__all__ = ['NamedSpanEncoder']


class NamedSpanEncoder:

    def __init__(self, names_in_order=None, frozen_names=False):
        if frozen_names and names_in_order is None or len(names_in_order) == 0:
            raise ValueError('frozen_names is True, but no names given')
        if names_in_order is not None:
            self.names = OrderedDict((n, i) for i, n in enumerate(names_in_order))
        self.frozen_names = frozen_names

    def encode(self, active_names):
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
