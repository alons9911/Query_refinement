import json

from numpy import int64


class Decoder(json.JSONDecoder):
    def decode(self, result, **kwargs):
        return self._decode(result)

    def _decode(self, o):
        if isinstance(o, int64):
            try:
                return int(o)
            except ValueError:
                return o
        elif isinstance(o, dict):
            return {k: self._decode(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self._decode(v) for v in o]
        else:
            return o
