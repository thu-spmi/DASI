"""Config registry.

Config registry is just a dictionary of config, but it copies
the configs on read and on write to avoid subtle bugs.

"""

import copy

class ConfigRegistry:

    def __init__(self):
        self._configs = {}

    def __getitem__(self, name):
        # copy on read
        return copy.deepcopy(self._configs[name])

    def __setitem__(self, name, config):
        if name in self._configs:
            raise KeyError("Config already registered " + name)
        self._configs[name] = copy.deepcopy(config)

    def __contains__(self, item):
        if item not in self._configs.keys():
            raise KeyError('Cant find suitable configs!')
        return item in self._configs.keys()

    def get_root_config(self):
        return self['root']

    def set_root_config(self, config):
        self['root'] = config

    def keys(self):
        return self._configs.keys()

    def show(self):
        import pprint
        for k , v in self._configs.items():
            print('config:', k, v)

