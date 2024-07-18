from cocoa.core.util import read_json, read_pickle
from core.price_tracker import PriceTracker
from model.generator import Templates, Generator
from model.manager import Manager

import options


def get_system(name, args, schema=None, timed=False, model_path=None):
    lexicon = PriceTracker(args.price_tracker_model)

    if name == 'rulebased':
        from .rulebased_system import RulebasedSystem
        templates = Templates.from_pickle(args.templates)
        generator = Generator(templates)
        manager = Manager.from_pickle(args.policy)
        return RulebasedSystem(lexicon, generator, manager, timed)
    
    elif name == 'hybrid':
        from .hybrid_system import HybridSystem
        from .neural_system import PytorchNeuralSystem
        templates = Templates.from_pickle(args.templates)
        manager = PytorchNeuralSystem(args, schema, lexicon, model_path, timed)
        generator = Generator(templates)
        return HybridSystem(lexicon, generator, manager, timed)
    
    elif name == 'cmd':
        from .cmd_system import CmdSystem
        return CmdSystem()
    
    elif name == 'pt-neural':
        from .neural_system import PytorchNeuralSystem
        assert model_path
        return PytorchNeuralSystem(args, schema, lexicon, model_path, timed)
    
    else:
        raise ValueError('Unknown system %s' % name)
