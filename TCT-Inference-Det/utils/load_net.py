import mxnet as mx
from mxnet.gluon import SymbolBlock

def load_model(param_file, symbol_file):
    net = SymbolBlock.imports(symbol_file=symbol_file,
                              input_names=["data"],
                              param_file=param_file)
    return net
