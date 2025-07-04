from .pycaffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, RMSPropSolver, \
    AdaDeltaSolver, AdamSolver
from ._caffe import set_mode_cpu, set_mode_gpu, set_device, set_devices, Layer, LayerParameter, \
    LayerBase, get_solver, layer_type_list, create_layer
from ._caffe import CAFFE_VERSION as __version__
from .proto.caffe_pb2 import TRAIN, TEST
from .classifier import Classifier
from .detector import Detector
from . import io
from .net_spec import layers, params, NetSpec, to_proto
