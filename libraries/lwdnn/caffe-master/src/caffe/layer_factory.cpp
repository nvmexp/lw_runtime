// Make sure we include Python.h before any system header
// to avoid _POSIX_C_SOURCE redefinition
#ifdef WITH_PYTHON_LAYER
#include <boost/python.hpp>
#include <boost/regex.hpp>
#endif
#include <string>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/colw_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/detectnet_transform_layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/layers/window_data_layer.hpp"

#ifdef USE_LWDNN
#include "caffe/layers/lwdnn_batch_norm_layer.hpp"
#include "caffe/layers/lwdnn_colw_layer.hpp"
#include "caffe/layers/lwdnn_lcn_layer.hpp"
#include "caffe/layers/lwdnn_lrn_layer.hpp"
#include "caffe/layers/lwdnn_pooling_layer.hpp"
#include "caffe/layers/lwdnn_relu_layer.hpp"
#include "caffe/layers/lwdnn_sigmoid_layer.hpp"
#include "caffe/layers/lwdnn_softmax_layer.hpp"
#include "caffe/layers/lwdnn_tanh_layer.hpp"
#include "caffe/layers/lwdnn_dropout_layer.hpp"
#endif

#ifdef WITH_PYTHON_LAYER
#include "caffe/layers/python_layer.hpp"
#endif

#pragma GCC diagnostic ignored "-Wreturn-type"

namespace caffe {

// Get colwolution layer according to engine.
shared_ptr<LayerBase> GetColwolutionLayer(const LayerParameter& param,
    Type ftype, Type btype, size_t) {
  ColwolutionParameter colw_param = param.colwolution_param();
  ColwolutionParameter_Engine engine = colw_param.engine();
  if (engine == ColwolutionParameter_Engine_DEFAULT) {
    engine = ColwolutionParameter_Engine_CAFFE;
#ifdef USE_LWDNN
    if (Caffe::mode() == Caffe::GPU) {
      engine = ColwolutionParameter_Engine_LWDNN;
    }
#endif
  }
  if (engine == ColwolutionParameter_Engine_CAFFE) {
    return CreateLayerBase<ColwolutionLayer>(param, ftype, btype);
#ifdef USE_LWDNN
  } else if (engine == ColwolutionParameter_Engine_LWDNN) {
    return CreateLayerBase<LwDNNColwolutionLayer>(param, ftype, btype);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Colwolution, GetColwolutionLayer);

// Get BN layer according to engine.
shared_ptr<LayerBase> GetBatchNormLayer(const LayerParameter& param,
    Type ftype, Type btype, size_t) {
  BatchNormParameter_Engine engine = param.batch_norm_param().engine();
  if (engine == BatchNormParameter_Engine_DEFAULT) {
    engine = BatchNormParameter_Engine_CAFFE;
#ifdef USE_LWDNN
    engine = BatchNormParameter_Engine_LWDNN;
#endif
  }
  if (engine == BatchNormParameter_Engine_CAFFE) {
    return CreateLayerBase<BatchNormLayer>(param, ftype, btype);
#ifdef USE_LWDNN
  } else if (engine == BatchNormParameter_Engine_LWDNN) {
    return CreateLayerBase<LwDNNBatchNormLayer>(param, ftype, btype);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(BatchNorm, GetBatchNormLayer);

// Get pooling layer according to engine.
shared_ptr<LayerBase> GetPoolingLayer(const LayerParameter& param,
    Type ftype, Type btype, size_t) {
  PoolingParameter_Engine engine = param.pooling_param().engine();
  if (engine == PoolingParameter_Engine_DEFAULT) {
    engine = PoolingParameter_Engine_CAFFE;
#ifdef USE_LWDNN
    if (Caffe::mode() == Caffe::GPU)
      engine = PoolingParameter_Engine_LWDNN;
#endif
  }
  if (engine == PoolingParameter_Engine_CAFFE) {
    return CreateLayerBase<PoolingLayer>(param, ftype, btype);
#ifdef USE_LWDNN
  } else if (engine == PoolingParameter_Engine_LWDNN) {
    if (param.top_size() > 1) {
      LOG(INFO) << "lwDNN does not support multiple tops. "
                << "Using Caffe's own pooling layer.";
      return CreateLayerBase<PoolingLayer>(param, ftype, btype);
    }
    return CreateLayerBase<LwDNNPoolingLayer>(param, ftype, btype);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayer);

// Get LRN layer according to engine
shared_ptr<LayerBase> GetLRNLayer(const LayerParameter& param,
    Type ftype, Type btype, size_t) {
  LRNParameter_Engine engine = param.lrn_param().engine();

  if (engine == LRNParameter_Engine_DEFAULT) {
    engine = LRNParameter_Engine_CAFFE;
#ifdef USE_LWDNN
    if (Caffe::mode() == Caffe::GPU)
      engine = LRNParameter_Engine_LWDNN;
#endif
  }

  if (engine == LRNParameter_Engine_CAFFE) {
    return CreateLayerBase<LRNLayer>(param, ftype, btype);
#ifdef USE_LWDNN
  } else if (engine == LRNParameter_Engine_LWDNN) {
    LRNParameter lrn_param = param.lrn_param();

    if (lrn_param.norm_region() ==LRNParameter_NormRegion_WITHIN_CHANNEL) {
      return CreateLayerBase<LwDNNLCNLayer>(param, ftype, btype);
    } else {
      // local size is too big to be handled through lwDNN
      if (param.lrn_param().local_size() > LWDNN_LRN_MAX_N) {
        return CreateLayerBase<LRNLayer>(param, ftype, btype);
      } else {
        return CreateLayerBase<LwDNNLRNLayer>(param, ftype, btype);
      }
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(LRN, GetLRNLayer);

// Get relu layer according to engine.
shared_ptr<LayerBase> GetReLULayer(const LayerParameter& param,
    Type ftype, Type btype, size_t) {
  ReLUParameter_Engine engine = param.relu_param().engine();
  if (engine == ReLUParameter_Engine_DEFAULT) {
    engine = ReLUParameter_Engine_CAFFE;
#ifdef USE_LWDNN
    if (Caffe::mode() == Caffe::GPU)
      engine = ReLUParameter_Engine_LWDNN;
#endif
  }
  if (engine == ReLUParameter_Engine_CAFFE) {
    return CreateLayerBase<ReLULayer>(param, ftype, btype);
#ifdef USE_LWDNN
  } else if (engine == ReLUParameter_Engine_LWDNN) {
    return CreateLayerBase<LwDNNReLULayer>(param, ftype, btype);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(ReLU, GetReLULayer);

// Get sigmoid layer according to engine.
shared_ptr<LayerBase> GetSigmoidLayer(const LayerParameter& param,
    Type ftype, Type btype, size_t) {
  SigmoidParameter_Engine engine = param.sigmoid_param().engine();
  if (engine == SigmoidParameter_Engine_DEFAULT) {
    engine = SigmoidParameter_Engine_CAFFE;
#ifdef USE_LWDNN
    if (Caffe::mode() == Caffe::GPU)
      engine = SigmoidParameter_Engine_LWDNN;
#endif
  }
  if (engine == SigmoidParameter_Engine_CAFFE) {
    return CreateLayerBase<SigmoidLayer>(param, ftype, btype);
#ifdef USE_LWDNN
  } else if (engine == SigmoidParameter_Engine_LWDNN) {
    return CreateLayerBase<LwDNNSigmoidLayer>(param, ftype, btype);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Sigmoid, GetSigmoidLayer);

// Get softmax layer according to engine.
shared_ptr<LayerBase> GetSoftmaxLayer(const LayerParameter& param,
    Type ftype, Type btype, size_t) {
  LayerParameter lparam(param);
  SoftmaxParameter_Engine engine = lparam.softmax_param().engine();
  if (engine == SoftmaxParameter_Engine_DEFAULT) {
    engine = SoftmaxParameter_Engine_CAFFE;
#ifdef USE_LWDNN
    if (Caffe::mode() == Caffe::GPU)
      engine = SoftmaxParameter_Engine_LWDNN;
#endif
  }
  if (engine == SoftmaxParameter_Engine_CAFFE) {
    return CreateLayerBase<SoftmaxLayer>(lparam, ftype, btype);
#ifdef USE_LWDNN
  } else if (engine == SoftmaxParameter_Engine_LWDNN) {
    return CreateLayerBase<LwDNNSoftmaxLayer>(lparam, ftype, btype);
#endif
  } else {
    LOG(FATAL) << "Layer " << lparam.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Softmax, GetSoftmaxLayer);

// Get tanh layer according to engine.
shared_ptr<LayerBase> GetTanHLayer(const LayerParameter& param,
    Type ftype, Type btype, size_t) {
  TanHParameter_Engine engine = param.tanh_param().engine();
  if (engine == TanHParameter_Engine_DEFAULT) {
    engine = TanHParameter_Engine_CAFFE;
#ifdef USE_LWDNN
    if (Caffe::mode() == Caffe::GPU)
      engine = TanHParameter_Engine_LWDNN;
#endif
  }
  if (engine == TanHParameter_Engine_CAFFE) {
    return CreateLayerBase<TanHLayer>(param, ftype, btype);
#ifdef USE_LWDNN
  } else if (engine == TanHParameter_Engine_LWDNN) {
    return CreateLayerBase<LwDNNTanHLayer>(param, ftype, btype);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}
REGISTER_LAYER_CREATOR(TanH, GetTanHLayer);

// Get dropout layer according to engine
shared_ptr<LayerBase> GetDropoutLayer(const LayerParameter& param,
  Type ftype, Type btype, size_t) {
  DropoutParameter_Engine engine = param.dropout_param().engine();
  if (engine == DropoutParameter_Engine_DEFAULT) {
    engine = DropoutParameter_Engine_CAFFE;
#ifdef USE_LWDNN
    if (Caffe::mode() == Caffe::GPU) {
      engine = DropoutParameter_Engine_LWDNN;
    }
#endif
  }
  if (engine == DropoutParameter_Engine_CAFFE) {
    return CreateLayerBase<DropoutLayer>(param, ftype, btype);
  }
#ifdef USE_LWDNN
  else if (engine == DropoutParameter_Engine_LWDNN) {
    return CreateLayerBase<LwDNNDropoutLayer>(param, ftype, btype);
  }
#endif
  else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}
REGISTER_LAYER_CREATOR(Dropout, GetDropoutLayer);

shared_ptr<LayerBase> GetMemoryDataLayer(const LayerParameter& param,
    Type ftype, Type btype, size_t) {
  LayerParameter lparam(param);
  check_precision_support(ftype, btype, lparam);
  shared_ptr<LayerBase> ret;
  if (is_type<double>(ftype)) {
    ret.reset(new MemoryDataLayer<double, double>(lparam));
  } else {
    ret.reset(new MemoryDataLayer<float, float>(lparam));
  }
  return ret;
}
REGISTER_LAYER_CREATOR(MemoryData, GetMemoryDataLayer);

shared_ptr<LayerBase> GetWindowDataLayer(const LayerParameter& param, Type ftype, Type btype,
    size_t solver_rank) {
  LayerParameter lparam(param);
  check_precision_support(ftype, btype, lparam);
  shared_ptr<LayerBase> ret;
  if (is_type<double>(ftype)) {
    ret.reset(new WindowDataLayer<double, double>(lparam, solver_rank));
  } else {
    ret.reset(new WindowDataLayer<float, float>(lparam, solver_rank));
  }
  return ret;
}
REGISTER_LAYER_CREATOR(WindowData, GetWindowDataLayer);

shared_ptr<LayerBase> GetDetectNetTransformationLayer(const LayerParameter& param,
    Type ftype, Type btype, size_t) {
  LayerParameter lparam(param);
  check_precision_support(ftype, btype, lparam);
  shared_ptr<LayerBase> ret;
  if (is_type<double>(ftype)) {
    ret.reset(new DetectNetTransformationLayer<double>(lparam));
  } else {
    ret.reset(new DetectNetTransformationLayer<float>(lparam));
  }
  return ret;
}
REGISTER_LAYER_CREATOR(DetectNetTransformation, GetDetectNetTransformationLayer);

#ifdef WITH_PYTHON_LAYER
shared_ptr<LayerBase> GetPythonLayer(const LayerParameter& param, Type, Type, size_t) {
  try {
    const string& module_name = param.python_param().module();
    const string& layer_name = param.python_param().layer();
    // Check injection. This allows nested import.
    boost::regex expression("[a-zA-Z_][a-zA-Z0-9_]*(\\.[a-zA-Z_][a-zA-Z0-9_]*)*");
    CHECK(boost::regex_match(module_name, expression))
        << "Module name is invalid: " << module_name;
    CHECK(boost::regex_match(layer_name, expression))
        << "Layer name is invalid: " << layer_name;

    PyGILAquire gil;
    LOG(INFO) << "Importing Python module '" << module_name << "'";
    bp::object module = bp::import(module_name.c_str());
    bp::object layer = module.attr(layer_name.c_str())(param);
    shared_ptr<LayerBase> ret = bp::extract<shared_ptr<LayerBase>>(layer)();
    CHECK(ret);
    return ret;
//    bp::object globals = bp::import("__main__").attr("__dict__");
//    bp::exec(("import " + module_name).c_str(), globals, globals);
//    bp::object layer_class = bp::eval((module_name + "." + layer_name).c_str(), globals, globals);
//    bp::object layer = layer_class(param);
//    return bp::extract<shared_ptr<LayerBase>>(layer)();
  } catch (...) {
    PyErrFatal();
  }
}

REGISTER_LAYER_CREATOR(Python, GetPythonLayer);
#endif

void check_precision_support(Type& ftype, Type& btype, LayerParameter& param, bool transf) {
  if (!is_precise(ftype) || !is_precise(btype) || transf) {
    Type MT = tp<float>();
    if (Caffe::root_solver()) {
      if (transf) {
        LOG(WARNING) << "Layer '" << param.name() << "' of type '"
                     << param.type() << "' has transform settings not supported in "
                     << Type_Name(FLOAT16) << " precision. Falling back to " << Type_Name(MT);
      } else {
        LOG(WARNING) << "Layer '" << param.name() << "' of type '"
                     << param.type() << "' is not supported in " << Type_Name(FLOAT16)
                     << " precision. Falling back to " << Type_Name(MT) << ". You might use "
                         "'forward_type: FLOAT' and 'backward_type: FLOAT' "
                         "settings to suppress this warning.";
      }
    }
    ftype = MT;
    btype = MT;
    param.set_forward_type(MT);
    param.set_backward_type(MT);
    param.set_forward_math(MT);
    param.set_backward_math(MT);
  }
}

// Layers that use their constructor as their default creator should be
// registered in their corresponding cpp files. Do not register them here.
}  // namespace caffe
