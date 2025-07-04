#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/relwrrent_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void RelwrrentLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "bottom[0] must have at least 2 axes -- (#timesteps, #streams, ...)";
  T_ = bottom[0]->shape(0);
  N_ = bottom[0]->shape(1);
  LOG(INFO) << "Initializing relwrrent layer: assuming input batch contains "
            << T_ << " timesteps of " << N_ << " independent streams.";

  CHECK_EQ(bottom[1]->num_axes(), 2)
      << "bottom[1] must have exactly 2 axes -- (#timesteps, #streams)";
  CHECK_EQ(T_, bottom[1]->shape(0));
  CHECK_EQ(N_, bottom[1]->shape(1));

  // If expose_hidden is set, we take as input and produce as output
  // the hidden state blobs at the first and last timesteps.
  expose_hidden_ = this->layer_param_.relwrrent_param().expose_hidden();

  // Get (relwrrent) input/output names.
  vector<string> output_names;
  OutputBlobNames(&output_names);
  vector<string> relwr_input_names;
  RelwrrentInputBlobNames(&relwr_input_names);
  vector<string> relwr_output_names;
  RelwrrentOutputBlobNames(&relwr_output_names);
  const int num_relwr_blobs = relwr_input_names.size();
  CHECK_EQ(num_relwr_blobs, relwr_output_names.size());

  // If provided, bottom[2] is a static input to the relwrrent net.
  const int num_hidden_exposed = expose_hidden_ * num_relwr_blobs;
  static_input_ = (bottom.size() > 2 + num_hidden_exposed);
  if (static_input_) {
    CHECK_GE(bottom[2]->num_axes(), 1);
    CHECK_EQ(N_, bottom[2]->shape(0));
  }

  // Create a NetParameter; setup the inputs that aren't unique to particular
  // relwrrent architectures.
  NetParameter net_param;
  net_param.set_default_forward_type(tp<Ftype>());
  net_param.set_default_backward_type(tp<Btype>());
  net_param.set_default_forward_math(tp<Ftype>());
  net_param.set_default_backward_math(tp<Btype>());

  LayerParameter* input_layer_param = net_param.add_layer();
  input_layer_param->set_type("Input");
  InputParameter* input_param = input_layer_param->mutable_input_param();
  input_layer_param->add_top("x");
  BlobShape input_shape;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    input_shape.add_dim(bottom[0]->shape(i));
  }
  input_param->add_shape()->CopyFrom(input_shape);

  input_shape.Clear();
  for (int i = 0; i < bottom[1]->num_axes(); ++i) {
    input_shape.add_dim(bottom[1]->shape(i));
  }
  input_layer_param->add_top("cont");
  input_param->add_shape()->CopyFrom(input_shape);

  if (static_input_) {
    input_shape.Clear();
    for (int i = 0; i < bottom[2]->num_axes(); ++i) {
      input_shape.add_dim(bottom[2]->shape(i));
    }
    input_layer_param->add_top("x_static");
    input_param->add_shape()->CopyFrom(input_shape);
  }

  // Call the child's FillUnrolledNet implementation to specify the unrolled
  // relwrrent architecture.
  this->FillUnrolledNet(&net_param);

  // Prepend this layer's name to the names of each layer in the unrolled net.
  const string& layer_name = this->layer_param_.name();
  if (layer_name.size()) {
    for (int i = 0; i < net_param.layer_size(); ++i) {
      LayerParameter* layer = net_param.mutable_layer(i);
      layer->set_name(layer_name + "_" + layer->name());
    }
  }

  // Add "pseudo-losses" to all outputs to force backpropagation.
  // (Setting force_backward is too aggressive as we may not need to backprop to
  // all inputs, e.g., the sequence continuation indicators.)
  vector<string> pseudo_losses(output_names.size());
  for (int i = 0; i < output_names.size(); ++i) {
    LayerParameter* layer = net_param.add_layer();
    pseudo_losses[i] = output_names[i] + "_pseudoloss";
    layer->set_name(pseudo_losses[i]);
    layer->set_type("Reduction");
    layer->add_bottom(output_names[i]);
    layer->add_top(pseudo_losses[i]);
    layer->add_loss_weight(1);
  }

  const size_t rank = this->parent_rank();
  // Create the unrolled net.
  unrolled_net_.reset(new Net(net_param, rank, nullptr, nullptr, true));
  unrolled_net_->set_debug_info(
      this->layer_param_.relwrrent_param().debug_info());

  // Setup pointers to the inputs.
  x_input_blob_ = CHECK_NOTNULL(unrolled_net_->blob_by_name("x").get());
  cont_input_blob_ = CHECK_NOTNULL(unrolled_net_->blob_by_name("cont").get());
  if (static_input_) {
    x_static_input_blob_ =
        CHECK_NOTNULL(unrolled_net_->blob_by_name("x_static").get());
  }

  // Setup pointers to paired relwrrent inputs/outputs.
  relwr_input_blobs_.resize(num_relwr_blobs);
  relwr_output_blobs_.resize(num_relwr_blobs);
  for (int i = 0; i < relwr_input_names.size(); ++i) {
    relwr_input_blobs_[i] =
        CHECK_NOTNULL(unrolled_net_->blob_by_name(relwr_input_names[i]).get());
    relwr_output_blobs_[i] =
        CHECK_NOTNULL(unrolled_net_->blob_by_name(relwr_output_names[i]).get());
  }

  // Setup pointers to outputs.
  CHECK_EQ(top.size() - num_hidden_exposed, output_names.size())
      << "OutputBlobNames must provide an output blob name for each top.";
  output_blobs_.resize(output_names.size());
  for (int i = 0; i < output_names.size(); ++i) {
    output_blobs_[i] =
        CHECK_NOTNULL(unrolled_net_->blob_by_name(output_names[i]).get());
  }

  // We should have 2 inputs (x and cont), plus a number of relwrrent inputs,
  // plus maybe a static input.
  CHECK_EQ(2 + num_relwr_blobs + static_input_,
           unrolled_net_->input_blobs().size());

  // This layer's parameters are any parameters in the layers of the unrolled
  // net. We only want one copy of each parameter, so check that the parameter
  // is "owned" by the layer, rather than shared with another.
  this->blobs_.clear();
  for (int i = 0; i < unrolled_net_->params().size(); ++i) {
    if (unrolled_net_->param_owners()[i] == -1) {
      LOG(INFO) << "Adding parameter " << i << ": "
                << unrolled_net_->param_display_names()[i];
      this->blobs_.push_back(unrolled_net_->params()[i]);
    }
  }
  // Check that param_propagate_down is set for all of the parameters in the
  // unrolled net; set param_propagate_down to true in this layer.
  for (int i = 0; i < unrolled_net_->layers().size(); ++i) {
    for (int j = 0; j < unrolled_net_->layers()[i]->blobs().size(); ++j) {
      CHECK(unrolled_net_->layers()[i]->param_propagate_down(j))
          << "param_propagate_down not set for layer " << i << ", param " << j;
    }
  }
  this->param_propagate_down_.clear();
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // Set the diffs of relwrrent outputs to 0 -- we can't backpropagate across
  // batches.
  for (int i = 0; i < relwr_output_blobs_.size(); ++i) {
    caffe_set(relwr_output_blobs_[i]->count(), Ftype(0),
              relwr_output_blobs_[i]->mutable_cpu_diff<Ftype>());
  }

  // Check that the last output_names.size() layers are the pseudo-losses;
  // set last_layer_index so that we don't actually run these layers.
  const vector<string>& layer_names = unrolled_net_->layer_names();
  last_layer_index_ = layer_names.size() - 1 - pseudo_losses.size();
  for (int i = last_layer_index_ + 1, j = 0; i < layer_names.size(); ++i, ++j) {
    CHECK_EQ(layer_names[i], pseudo_losses[j]);
  }
}

template<typename Ftype, typename Btype>
void RelwrrentLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "bottom[0] must have at least 2 axes -- (#timesteps, #streams, ...)";
  CHECK_EQ(T_, bottom[0]->shape(0)) << "input number of timesteps changed";
  N_ = bottom[0]->shape(1);
  CHECK_EQ(bottom[1]->num_axes(), 2)
      << "bottom[1] must have exactly 2 axes -- (#timesteps, #streams)";
  CHECK_EQ(T_, bottom[1]->shape(0));
  CHECK_EQ(N_, bottom[1]->shape(1));
  x_input_blob_->ReshapeLike(*bottom[0]);
  vector<int> cont_shape = bottom[1]->shape();
  cont_input_blob_->Reshape(cont_shape);
  if (static_input_) {
    x_static_input_blob_->ReshapeLike(*bottom[2]);
  }
  vector<BlobShape> relwr_input_shapes;
  RelwrrentInputShapes(&relwr_input_shapes);
  CHECK_EQ(relwr_input_shapes.size(), relwr_input_blobs_.size());
  for (int i = 0; i < relwr_input_shapes.size(); ++i) {
    relwr_input_blobs_[i]->Reshape(relwr_input_shapes[i]);
  }
  unrolled_net_->Reshape();
  x_input_blob_->ShareData(*bottom[0]);
  x_input_blob_->ShareDiff(*bottom[0]);
  cont_input_blob_->ShareData(*bottom[1]);
  if (static_input_) {
    x_static_input_blob_->ShareData(*bottom[2]);
    x_static_input_blob_->ShareDiff(*bottom[2]);
  }
  if (expose_hidden_) {
    const int bottom_offset = 2 + static_input_;
    for (int i = bottom_offset, j = 0; i < bottom.size(); ++i, ++j) {
      CHECK(relwr_input_blobs_[j]->shape() == bottom[i]->shape())
          << "bottom[" << i << "] shape must match hidden state input shape: "
          << relwr_input_blobs_[j]->shape_string();
      relwr_input_blobs_[j]->ShareData(*bottom[i]);
    }
  }
  for (int i = 0; i < output_blobs_.size(); ++i) {
    top[i]->ReshapeLike(*output_blobs_[i]);
    top[i]->ShareData(*output_blobs_[i]);
    top[i]->ShareDiff(*output_blobs_[i]);
  }
  if (expose_hidden_) {
    const int top_offset = output_blobs_.size();
    for (int i = top_offset, j = 0; i < top.size(); ++i, ++j) {
      top[i]->ReshapeLike(*relwr_output_blobs_[j]);
    }
  }
}

template<typename Ftype, typename Btype>
void RelwrrentLayer<Ftype, Btype>::Reset() {
  // "Reset" the hidden state of the net by zeroing out all relwrrent outputs.
  for (int i = 0; i < relwr_output_blobs_.size(); ++i) {
    caffe_set(relwr_output_blobs_[i]->count(), Ftype(0),
              relwr_output_blobs_[i]->mutable_cpu_data<Ftype>());
  }
}

template<typename Ftype, typename Btype>
void RelwrrentLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  // Hacky fix for test time: reshare all the internal shared blobs, which may
  // lwrrently point to a stale owner blob that was dropped when Solver::Test
  // called test_net->ShareTrainedLayersWith(net_.get()).
  // TODO: somehow make this work non-hackily.
  if (this->phase_ == TEST) {
    unrolled_net_->ShareWeights();
  }

  DCHECK_EQ(relwr_input_blobs_.size(), relwr_output_blobs_.size());
  if (!expose_hidden_) {
    for (int i = 0; i < relwr_input_blobs_.size(); ++i) {
      const int count = relwr_input_blobs_[i]->count();
      DCHECK_EQ(count, relwr_output_blobs_[i]->count());
      const Ftype* timestep_T_data = relwr_output_blobs_[i]->cpu_data<Ftype>();
      Ftype* timestep_0_data = relwr_input_blobs_[i]->mutable_cpu_data<Ftype>();
      caffe_copy(count, timestep_T_data, timestep_0_data);
    }
  }

  unrolled_net_->ForwardTo(last_layer_index_);

  if (expose_hidden_) {
    const int top_offset = output_blobs_.size();
    for (int i = top_offset, j = 0; i < top.size(); ++i, ++j) {
      top[i]->ShareData(*relwr_output_blobs_[j]);
    }
  }
}

template<typename Ftype, typename Btype>
void RelwrrentLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  CHECK(!propagate_down[1]) << "Cannot backpropagate to sequence indicators.";

  // TODO: skip backpropagation to inputs and parameters inside the unrolled
  // net according to propagate_down[0] and propagate_down[2]. For now just
  // backprop to inputs and parameters unconditionally, as either the inputs or
  // the parameters do need backward (or Net would have set
  // layer_needs_backward_[i] == false for this layer).
  unrolled_net_->BackwardFromToAu(last_layer_index_, 0, false);
}

INSTANTIATE_CLASS_FB(RelwrrentLayer);

}  // namespace caffe
