#ifdef WITH_PYTHON_LAYER
#include <boost/python.hpp>
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <map>
#include <vector>
#include <boost/algorithm/string.hpp>

#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/util/bbox_util.hpp"


using caffe::TBlob;
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::LayerBase;
using caffe::Solver;
using caffe::Timer;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ', '."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(phase, "",
    "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32(level, 0,
    "Optional; network level.");
DEFINE_string(stage, "",
    "Optional; network stages (not to be confused with phase), "
    "separated by ','.");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ', '. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");
DEFINE_string(ap_version, "11point",
    "Average Precision type for object detection");
DEFINE_bool(show_per_class_result, true,
    "Show per class result for object detection");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<std::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    const int count = Caffe::device_count();
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(", "));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// Parse phase from flags
caffe::Phase get_phase_from_flags(caffe::Phase default_value) {
  if (FLAGS_phase == "")
    return default_value;
  if (FLAGS_phase == "TRAIN")
    return caffe::TRAIN;
  if (FLAGS_phase == "TEST")
    return caffe::TEST;
  LOG(FATAL) << "phase must be \"TRAIN\" or \"TEST\"";
  return caffe::TRAIN;  // Avoid warning
}

// Parse stages from flags
vector<string> get_stages_from_flags() {
  vector<string> stages;
  boost::split(stages, FLAGS_stage, boost::is_any_of(","));
  return stages;
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    std::cout << caffe::Caffe::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(", ") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
  return caffe::SolverAction::NONE;
}

// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";
  vector<string> stages = get_stages_from_flags();

  caffe::SolverParameter solver_param = caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver);

  solver_param.mutable_train_state()->set_level(FLAGS_level);
  for (int i = 0; i < stages.size(); i++) {
    solver_param.mutable_train_state()->add_stage(stages[i]);
  }

  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu.size() == 0
      && solver_param.has_solver_mode()
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
      if (solver_param.has_device_id()) {
          FLAGS_gpu = std::to_string(solver_param.device_id());
      } else {  // Set default GPU if unspecified
          FLAGS_gpu = std::to_string(0);
      }
  }

  // Read flags for list of GPUs
  vector<int> gpus;
  get_gpus(&gpus);

  // Set mode and device id[s]
  if (gpus.size() == 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];
    }
    LOG(INFO) << "Using GPUs " << s.str();

    lwdaDeviceProp device_prop;
    for (int i = 0; i < gpus.size(); ++i) {
      lwdaGetDeviceProperties(&device_prop, gpus[i]);
      LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
    }
    Caffe::SetDevice(gpus[0]);
    Caffe::set_gpus(gpus);
    solver_param.set_device_id(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpus.size() * caffe::P2PManager::global_count());
  }

  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

  shared_ptr<caffe::Solver> solver(
      caffe::SolverRegistry::CreateSolver(solver_param, nullptr,
          gpus.size() * caffe::P2PManager::global_rank()));
  solver->SetActionFunction(signal_handler.GetActionFunction());

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  } else if (FLAGS_weights.size()) {
    CopyLayers(solver.get(), FLAGS_weights);
  }

  if (gpus.size() > 1 || caffe::P2PManager::global_count() > 1) {
    caffe::P2PManager p2p_mgr(solver, Caffe::solver_count(), (int)gpus.size(), solver->param());
    p2p_mgr.Run(gpus);
  } else {
    LOG(INFO) << "Starting Optimization";

    solver->Solve();

    if (gpus.size() == 1) {
      std::ostringstream os;
      os.precision(4);
      solver->perf_report(os, gpus[0]);
      LOG(INFO) << os.str();
    }
  }
  if (caffe::P2PManager::global_rank() == 0) {
    LOG(INFO) << "Optimization Done in " << Caffe::time_from_init();
  }
  return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
  vector<string> stages = get_stages_from_flags();

  // Read flags for list of GPUs
  vector<int> gpus;
  get_gpus(&gpus);
  while (gpus.size() > 1) {
    // Only use one GPU
    LOG(INFO) << "Not using GPU #" << gpus.back() << " for single-GPU function";
    gpus.pop_back();
  }
  if (gpus.size() > 0) {
    Caffe::SetDevice(gpus[0]);
  }
  caffe::GPUMemory::Scope gpu_memory_scope(gpus);

  // Set mode and device id
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    lwdaDeviceProp device_prop;
    lwdaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  // Instantiate the caffe net.
  Net caffe_net(FLAGS_model, caffe::TEST, 0U, nullptr, nullptr, false, FLAGS_level, &stages);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob*>& result =
        caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data<float>();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << (loss_weight * mean_score) << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }
  return 0;
}
RegisterBrewFunction(test);


// Test: score a detection model.
int test_detection() {
  typedef float Dtype;
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Read flags for list of GPUs
  vector<int> gpus;
  get_gpus(&gpus);
  while (gpus.size() > 1) {
    // Only use one GPU
    LOG(INFO) << "Not using GPU #" << gpus.back() << " for single-GPU function";
    gpus.pop_back();
  }
  if (gpus.size() > 0) {
    Caffe::SetDevice(gpus[0]);
  }
  caffe::GPUMemory::Scope gpu_memory_scope(gpus);

  // Set mode and device id
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    lwdaDeviceProp device_prop;
    lwdaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  // Instantiate the caffe net.
  Net caffe_net(FLAGS_model, caffe::TEST, 0U);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  std::map<int, std::map<int, vector<std::pair<float, int> > > > all_true_pos;
  std::map<int, std::map<int, vector<std::pair<float, int> > > > all_false_pos;
  std::map<int, std::map<int, int> > all_num_pos;

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob*>& result =
        caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data<float>();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }

    //To compute mAP
    for (int j = 0; j < result.size(); ++j) {
      CHECK_EQ(result[j]->width(), 5);
      const Dtype* result_vec = result[j]->cpu_data<Dtype>();
      int num_det = result[j]->height();
      for (int k = 0; k < num_det; ++k) {
        int item_id = static_cast<int>(result_vec[k * 5]);
        int label = static_cast<int>(result_vec[k * 5 + 1]);
        if (item_id == -1) {
          // Special row of storing number of positives for a label.
          if (all_num_pos[j].find(label) == all_num_pos[j].end()) {
            all_num_pos[j][label] = static_cast<int>(result_vec[k * 5 + 2]);
          } else {
            all_num_pos[j][label] += static_cast<int>(result_vec[k * 5 + 2]);
          }
        } else {
          // Normal row storing detection status.
          float score = result_vec[k * 5 + 2];
          int tp = static_cast<int>(result_vec[k * 5 + 3]);
          int fp = static_cast<int>(result_vec[k * 5 + 4]);
          if (tp == 0 && fp == 0) {
            // Ignore such case. It happens when a detection bbox is matched to
            // a diffilwlt gt bbox and we don't evaluate on diffilwlt gt bbox.
            continue;
          }
          all_true_pos[j][label].push_back(std::make_pair(score, tp));
          all_false_pos[j][label].push_back(std::make_pair(score, fp));
        }
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;

  for (int i = 0; i < test_score.size(); ++i) {
    int test_score_output_id_value = test_score_output_id[i];
    const vector<int>& output_blob_indices = caffe_net.output_blob_indices();
    const vector<string>& blob_names = caffe_net.blob_names();
    const vector<float>& blob_loss_weights = caffe_net.blob_loss_weights();
    if (test_score_output_id_value < output_blob_indices.size()) {
      int blob_index = output_blob_indices[test_score_output_id_value];
      if (blob_index < blob_names.size() && blob_index < blob_loss_weights.size()) {
        const std::string& output_name = blob_names[blob_index];
        const float loss_weight = blob_loss_weights[blob_index];
        std::ostringstream loss_msg_stream;
        const float mean_score = test_score[i] / FLAGS_iterations;
        if (loss_weight) {
          loss_msg_stream << " (* " << loss_weight
                          << " = " << (loss_weight * mean_score) << " loss)";
        }
        LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
      }
    }
  }

  //To compute mAP
  for (int i = 0; i < all_true_pos.size(); ++i) {
    if (all_true_pos.find(i) == all_true_pos.end()) {
      LOG(FATAL) << "Missing output_blob true_pos: " << i;
    }
    const std::map<int, vector<std::pair<float, int> > >& true_pos =
        all_true_pos.find(i)->second;
    if (all_false_pos.find(i) == all_false_pos.end()) {
      LOG(FATAL) << "Missing output_blob false_pos: " << i;
    }
    const std::map<int, vector<std::pair<float, int> > >& false_pos =
        all_false_pos.find(i)->second;
    if (all_num_pos.find(i) == all_num_pos.end()) {
      LOG(FATAL) << "Missing output_blob num_pos: " << i;
    }
    const std::map<int, int>& num_pos = all_num_pos.find(i)->second;
    std::map<int, float> APs;
    float mAP = 0.;
    // Sort true_pos and false_pos with descend scores.
    for (std::map<int, int>::const_iterator it = num_pos.begin();
         it != num_pos.end(); ++it) {
      int label = it->first;
      int label_num_pos = it->second;
      if (true_pos.find(label) == true_pos.end()) {
        LOG(WARNING) << "Missing true_pos for label: " << label;
        continue;
      }
      const vector<std::pair<float, int> >& label_true_pos =
          true_pos.find(label)->second;
      if (false_pos.find(label) == false_pos.end()) {
        LOG(WARNING) << "Missing false_pos for label: " << label;
        continue;
      }
      const vector<std::pair<float, int> >& label_false_pos =
          false_pos.find(label)->second;
      vector<float> prec, rec;
      caffe::ComputeAP(label_true_pos, label_num_pos, label_false_pos,
                FLAGS_ap_version, &prec, &rec, &(APs[label]));
      mAP += APs[label];
      if (FLAGS_show_per_class_result) {
        LOG(INFO) << "class AP " << label << ": " << APs[label];
      }
    }
    mAP /= num_pos.size();
    const int output_blob_index = caffe_net.output_blob_indices()[i];
    const string& output_name = caffe_net.blob_names()[output_blob_index];
    LOG(INFO) << "Test net output mAP #" << i << ": " << output_name << " = " << mAP;
  }
  return 0;
}
RegisterBrewFunction(test_detection);

void print_iters(int j, int iters, Timer& iter_timer) {
  if (j > 0 && (j + 1) % iters == 0) {
    LOG(INFO) << "Iterations: " << (j - iters + 1) << "-" << (j + 1)
              << " average forward-backward time: "
              << iter_timer.MicroSeconds() / (iters * 1000.F) << " ms.";
    iter_timer.Start();
  }
}

// Time: benchmark the exelwtion time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";
  caffe::Phase phase = get_phase_from_flags(caffe::TRAIN);
  vector<string> stages = get_stages_from_flags();

  vector<int> gpus;
  // Read flags for list of GPUs
  get_gpus(&gpus);
  while (gpus.size() > 1) {
    // Only use one GPU
    LOG(INFO) << "Not using GPU #" << gpus.back() << " for single-GPU function";
    gpus.pop_back();
  }
  if (gpus.size() > 0) {
    Caffe::SetDevice(gpus[0]);
  }
  caffe::GPUMemory::Scope gpu_memory_scope(gpus);
  // Set mode and device_id
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    lwdaDeviceProp device_prop;
    lwdaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU " << gpus[0] << ": " << device_prop.name;
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  const int kInitIterations = 5;

  caffe::SolverParameter solver_param;
  caffe::ReadNetParamsFromTextFileOrDie(FLAGS_model, solver_param.mutable_net_param());
  solver_param.set_max_iter(kInitIterations + FLAGS_iterations);
  solver_param.set_lr_policy("fixed");
  solver_param.set_snapshot_after_train(false);
  solver_param.set_base_lr(0.01F);
  solver_param.set_random_seed(1371LL);
  solver_param.set_test_interval(kInitIterations + FLAGS_iterations + 1);
  solver_param.set_display(0);

  solver_param.mutable_train_state()->set_level(FLAGS_level);
  for (int i = 0; i < stages.size(); i++) {
    solver_param.mutable_train_state()->add_stage(stages[i]);
  }
  solver_param.mutable_net_param()->mutable_state()->set_phase(phase);

  shared_ptr<Solver> solver(caffe::SolverRegistry::CreateSolver(solver_param));
  shared_ptr<Net> caffe_net = solver->net();

  // Do a number of clean forward and backward pass,
  // so that memory allocation are done,
  // and future iterations will be more stable.
  Timer init_timer(true);
  Timer forward_timer(true);
  Timer backward_timer(true);
  double forward_time = 0.0;
  double backward_time = 0.0;
  LOG(INFO) << "Initialization for " << kInitIterations << " iterations.";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  LOG(INFO) << "Performing initial Forward/Backward";
  const vector<shared_ptr<LayerBase> >& layers = caffe_net->layers();
  const vector<vector<Blob*> >& bottom_vecs = caffe_net->bottom_vecs();
  const vector<vector<Blob*> >& top_vecs = caffe_net->top_vecs();
  const vector<vector<bool> >& bottom_need_backward = caffe_net->bottom_need_backward();
  init_timer.Start();
  solver->Step(kInitIterations);
  double init_time = init_timer.MilliSeconds();
  LOG(INFO) << "Initial Forward/Backward complete";
  LOG(INFO) << "Average Initialization Forward/Backward pass: "
            << init_time / kInitIterations << " ms.";

  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer(true);
  total_timer.Start();
  Timer timer(true);
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  forward_time = 0.0;
  backward_time = 0.0;
  Timer iter_timer(true);
  iter_timer.Start();
  for (int j = 0; j < FLAGS_iterations; ++j) {
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    if (FLAGS_iterations >= 50000) {
      print_iters(j, 10000, iter_timer);
    } else if (FLAGS_iterations >= 10000) {
      print_iters(j, 1000, iter_timer);
    } else if (FLAGS_iterations >= 1000) {
      print_iters(j, 100, iter_timer);
    } else if (FLAGS_iterations >= 100) {
      print_iters(j, 10, iter_timer);
    } else {
      LOG(INFO) << "Iteration: " << (j + 1) << " forward-backward time: "
                << iter_timer.MicroSeconds() / 1000.F << " ms.";
      iter_timer.Start();
    }
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const std::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000. /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000. /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000. /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000. /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";

  std::string stats = layers.back()->print_stats();  // TODO all layers
  if (!stats.empty()) {
    LOG(INFO) << "*** Stats ***";
    LOG(INFO) << stats;
  }

  return 0;
}
RegisterBrewFunction(time);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model exelwtion time");

  std::ostringstream os;
  os << std::endl;
  for (int n = 0; n < argc; ++n) {
    os << "[" << n << "]: " << argv[n] << std::endl;
  }
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);

  vector<int> gpus;
  get_gpus(&gpus);
  Caffe::SetDevice(gpus.size() > 0 ? gpus[0] : 0);
  Caffe::set_gpus(gpus);

  LOG(INFO) << "This is LWCaffe " << Caffe::caffe_version()
            << " started at " << Caffe::start_time();
  LOG(INFO) << "LwDNN version: " << Caffe::lwdnn_version();
  LOG(INFO) << "LwBLAS version: " << Caffe::lwblas_version();
  LOG(INFO) << "LWCA version: " << Caffe::lwda_version();
  LOG(INFO) << "LWCA driver version: " << Caffe::lwda_driver_version();
  LOG(INFO) << "Arguments: " << os.str();

  if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
    try {
      Py_InitializeEx(0);
      if (!PyEval_ThreadsInitialized()) {
        PyEval_InitThreads();
        static PyThreadState* mainPyThread = PyEval_SaveThread();
        (void)mainPyThread;
      }
#endif
      return GetBrewFunction(std::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set&) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
