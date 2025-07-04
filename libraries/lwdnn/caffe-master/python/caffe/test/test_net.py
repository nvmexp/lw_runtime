import unittest
import tempfile
import os
import numpy as np
import six

import caffe


def simple_net_file(num_output):
    """Make a simple net prototxt, based on test_net.cpp, returning the name
    of the (temporary) file."""

    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write("""name: 'testnet' force_backward: true
    layer { type: 'DummyData' name: 'data' top: 'data' top: 'label'
      dummy_data_param { num: 5 channels: 2 height: 3 width: 4
        num: 5 channels: 1 height: 1 width: 1
        data_filler { type: 'gaussian' std: 1 }
        data_filler { type: 'constant' } } }
    layer { type: 'Colwolution' name: 'colw' bottom: 'data' top: 'colw'
      colwolution_param { num_output: 11 kernel_size: 2 pad: 3
        weight_filler { type: 'gaussian' std: 1 }
        bias_filler { type: 'constant' value: 2 } }
        param { decay_mult: 1 } param { decay_mult: 0 }
        }
    layer { type: 'InnerProduct' name: 'ip' bottom: 'colw' top: 'ip'
      inner_product_param { num_output: """ + str(num_output) + """
        weight_filler { type: 'gaussian' std: 2.5 }
        bias_filler { type: 'constant' value: -3 } } }
    layer { type: 'SoftmaxWithLoss' name: 'loss' bottom: 'ip' bottom: 'label'
      top: 'loss' }""")
    f.close()
    return f.name


class TestNet(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        super(TestNet, self).setUpClass()
        caffe.set_device(0)
        print('TestNet.setUpClass')

    @classmethod
    def tearDownClass(self):
        super(TestNet, self).tearDownClass()
        print('TestNet.tearDownClass')

    def setUp(self):
        self.num_output = 13
        net_file = simple_net_file(self.num_output)
        self.net = caffe.Net(net_file, caffe.TRAIN)
        # fill in valid labels
        self.net.blobs['label'].data[...] = \
                np.random.randint(self.num_output,
                    size=self.net.blobs['label'].data.shape)
        os.remove(net_file)

    def test_memory(self):
        """Check that holding onto blob data beyond the life of a Net is OK"""

        params = sum(map(list, six.itervalues(self.net.params)), [])
        blobs = self.net.blobs.values()
        del self.net

        # now sum everything (forcing all memory to be read)
        total = 0
        for p in params:
            total += p.data.sum() + p.diff.sum()
        for bl in blobs:
            total += bl.data.sum() + bl.diff.sum()

    def test_forward_backward(self):
        self.net.forward()
        self.net.backward()

    def test_inputs_outputs(self):
        self.assertEqual(self.net.inputs, [])
        self.assertEqual(self.net.outputs, ['loss'])

    def test_save_and_read(self):
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        f.close()
        self.net.save(f.name)
        net_file = simple_net_file(self.num_output)
        # Test legacy constructor
        #   should print deprecation warning
        caffe.Net(net_file, f.name, caffe.TRAIN)
        # Test named constructor
        net2 = caffe.Net(net_file, caffe.TRAIN, weights=f.name)
#        net2 = caffe.Net(net_file, f.name, caffe.TRAIN)
        os.remove(net_file)
        os.remove(f.name)
        for name in self.net.params:
            for i in range(len(self.net.params[name])):
                self.assertEqual(abs(self.net.params[name][i].data
                    - net2.params[name][i].data).sum(), 0)

class TestLevels(unittest.TestCase):

    TEST_NET = """
layer {
  name: "data"
  type: "DummyData"
  top: "data"
  dummy_data_param { shape { dim: 1 dim: 1 dim: 10 dim: 10 } }
}
layer {
  name: "NoLevel"
  type: "InnerProduct"
  bottom: "data"
  top: "NoLevel"
  inner_product_param { num_output: 1 }
}
layer {
  name: "Level0Only"
  type: "InnerProduct"
  bottom: "data"
  top: "Level0Only"
  include { min_level: 0 max_level: 0 }
  inner_product_param { num_output: 1 }
}
layer {
  name: "Level1Only"
  type: "InnerProduct"
  bottom: "data"
  top: "Level1Only"
  include { min_level: 1 max_level: 1 }
  inner_product_param { num_output: 1 }
}
layer {
  name: "Level>=0"
  type: "InnerProduct"
  bottom: "data"
  top: "Level>=0"
  include { min_level: 0 }
  inner_product_param { num_output: 1 }
}
layer {
  name: "Level>=1"
  type: "InnerProduct"
  bottom: "data"
  top: "Level>=1"
  include { min_level: 1 }
  inner_product_param { num_output: 1 }
}
"""

    def setUp(self):
        self.f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.f.write(self.TEST_NET)
        self.f.close()

    def tearDown(self):
        os.remove(self.f.name)

    def check_net(self, net, blobs):
        net_blobs = [b for b in net.blobs.keys() if 'data' not in b]
        self.assertEqual(net_blobs, blobs)

    def test_0(self):
        net = caffe.Net(self.f.name, caffe.TEST)
        self.check_net(net, ['NoLevel', 'Level0Only', 'Level>=0'])

    def test_1(self):
        net = caffe.Net(self.f.name, caffe.TEST, level=1)
        self.check_net(net, ['NoLevel', 'Level1Only', 'Level>=0', 'Level>=1'])


class TestStages(unittest.TestCase):

    TEST_NET = """
layer {
  name: "data"
  type: "DummyData"
  top: "data"
  dummy_data_param { shape { dim: 1 dim: 1 dim: 10 dim: 10 } }
}
layer {
  name: "A"
  type: "InnerProduct"
  bottom: "data"
  top: "A"
  include { stage: "A" }
  inner_product_param { num_output: 1 }
}
layer {
  name: "B"
  type: "InnerProduct"
  bottom: "data"
  top: "B"
  include { stage: "B" }
  inner_product_param { num_output: 1 }
}
layer {
  name: "AorB"
  type: "InnerProduct"
  bottom: "data"
  top: "AorB"
  include { stage: "A" }
  include { stage: "B" }
  inner_product_param { num_output: 1 }
}
layer {
  name: "AandB"
  type: "InnerProduct"
  bottom: "data"
  top: "AandB"
  include { stage: "A" stage: "B" }
  inner_product_param { num_output: 1 }
}
"""

    def setUp(self):
        self.f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.f.write(self.TEST_NET)
        self.f.close()

    def tearDown(self):
        os.remove(self.f.name)

    def check_net(self, net, blobs):
        net_blobs = [b for b in net.blobs.keys() if 'data' not in b]
        self.assertEqual(net_blobs, blobs)

    def test_A(self):
        net = caffe.Net(self.f.name, caffe.TEST, stages=['A'])
        self.check_net(net, ['A', 'AorB'])

    def test_B(self):
        net = caffe.Net(self.f.name, caffe.TEST, stages=['B'])
        self.check_net(net, ['B', 'AorB'])

    def test_AandB(self):
        net = caffe.Net(self.f.name, caffe.TEST, stages=['A', 'B'])
        self.check_net(net, ['A', 'B', 'AorB', 'AandB'])


class TestAllInOne(unittest.TestCase):

    TEST_NET = """
layer {
  name: "train_data"
  type: "DummyData"
  top: "data"
  top: "label"
  dummy_data_param {
    shape { dim: 1 dim: 1 dim: 10 dim: 10 }
    shape { dim: 1 dim: 1 dim: 1 dim: 1 }
  }
  include { phase: TRAIN stage: "train" }
}
layer {
  name: "val_data"
  type: "DummyData"
  top: "data"
  top: "label"
  dummy_data_param {
    shape { dim: 1 dim: 1 dim: 10 dim: 10 }
    shape { dim: 1 dim: 1 dim: 1 dim: 1 }
  }
  include { phase: TEST stage: "val" }
}
layer {
  name: "deploy_data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 1 dim: 10 dim: 10 } }
  include { phase: TEST stage: "deploy" }
}
layer {
  name: "ip"
  type: "InnerProduct"
  bottom: "data"
  top: "ip"
  inner_product_param { num_output: 2 }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip"
  bottom: "label"
  top: "loss"
  include: { phase: TRAIN stage: "train" }
  include: { phase: TEST stage: "val" }
}
layer {
  name: "pred"
  type: "Softmax"
  bottom: "ip"
  top: "pred"
  include: { phase: TEST stage: "deploy" }
}
"""

    def setUp(self):
        self.f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.f.write(self.TEST_NET)
        self.f.close()

    def tearDown(self):
        os.remove(self.f.name)

    def check_net(self, net, outputs):
        self.assertEqual(list(net.blobs['data'].shape), [1,1,10,10])
        self.assertEqual(net.outputs, outputs)

    def test_train(self):
        net = caffe.Net(self.f.name, caffe.TRAIN, stages=['train'])
        self.check_net(net, ['loss'])

    def test_val(self):
        net = caffe.Net(self.f.name, caffe.TEST, stages=['val'])
        self.check_net(net, ['loss'])

    def test_deploy(self):
        net = caffe.Net(self.f.name, caffe.TEST, stages=['deploy'])
        self.check_net(net, ['pred'])

# if __name__ == '__main__':
#     unittest.main()