import os
import tempfile
import unittest

import numpy as np

import caffe
from caffe.proto import caffe_pb2

def create_blob(shape):
    net_file = None
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(
            "input: 'data' input_shape { %s }" % (
                ' '.join(['dim: %d' % i for i in shape])))
        net_file = f.name
    net = caffe.Net(net_file, caffe.TRAIN)
    os.remove(net_file)
    return net.blobs['data']


class TestCreateLayer(unittest.TestCase):

    def setUp(self):
        self.shapei = [2, 2, 4, 4]
        self.blobi = create_blob(self.shapei)
        self.blobo = create_blob([1])

    def test_create_colw_layer(self):
        # Setting layer parameter for colwolution
        layer_param = caffe_pb2.LayerParameter()
        layer_param.type = 'Colwolution'
        layer_param.name = 'colw1'
        cparam = layer_param.colwolution_param
        cparam.num_output = 3
        cparam.kernel_size.append(2)
        wfiller = cparam.weight_filler
        wfiller.type = "uniform"
        wfiller.max = 3
        wfiller.min = 1.5
        # Create layer
        colw_layer = caffe.create_layer(layer_param)
        self.assertEqual(colw_layer.type, 'Colwolution')
        # Set up layer
        colw_layer.SetUp([self.blobi], [self.blobo])
        weights = colw_layer.blobs[0]
        self.assertTrue(np.all(weights.data >= 1.5))
        self.assertTrue(np.all(weights.data <= 3.0))
        # Reshape out blobs
        colw_layer.Reshape([self.blobi], [self.blobo])
        shapei = self.shapei
        shapeo = self.blobo.data.shape
        self.assertEqual(
            shapeo,
            (shapei[0], cparam.num_output,
                shapei[2] - cparam.kernel_size[0] + 1,
                shapei[3] - cparam.kernel_size[0] + 1))
        # Forward, Backward
        colw_layer.Forward([self.blobi], [self.blobo])
        colw_layer.Backward([self.blobo], [True], [self.blobi])
