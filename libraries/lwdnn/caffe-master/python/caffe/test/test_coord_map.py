import unittest

import numpy as np
import random

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.coord_map import coord_map_from_to, crop


def coord_net_spec(ks=3, stride=1, pad=0, pool=2, dstride=2, dpad=0):
    """
    Define net spec for simple colw-pool-decolw pattern common to all
    coordinate mapping tests.
    """
    n = caffe.NetSpec()
    n.data = L.Input(shape=dict(dim=[2, 1, 100, 100]))
    n.aux = L.Input(shape=dict(dim=[2, 1, 20, 20]))
    n.colw = L.Colwolution(
        n.data, num_output=10, kernel_size=ks, stride=stride, pad=pad)
    n.pool = L.Pooling(
        n.colw, pool=P.Pooling.MAX, kernel_size=pool, stride=pool, pad=0)
    # for upsampling kernel size is 2x stride
    try:
        decolw_ks = [s*2 for s in dstride]
    except:
        decolw_ks = dstride*2
    n.decolw = L.Decolwolution(
        n.pool, num_output=10, kernel_size=decolw_ks, stride=dstride, pad=dpad)
    return n


class TestCoordMap(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        super(TestCoordMap, self).setUpClass()
        caffe.set_device(0)
        print('TestCoordMap.setUpClass')

    @classmethod
    def tearDownClass(self):
        super(TestCoordMap, self).tearDownClass()
        print('TestCoordMap.tearDownClass')

    def setUp(self):
        pass

    def test_colw_pool_decolw(self):
        """
        Map through colw, pool, and decolw.
        """
        n = coord_net_spec()
        # identity for 2x pool, 2x decolw
        ax, a, b = coord_map_from_to(n.decolw, n.data)
        self.assertEquals(ax, 1)
        self.assertEquals(a, 1)
        self.assertEquals(b, 0)
        # shift-by-one for 4x pool, 4x decolw
        n = coord_net_spec(pool=4, dstride=4)
        ax, a, b = coord_map_from_to(n.decolw, n.data)
        self.assertEquals(ax, 1)
        self.assertEquals(a, 1)
        self.assertEquals(b, -1)

    def test_pass(self):
        """
        A pass-through layer (ReLU) and colw (1x1, stride 1, pad 0)
        both do identity mapping.
        """
        n = coord_net_spec()
        ax, a, b = coord_map_from_to(n.decolw, n.data)
        n.relu = L.ReLU(n.decolw)
        n.colw1x1 = L.Colwolution(
            n.relu, num_output=10, kernel_size=1, stride=1, pad=0)
        for top in [n.relu, n.colw1x1]:
            ax_pass, a_pass, b_pass = coord_map_from_to(top, n.data)
            self.assertEquals(ax, ax_pass)
            self.assertEquals(a, a_pass)
            self.assertEquals(b, b_pass)

    def test_padding(self):
        """
        Padding colw adds offset while padding decolw subtracts offset.
        """
        n = coord_net_spec()
        ax, a, b = coord_map_from_to(n.decolw, n.data)
        pad = random.randint(0, 10)
        # colw padding
        n = coord_net_spec(pad=pad)
        _, a_pad, b_pad = coord_map_from_to(n.decolw, n.data)
        self.assertEquals(a, a_pad)
        self.assertEquals(b - pad, b_pad)
        # decolw padding
        n = coord_net_spec(dpad=pad)
        _, a_pad, b_pad = coord_map_from_to(n.decolw, n.data)
        self.assertEquals(a, a_pad)
        self.assertEquals(b + pad, b_pad)
        # pad both to cancel out
        n = coord_net_spec(pad=pad, dpad=pad)
        _, a_pad, b_pad = coord_map_from_to(n.decolw, n.data)
        self.assertEquals(a, a_pad)
        self.assertEquals(b, b_pad)

    def test_multi_colw(self):
        """
        Multiple bottoms/tops of a layer are identically mapped.
        """
        n = coord_net_spec()
        # multi bottom/top
        n.colw_data, n.colw_aux = L.Colwolution(
            n.data, n.aux, ntop=2, num_output=10, kernel_size=5, stride=2,
            pad=0)
        ax1, a1, b1 = coord_map_from_to(n.colw_data, n.data)
        ax2, a2, b2 = coord_map_from_to(n.colw_aux, n.aux)
        self.assertEquals(ax1, ax2)
        self.assertEquals(a1, a2)
        self.assertEquals(b1, b2)

    def test_rect(self):
        """
        Anisotropic mapping is equivalent to its isotropic parts.
        """
        n3x3 = coord_net_spec(ks=3, stride=1, pad=0)
        n5x5 = coord_net_spec(ks=5, stride=2, pad=10)
        n3x5 = coord_net_spec(ks=[3, 5], stride=[1, 2], pad=[0, 10])
        ax_3x3, a_3x3, b_3x3 = coord_map_from_to(n3x3.decolw, n3x3.data)
        ax_5x5, a_5x5, b_5x5 = coord_map_from_to(n5x5.decolw, n5x5.data)
        ax_3x5, a_3x5, b_3x5 = coord_map_from_to(n3x5.decolw, n3x5.data)
        self.assertTrue(ax_3x3 == ax_5x5 == ax_3x5)
        self.assertEquals(a_3x3, a_3x5[0])
        self.assertEquals(b_3x3, b_3x5[0])
        self.assertEquals(a_5x5, a_3x5[1])
        self.assertEquals(b_5x5, b_3x5[1])

    def test_nd_colw(self):
        """
        ND colw maps the same way in more dimensions.
        """
        n = caffe.NetSpec()
        # define data with 3 spatial dimensions, otherwise the same net
        n.data = L.Input(shape=dict(dim=[2, 3, 100, 100, 100]))
        n.colw = L.Colwolution(
            n.data, num_output=10, kernel_size=[3, 3, 3], stride=[1, 1, 1],
            pad=[0, 1, 2])
        n.pool = L.Pooling(
            n.colw, pool=P.Pooling.MAX, kernel_size=2, stride=2, pad=0)
        n.decolw = L.Decolwolution(
            n.pool, num_output=10, kernel_size=4, stride=2, pad=0)
        ax, a, b = coord_map_from_to(n.decolw, n.data)
        self.assertEquals(ax, 1)
        self.assertTrue(len(a) == len(b))
        self.assertTrue(np.all(a == 1))
        self.assertEquals(b[0] - 1, b[1])
        self.assertEquals(b[1] - 1, b[2])

    def test_crop_of_crop(self):
        """
        Map coordinates through Crop layer:
        crop an already-cropped output to the input and check change in offset.
        """
        n = coord_net_spec()
        offset = random.randint(0, 10)
        ax, a, b = coord_map_from_to(n.decolw, n.data)
        n.crop = L.Crop(n.decolw, n.data, axis=2, offset=offset)
        ax_crop, a_crop, b_crop = coord_map_from_to(n.crop, n.data)
        self.assertEquals(ax, ax_crop)
        self.assertEquals(a, a_crop)
        self.assertEquals(b + offset, b_crop)

    def test_crop_helper(self):
        """
        Define Crop layer by crop().
        """
        n = coord_net_spec()
        crop(n.decolw, n.data)

    def test_catch_unconnected(self):
        """
        Catch mapping spatially unconnected tops.
        """
        n = coord_net_spec()
        n.ip = L.InnerProduct(n.decolw, num_output=10)
        with self.assertRaises(RuntimeError):
            coord_map_from_to(n.ip, n.data)

    def test_catch_scale_mismatch(self):
        """
        Catch incompatible scales, such as when the top to be cropped
        is mapped to a differently strided reference top.
        """
        n = coord_net_spec(pool=3, dstride=2)  # pool 3x but decolw 2x
        with self.assertRaises(AssertionError):
            crop(n.decolw, n.data)

    def test_catch_negative_crop(self):
        """
        Catch impossible offsets, such as when the top to be cropped
        is mapped to a larger reference top.
        """
        n = coord_net_spec(dpad=10)  # make output smaller than input
        with self.assertRaises(AssertionError):
            crop(n.decolw, n.data)
