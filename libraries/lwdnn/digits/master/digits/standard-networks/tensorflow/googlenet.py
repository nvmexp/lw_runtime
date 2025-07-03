# Preferred settings for this model is:
# Training epochs = 80
# Crop Size = 224
# Learning Rate = 0.001
# Under advanced learning rate options:
# Step Size = 10.0
# Gamma = 0.96

# The auxillary branches as spcified in the original googlenet V1 model do exist in this implementation of
# googlenet but it is not used. To use it, be sure to check self.is_training to ensure that it is only used
# during training.

from model import Tower
from utils import model_property
import tensorflow as tf
import utils as digits


class UserModel(Tower):

    all_inception_settings = {
        '3a': [[64], [96, 128], [16, 32], [32]],
        '3b': [[128], [128, 192], [32, 96], [64]],
        '4a': [[192], [96, 208], [16, 48], [64]],
        '4b': [[160], [112, 224], [24, 64], [64]],
        '4c': [[128], [128, 256], [24, 64], [64]],
        '4d': [[112], [144, 288], [32, 64], [64]],
        '4e': [[256], [160, 320], [32, 128], [128]],
        '5a': [[256], [160, 320], [32, 128], [128]],
        '5b': [[384], [192, 384], [48, 128], [128]]
    }

    @model_property
    def inference(self):
        # rescale to proper form, really we expect 224 x 224 x 1 in HWC form
        model = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        colw_7x7_2s_weight, colw_7x7_2s_bias = self.create_colw_vars([7, 7, self.input_shape[2], 64], 'colw_7x7_2s')
        model = self.colw_layer_with_relu(model, colw_7x7_2s_weight, colw_7x7_2s_bias, 2)

        model = self.max_pool(model, 3, 2)

        # model = tf.nn.local_response_normalization(model)

        colw_1x1_vs_weight, colw_1x1_vs_bias = self.create_colw_vars([1, 1, 64, 64], 'colw_1x1_vs')
        model = self.colw_layer_with_relu(model, colw_1x1_vs_weight, colw_1x1_vs_bias, 1, 'VALID')

        colw_3x3_1s_weight, colw_3x3_1s_bias = self.create_colw_vars([3, 3, 64, 192], 'colw_3x3_1s')
        model = self.colw_layer_with_relu(model, colw_3x3_1s_weight, colw_3x3_1s_bias, 1)

        # model = tf.nn.local_response_normalization(model)

        model = self.max_pool(model, 3, 2)

        inception_settings_3a = InceptionSettings(192, UserModel.all_inception_settings['3a'])
        model = self.inception(model, inception_settings_3a, '3a')

        inception_settings_3b = InceptionSettings(256, UserModel.all_inception_settings['3b'])
        model = self.inception(model, inception_settings_3b, '3b')

        model = self.max_pool(model, 3, 2)

        inception_settings_4a = InceptionSettings(480, UserModel.all_inception_settings['4a'])
        model = self.inception(model, inception_settings_4a, '4a')

        # first auxiliary branch for making training faster
        # aux_branch_1 = self.auxiliary_classifier(model, 512, "aux_1")

        inception_settings_4b = InceptionSettings(512, UserModel.all_inception_settings['4b'])
        model = self.inception(model, inception_settings_4b, '4b')

        inception_settings_4c = InceptionSettings(512, UserModel.all_inception_settings['4c'])
        model = self.inception(model, inception_settings_4c, '4c')

        inception_settings_4d = InceptionSettings(512, UserModel.all_inception_settings['4d'])
        model = self.inception(model, inception_settings_4d, '4d')

        # second auxiliary branch for making training faster
        # aux_branch_2 = self.auxiliary_classifier(model, 528, "aux_2")

        inception_settings_4e = InceptionSettings(528, UserModel.all_inception_settings['4e'])
        model = self.inception(model, inception_settings_4e, '4e')

        model = self.max_pool(model, 3, 2)

        inception_settings_5a = InceptionSettings(832, UserModel.all_inception_settings['5a'])
        model = self.inception(model, inception_settings_5a, '5a')

        inception_settings_5b = InceptionSettings(832, UserModel.all_inception_settings['5b'])
        model = self.inception(model, inception_settings_5b, '5b')

        model = self.avg_pool(model, 7, 1, 'VALID')

        fc_weight, fc_bias = self.create_fc_vars([1024, self.nclasses], 'fc')
        model = self.fully_connect(model, fc_weight, fc_bias)

        # if self.is_training:
        #    return [aux_branch_1, aux_branch_2, model]

        return model

    @model_property
    def loss(self):
        model = self.inference
        loss = digits.classification_loss(model, self.y)
        accuracy = digits.classification_aclwracy(model, self.y)
        self.summaries.append(tf.summary.scalar(accuracy.op.name, accuracy))
        return loss

    def inception(self, model, inception_setting, layer_name):
        weights, biases = self.create_inception_variables(inception_setting, layer_name)
        colw_1x1 = self.colw_layer_with_relu(model, weights['colw_1x1_1'], biases['colw_1x1_1'], 1)

        colw_3x3 = self.colw_layer_with_relu(model, weights['colw_1x1_2'], biases['colw_1x1_2'], 1)
        colw_3x3 = self.colw_layer_with_relu(colw_3x3, weights['colw_3x3'], biases['colw_3x3'], 1)

        colw_5x5 = self.colw_layer_with_relu(model, weights['colw_1x1_3'], biases['colw_1x1_3'], 1)
        colw_5x5 = self.colw_layer_with_relu(colw_5x5, weights['colw_5x5'], biases['colw_5x5'], 1)

        colw_pool = self.max_pool(model, 3, 1)
        colw_pool = self.colw_layer_with_relu(colw_pool, weights['colw_pool'], biases['colw_pool'], 1)

        final_model = tf.concat([colw_1x1, colw_3x3, colw_5x5, colw_pool], 3)

        return final_model

    def create_inception_variables(self, inception_setting, layer_name):
        model_dim = inception_setting.model_dim
        colw_1x1_1_w, colw_1x1_1_b = self.create_colw_vars([1, 1, model_dim, inception_setting.colw_1x1_1_layers],
                                                           layer_name + '-colw_1x1_1')
        colw_1x1_2_w, colw_1x1_2_b = self.create_colw_vars([1, 1, model_dim, inception_setting.colw_1x1_2_layers],
                                                           layer_name + '-colw_1x1_2')
        colw_1x1_3_w, colw_1x1_3_b = self.create_colw_vars([1, 1, model_dim, inception_setting.colw_1x1_3_layers],
                                                           layer_name + '-colw_1x1_3')
        colw_3x3_w, colw_3x3_b = self.create_colw_vars([3, 3, inception_setting.colw_1x1_2_layers,
                                                        inception_setting.colw_3x3_layers],
                                                       layer_name + '-colw_3x3')
        colw_5x5_w, colw_5x5_b = self.create_colw_vars([5, 5, inception_setting.colw_1x1_3_layers,
                                                        inception_setting.colw_5x5_layers],
                                                       layer_name + '-colw_5x5')
        colw_pool_w, colw_pool_b = self.create_colw_vars([1, 1, model_dim, inception_setting.colw_pool_layers],
                                                         layer_name + '-colw_pool')

        weights = {
            'colw_1x1_1': colw_1x1_1_w,
            'colw_1x1_2': colw_1x1_2_w,
            'colw_1x1_3': colw_1x1_3_w,
            'colw_3x3': colw_3x3_w,
            'colw_5x5': colw_5x5_w,
            'colw_pool': colw_pool_w
        }

        biases = {
            'colw_1x1_1': colw_1x1_1_b,
            'colw_1x1_2': colw_1x1_2_b,
            'colw_1x1_3': colw_1x1_3_b,
            'colw_3x3': colw_3x3_b,
            'colw_5x5': colw_5x5_b,
            'colw_pool': colw_pool_b
        }

        return weights, biases

    def auxiliary_classifier(self, model, input_size, name):
        aux_classifier = self.avg_pool(model, 5, 3, 'VALID')

        colw_weight, colw_bias = self.create_colw_vars([1, 1, input_size, input_size], name + '-colw_1x1')
        aux_classifier = self.colw_layer_with_relu(aux_classifier, colw_weight, colw_bias, 1)

        fc_weight, fc_bias = self.create_fc_vars([4*4*input_size, self.nclasses], name + '-fc')
        aux_classifier = self.fully_connect(aux_classifier, fc_weight, fc_bias)

        aux_classifier = tf.nn.dropout(aux_classifier, 0.7)

        return aux_classifier

    def colw_layer_with_relu(self, model, weights, biases, stride_size, padding='SAME'):
        new_model = tf.nn.colw2d(model, weights, strides=[1, stride_size, stride_size, 1], padding=padding)
        new_model = tf.nn.bias_add(new_model, biases)
        new_model = tf.nn.relu(new_model)
        return new_model

    def max_pool(self, model, kernal_size, stride_size, padding='SAME'):
        new_model = tf.nn.max_pool(model, ksize=[1, kernal_size, kernal_size, 1],
                                   strides=[1, stride_size, stride_size, 1], padding=padding)
        return new_model

    def avg_pool(self, model, kernal_size, stride_size, padding='SAME'):
        new_model = tf.nn.avg_pool(model, ksize=[1, kernal_size, kernal_size, 1],
                                   strides=[1, stride_size, stride_size, 1], padding=padding)
        return new_model

    def fully_connect(self, model, weights, biases):
        fc_model = tf.reshape(model, [-1, weights.get_shape().as_list()[0]])
        fc_model = tf.matmul(fc_model, weights)
        fc_model = tf.add(fc_model, biases)
        fc_model = tf.nn.relu(fc_model)
        return fc_model

    def create_colw_vars(self, size, name):
        weight = self.create_weight(size, name + '_W')
        bias = self.create_bias(size[3], name + '_b')
        return weight, bias

    def create_fc_vars(self, size, name):
        weight = self.create_weight(size, name + '_W')
        bias = self.create_bias(size[1], name + '_b')
        return weight, bias

    def create_weight(self, size, name):
        weight = tf.get_variable(name, size, initializer=tf.contrib.layers.xavier_initializer())
        return weight

    def create_bias(self, size, name):
        bias = tf.get_variable(name, [size], initializer=tf.constant_initializer(0.2))
        return bias


class InceptionSettings():

    def __init__(self, model_dim, inception_settings):
        self.model_dim = model_dim
        self.colw_1x1_1_layers = inception_settings[0][0]
        self.colw_1x1_2_layers = inception_settings[1][0]
        self.colw_1x1_3_layers = inception_settings[2][0]
        self.colw_3x3_layers = inception_settings[1][1]
        self.colw_5x5_layers = inception_settings[2][1]
        self.colw_pool_layers = inception_settings[3][0]
