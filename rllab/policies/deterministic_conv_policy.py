import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne.init as LI
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.lasagne_layers import batch_norm, SpatialSoftmaxLayer, WeightLayer
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.policies.base import Policy
from theano import tensor as T
import numpy as np

class DeterministicConvMLPPolicy(Policy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            conv_kernels,
            hidden_sizes,
            hidden_conv_sizes=(),
            conv_nonlinearity=NL.rectify,
            hidden_nonlinearity=NL.rectify,
            conv_W_init=LI.HeUniform(),
            conv_b_init=LI.Constant(0.),
            hidden_W_init=LI.HeUniform(),
            hidden_b_init=LI.Constant(0.),
            output_nonlinearity=NL.tanh,
            output_W_init=LI.Uniform(-3e-3, 3e-3),
            output_b_init=LI.Uniform(-3e-3, 3e-3),
            bn=False,
            pooling=False,
            initializer=None,
            fc_out=True,
            freeze_vision=False,
            fc_initializer=None,
    ):
        Serializable.quick_init(self, locals())

        # observation image stream
        l_img_obs = L.InputLayer(shape=(None,) + env_spec.observation_space["image"].shape, name='img_obs')
        l_img_hidden = l_img_obs
        if bn:
            l_img_hidden = batch_norm(l_img_hidden)

        # first go through several conv layers
        for idx, kernel in enumerate(conv_kernels):
            name = "arg:conv%d" % (idx + 1)
            if initializer:
                W_init = initializer.get(name + "_weight", conv_W_init)
                b_init = initializer.get(name + "_bias", conv_b_init)
            else:
                W_init = conv_W_init
                b_init = conv_b_init

            l_img_hidden = L.Conv2DLayer(
                l_img_hidden,
                num_filters=kernel[0],
                filter_size=kernel[1],
                stride=kernel[2],
                pad=kernel[3],
                nonlinearity=conv_nonlinearity,
                W=W_init,
                b=b_init,
                name=name,
                trainable=(not freeze_vision),
                regularizable=(not freeze_vision)
            )
            if pooling:
                l_img_hidden = L.MaxPool2DLayer(
                    l_img_hidden,
                    pool_size=2)
            if bn:
                l_img_hidden = batch_norm(l_img_hidden)

        # spatial softmax layer
        _, channels, rows, columns = l_img_hidden.output_shape
        l_img_hidden = SpatialSoftmaxLayer(l_img_hidden)

        def expect2Dweight(channels, rows, cols):
            sparse_w = np.zeros((channels * rows * cols, 2 * channels), dtype="float32")
            xy = rows * cols
            w_x = np.tile(np.arange(cols), (rows, 1)).flatten()
            w_x = np.float32(w_x) / (cols - 1) - 0.5
            w_y = np.tile(np.arange(rows), (cols, 1)).flatten('F')
            w_y = np.float32(w_y) / (rows - 1) - 0.5
            for i in range(channels):
                sparse_w[i*xy:(i+1)*xy, 2*i] = w_x
                sparse_w[i*xy:(i+1)*xy, 2*i+1] = w_y
            # sparse_w = sparse_w.transpose()
            return sparse_w

        weight = expect2Dweight(channels, rows, columns)
        # expectation 2D position layer
        l_flat = L.FlattenLayer(l_img_hidden)
        l_flat = WeightLayer(l_flat, weight, channels * 2)

        # then several fc layers
        for idx, size in enumerate(hidden_conv_sizes):
            name = "arg:fc%d" % (idx + 1)
            if initializer:
                W_init = initializer.get(name + "_weight", hidden_W_init)
                b_init = initializer.get(name + "_bias", hidden_b_init)
            else:
                W_init = hidden_W_init
                b_init = hidden_b_init
            l_flat = L.DenseLayer(
                l_flat,
                num_units=size,
                W=W_init,
                b=b_init,
                nonlinearity=hidden_nonlinearity,
                name=name,
                trainable=(not freeze_vision),
                regularizable=(not freeze_vision)
            )
            if bn:
                l_flat = batch_norm(l_flat)

        # fc layer regress to the state representation
        # We can also skip this layer, and directly concatenate the expectd 2d position to the quadrotor states
        if fc_out:
            name = "arg:fc_output"
            if initializer:
                W_init = initializer.get(name + "_weight", hidden_W_init)
                b_init = initializer.get(name + "_bias", hidden_b_init)
            else:
                W_init = hidden_W_init
                b_init = hidden_b_init
            l_flat = L.DenseLayer(
                l_flat,
                num_units=3,
                W=W_init,
                b=b_init,
                nonlinearity=None,
                name=name,
                trainable=(not freeze_vision),
                regularizable=(not freeze_vision)
            )

        # observation state stream
        l_state_obs = L.InputLayer(shape=(None, env_spec.observation_space["state"].flat_dim), name='state_obs')
        l_hidden = L.ConcatLayer([l_flat, l_state_obs])

        for idx, size in enumerate(hidden_sizes):
            name = "h%d" % idx
            if fc_initializer:
                W_init = fc_initializer[name+'.W']
                b_init = fc_initializer[name+'.b']
            else:
                W_init = hidden_W_init
                b_init = hidden_b_init
            l_hidden = L.DenseLayer(
                l_hidden,
                num_units=size,
                W=W_init,
                b=b_init,
                nonlinearity=hidden_nonlinearity,
                name="h%d" % idx
            )
            if bn:
                l_hidden = batch_norm(l_hidden)

        if fc_initializer:
            W_init = fc_initializer['output.W']
            b_init = fc_initializer['output.b']
        else:
            W_init = output_W_init
            b_init = output_b_init
        l_output = L.DenseLayer(
            l_hidden,
            num_units=env_spec.action_space.flat_dim,
            W=W_init,
            b=b_init,
            nonlinearity=output_nonlinearity,
            name="output"
        )

        # Note the deterministic=True argument. It makes sure that when getting
        # actions from single observations, we do not update params in the
        # batch normalization layers
        action_var = L.get_output(l_output, deterministic=True)
        self._output_layer = l_output
        self._img_obs_layer = l_img_obs
        self._state_obs_layer = l_state_obs

        self._f_actions = ext.compile_function([l_img_obs.input_var, l_state_obs.input_var], action_var)

        super(DeterministicConvMLPPolicy, self).__init__(env_spec)
        LasagnePowered.__init__(self, [l_output])

    def get_action(self, observation):
        action = self._f_actions([observation['image']], [observation['state']])[0]
        return action, dict()

    def get_actions(self, observations):
        return self._f_actions(observations['image'], observations['state']), dict()

    def get_action_sym(self, obs_var):
        return L.get_output(
            self._output_layer,
            {self._img_obs_layer: obs_var['image'], self._state_obs_layer: obs_var['state']}
        )
