import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne.init as LI
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.lasagne_layers import batch_norm, SpatialSoftmaxLayer, WeightLayer
from rllab.core.serializable import Serializable
from rllab.misc import ext
from theano import tensor as T
import numpy as np


class DeterministicConvMLPPolicyWithQFunction(LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            conv_kernels,
            policy_hidden_sizes,
            qf_hidden_sizes,
            conv_nonlinearity=NL.rectify,
            hidden_nonlinearity=NL.rectify,
            conv_W_init=LI.HeUniform(),
            conv_b_init=LI.Constant(0.),
            hidden_W_init=LI.HeUniform(),
            hidden_b_init=LI.Constant(0.),
            policy_output_nonlinearity=NL.tanh,
            qf_output_nonlinearity=None,
            output_W_init=LI.Uniform(-3e-3, 3e-3),
            output_b_init=LI.Uniform(-3e-3, 3e-3),
            action_merge_layer=-2,
            bn=False,
            pooling=False,
            initializer=None,
            fc_out=True,
    ):
        Serializable.quick_init(self, locals())

        # observation image stream
        l_img_obs = L.InputLayer(shape=(None,) + env_spec.observation_space["image"].shape, name='img_obs')
        # observation state stream
        l_state_obs = L.InputLayer(shape=(None, env_spec.observation_space["state"].flat_dim), name='state_obs')
        # action stream
        l_action = L.InputLayer(shape=(None, env_spec.action_space.flat_dim), name="actions")

        l_img_hidden = l_img_obs
        if bn:
            l_img_hidden = batch_norm(l_img_hidden)

        # first go through several conv layers
        for idx, kernel in enumerate(conv_kernels):
            name = "arg:conv%d" % (idx + 1)
            if initializer:
                W_init = initializer[name + "_weight"]
                b_init = initializer[name + "_bias"]
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
                name=name[4::]
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
                sparse_w[i * xy:(i + 1) * xy, 2 * i] = w_x
                sparse_w[i * xy:(i + 1) * xy, 2 * i + 1] = w_y
            # sparse_w = sparse_w.transpose()
            return sparse_w

        weight = expect2Dweight(channels, rows, columns)
        # expectation 2D position layer
        l_flat = L.FlattenLayer(l_img_hidden)
        l_flat = WeightLayer(l_flat, weight, channels * 2)

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
                name=name[4::]
            )

        l_hidden = L.ConcatLayer([l_flat, l_state_obs])
        l_policy = l_hidden
        l_qf = l_hidden
        # now, from here the policy network and the Q function network will have different layers

        # policy network related
        for idx, size in enumerate(policy_hidden_sizes):
            l_policy = L.DenseLayer(
                l_policy,
                num_units=size,
                W=hidden_W_init,
                b=hidden_b_init,
                nonlinearity=hidden_nonlinearity,
                name="policy_h%d" % idx
            )
            if bn:
                l_policy = batch_norm(l_policy)

        l_policy = L.DenseLayer(
            l_policy,
            num_units=env_spec.action_space.flat_dim,
            W=output_W_init,
            b=output_b_init,
            nonlinearity=policy_output_nonlinearity,
            name="policy"
        )

        # Q function network related
        n_layers = len(qf_hidden_sizes) + 1

        if n_layers > 1:
            action_merge_layer = \
                (action_merge_layer % n_layers + n_layers) % n_layers
        else:
            action_merge_layer = 1

        for idx, size in enumerate(qf_hidden_sizes):
            if bn:
                l_qf = batch_norm(l_qf)

            if idx == action_merge_layer:
                l_qf = L.ConcatLayer([l_qf, l_action])

            l_qf = L.DenseLayer(
                l_qf,
                num_units=size,
                W=hidden_W_init,
                b=hidden_b_init,
                nonlinearity=hidden_nonlinearity,
                name="qf_h%d" % (idx + 1)
            )

        if action_merge_layer == n_layers:
            l_qf = L.ConcatLayer([l_qf, l_action])

        l_qf = L.DenseLayer(
            l_qf,
            num_units=1,
            W=output_W_init,
            b=output_b_init,
            nonlinearity=qf_output_nonlinearity,
            name="qf"
        )

        # Note the deterministic=True argument. It makes sure that when getting
        # actions from single observations, we do not update params in the
        # batch normalization layers
        action_var = L.get_output(l_policy, deterministic=True)
        qf_var = L.get_output(l_qf, deterministic=True).flatten()
        self._policy_output_layer = l_policy
        self._qf_output_layer = l_qf
        self._img_obs_layer = l_img_obs
        self._state_obs_layer = l_state_obs
        self._action_layer = l_action

        self._f_actions = ext.compile_function([l_img_obs.input_var, l_state_obs.input_var], action_var)
        self._f_qval = ext.compile_function([l_img_obs.input_var, l_state_obs.input_var, l_action.input_var], qf_var)

        LasagnePowered.__init__(self, [l_policy, l_qf])
        self._env_spec = env_spec

    def get_action(self, observation):
        action = self._f_actions([observation['image']], [observation['state']])[0]
        return action, dict()

    def get_actions(self, observations):
        return self._f_actions(observations['image'], observations['state']), dict()

    def get_action_sym(self, obs_var):
        return L.get_output(
            self._policy_output_layer,
            {self._img_obs_layer: obs_var['image'], self._state_obs_layer: obs_var['state']}
        )

    def reset(self):
        pass

    @property
    def observation_space(self):
        return self._env_spec.observation_space

    @property
    def action_space(self):
        return self._env_spec.action_space

    def get_qval(self, observations, actions):
        return self._f_qval(observations['image'], observations['state'], actions)

    def get_qval_sym(self, obs_var, action_var, **kwargs):
        qvals = L.get_output(
            self._qf_output_layer,
            {self._img_obs_layer: obs_var['image'],
             self._state_obs_layer: obs_var['state'],
             self._action_layer: action_var},
            **kwargs
        )
        return T.reshape(qvals, (-1,))