from rllab.envs.gym_env import RLVREPHierarchyEnv, RLVREPHierarchyTargetEnv

import numpy
import random
import argparse
import time
import logging
import sys
import os
import cv2

def show_coord(im, delay=0, color=(0, 0, 255), win_name='tracking', save_path=None):
    """Plot the ROI bounding box on the image using OpenCV

    Parameters
    ----------
    im : numpy.ndarray
      shape (3, height, width)
    delay
    color
    win_name

    Returns
    -------

    """
    im = im.transpose(1, 2, 0)
    width = im.shape[1]
    height = im.shape[0]
    im2 = numpy.zeros(im.shape)
    im2[:] = im
    win = cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win_name, im2[:, :, ::-1] / 255.0)
    if save_path is not None:
        cv2.imwrite(os.path.join(save_path, win_name+'.jpg'), im2[:, :, ::-1])
    cv2.waitKey(delay)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler('hierarchy.log')
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

# env = RLVREPHierarchyEnv(headless=False, reward_func=0, server_silent=False)
env = RLVREPHierarchyTargetEnv(headless=False, reward_func=11, server_silent=True,
                               state_type="body", random_start=True,
                               obs_type='state',
                               scene_path='/home/sliay/Documents/vrep-uav/scenes/quadcopter_hierarchy_64x64.ttt',
                               action_type='continuous', log=False)
for i_episode in range(10):
    observation = env.reset()
    start_time = time.time()
    rewards = []
    # logging.info('reset obs image:')
    # logging.info(observation['obs_img'])
    # logging.info('reset obs state:')
    # logging.info(observation['obs_state'])
    # show_coord(observation['image'])
    for t in range(500):
        # policy_action = numpy.random.randn(4)
        policy_action = env.action_space.sample()
        res = env.step(policy_action)
        print res
        rewards.append(res.reward)
        print res.observation
        # logging.info('obs image shape:')
        # logging.info(res.observation.shape)
        # logging.info('obs state shape:')
        # logging.info(res.info['obs_state'].shape)
        # show_coord(res.observation['image'])
        if res.done:
            break
    end_time = time.time()
    # print "Episode %d finished after %d timesteps, Rewards:%f" % (i_episode, t+1, numpy.sum(rewards))
    print "FPS/Steps:%f/%d" % ((t+1)/(end_time - start_time), t+1)