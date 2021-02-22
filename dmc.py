import numpy as np
from collections import OrderedDict, deque
import warnings

import dm_env
from dm_env import specs
from dm_control import suite
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    from dm_control import manipulation
from dm_control.suite.wrappers import action_scale, pixels

MANIP_PIXELS_KEY = 'front_close'


class FlattenObservationWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        self._obs_spec = OrderedDict()
        wrapped_obs_spec = env.observation_spec().copy()
        dim = 0
        for key in wrapped_obs_spec.keys():
            if key != MANIP_PIXELS_KEY:
                spec = wrapped_obs_spec[key]
                assert spec.dtype == np.float64
                assert type(spec) == specs.Array
                dim += np.prod(spec.shape)

        self._obs_spec['features'] = specs.Array(shape=(dim,),
                                                 dtype=np.float32,
                                                 name='features')

        if MANIP_PIXELS_KEY in wrapped_obs_spec:
            spec = wrapped_obs_spec[MANIP_PIXELS_KEY]
            self._obs_spec['pixels'] = specs.BoundedArray(shape=spec.shape[1:],
                                                          dtype=spec.dtype,
                                                          minimum=spec.minimum,
                                                          maximum=spec.maximum,
                                                          name='pixels')
        self._obs_spec['state'] = specs.Array(
            shape=self._env.physics.get_state().shape,
            dtype=np.float32,
            name='state')

    def _transform_observation(self, time_step):
        obs = OrderedDict()

        features = []
        for key, value in time_step.observation.items():
            if key != MANIP_PIXELS_KEY:
                features.append(value.ravel())
        obs['features'] = np.concatenate(features, axis=0)
        obs['state'] = self._env.physics.get_state().copy()
        if MANIP_PIXELS_KEY in time_step.observation:
            obs['pixels'] = time_step.observation[MANIP_PIXELS_KEY][0]
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, k):
        self._env = env
        self._k = k
        self._frames = deque([], maxlen=k)

        wrapped_obs_spec = env.observation_spec()
        assert 'features' in wrapped_obs_spec
        assert 'pixels' in wrapped_obs_spec

        self._obs_spec = OrderedDict()
        self._obs_spec['features'] = wrapped_obs_spec['features']
        self._obs_spec['state'] = wrapped_obs_spec['state']
        pixels_spec = wrapped_obs_spec['pixels']
        self._obs_spec['pixels'] = specs.BoundedArray(shape=np.concatenate(
            [[pixels_spec.shape[2] * k], pixels_spec.shape[:2]], axis=0),
                                                      dtype=pixels_spec.dtype,
                                                      minimum=0,
                                                      maximum=255,
                                                      name=pixels_spec.name)

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._k
        obs = OrderedDict()
        obs['features'] = time_step.observation['features']
        obs['state'] = time_step.observation['state']
        obs['pixels'] = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        pixels = time_step.observation['pixels'].transpose(2, 0, 1).copy()
        for _ in range(self._k):
            self._frames.append(pixels.copy())
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        self._frames.append(time_step.observation['pixels'].transpose(
            2, 0, 1).copy())
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, amount):
        self._env = env
        self._amount = amount

    def step(self, action):
        reward = 0.0
        for i in range(self._amount):
            time_step = self._env.step(action)
            reward += time_step.reward or 0.0
            if time_step.last():
                break

        return time_step._replace(reward=reward)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


def split_env_name(env_name):
    if env_name == 'ball_in_cup_catch':
        return 'ball_in_cup', 'catch'
    if env_name.startswith('point_mass'):
        return 'point_mass', env_name.split('_')[-1]
    domain = env_name.split('_')[0]
    task = '_'.join(env_name.split('_')[1:])
    return domain, task


def make(env_name, frame_stack, action_repeat, seed):
    domain, task = split_env_name(env_name)

    if domain == 'manip':
        env = manipulation.load(f'{task}_vision', seed=seed)
    else:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)

    # apply action repeat and scaling
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # flatten features
    env = FlattenObservationWrapper(env)

    if domain != 'manip':
        # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
        camera_id = 2 if domain == 'quadruped' else 0
        render_kwargs = {'height': 84, 'width': 84, 'camera_id': camera_id}
        env = pixels.Wrapper(env,
                             pixels_only=False,
                             render_kwargs=render_kwargs)

    env = FrameStackWrapper(env, frame_stack)

    action_spec = env.action_spec()
    assert np.all(action_spec.minimum >= -1.0)
    assert np.all(action_spec.maximum <= +1.0)

    return env
