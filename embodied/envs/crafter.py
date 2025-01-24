import json
from typing import Any, Dict

import embodied
import numpy as np

import crafter
import crafter.constants as constants
import crafter.objects as objects


class Crafter(embodied.Env):

  PLAYER_ID = 13
  ZOMBIE_ID = 15
  SKELETON_ID = 16
  PRESENCE_FILTER = ['water', 'tree', 'lava', 'coal', 'iron', 'diamond', 'table', 'furnace',
                      crafter.objects.Cow, crafter.objects.Zombie, crafter.objects.Skeleton, crafter.objects.Plant]

  def __init__(self, task, size=(64, 64), logs=False, logdir=None, seed=None):
    assert task in ('reward', 'noreward')
    self._env = crafter.Env(size=size, reward=(task == 'reward'), seed=seed)
    self._logs = logs
    self._logdir = logdir and embodied.Path(logdir)
    self._logdir and self._logdir.mkdir()
    self._episode = 0
    self._length = None
    self._reward = None
    self._achievements = crafter.constants.achievements.copy()
    self._done = True

    # concept supervision
    self.obj2id = self._env._world._mat_ids | self._env._sem_view._obj_ids
    self.id2obj = {v: k for k, v in self.obj2id.items() }
    self.obj_concept_filter = np.array([self.obj2id[concept] for concept in Crafter.PRESENCE_FILTER])
    self.obj_id = np.array([*self.obj2id.values()])
    
    # supervision thresholds
    self.ENTITY_NEAR_THRESHOLD = 8
    self.MATERIAL_PRESENCE_THRESHOLD = 16

  @property
  def obs_space(self):
    spaces = {
        'image': embodied.Space(np.uint8, self._env.observation_space.shape),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
        'log_reward': embodied.Space(np.float32),
        'concepts': embodied.Space(np.float32, (22,), low=0.0, high=1.0)
    }
    if self._logs:
      spaces.update({
          f'log_achievement_{k}': embodied.Space(np.int32)
          for k in self._achievements})
    return spaces

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': embodied.Space(bool),
    }
  
  def _can_make(self) -> None:
    make_info = constants.make
    make_concepts = {}
    for item, info in make_info.items():
        # don't require crafting station to fire concept
        # nearby, _ = self.env._world.nearby(self.env._player.pos, 1)
        # if not all(util in nearby for util in info['nearby']):
        #     make_concepts[item] = 0
        #     continue
        if any(self._env._player.inventory[k] < v for k, v in info['uses'].items()):
            make_concepts[item] = 0
            continue
        make_concepts[item] = 1
    
    labels, concepts = list(zip(*make_concepts.items()))    
    
    return np.array(concepts, dtype=np.float32), labels
  
  @staticmethod
  def manhattan(x :  np.ndarray, y : np.ndarray):
      if x.size == 0:
          return np.inf
      return np.abs(x - y).sum(axis=1).min()
  
  def _is_near_hostile(self) -> None:
      
      # player is id = 13, zombie = 15, skele = 16
      view = self._env._sem_view()
      player_pos = np.argwhere(view == Crafter.PLAYER_ID)[0]
      zombies_pos = np.argwhere(view == Crafter.ZOMBIE_ID)
      skeles_pos = np.argwhere(view == Crafter.SKELETON_ID)
      
      # Same as aggro range
      nearby_zombie, nearby_skele = Crafter.manhattan(zombies_pos, player_pos) <= 3, Crafter.manhattan(skeles_pos, player_pos) <= 3
      
      return np.array([nearby_zombie, nearby_skele], dtype=np.float32)
              
  def _is_near_material(self) -> None:    
      concept_keys = np.array(self.obj_id[self.obj_concept_filter])
      
      view = self._env._sem_view()
      player_pos = np.argwhere(view == Crafter.PLAYER_ID)[0]
      
      concept_presences = np.array([Crafter.manhattan(np.argwhere(view == i), player_pos) for i in concept_keys])
      
      return (concept_presences < self.MATERIAL_PRESENCE_THRESHOLD).astype(np.float32)
      

  def _get_concepts(self) -> Dict[str, np.ndarray]:
      """
      info (dict): {'inventory', 'achievements', 'discount', 'semantic', 'player_pos', 'reward'}
      """
      
      # See presence filter for ids        
      material_presence = self._is_near_material()
      # can make [wood_pick, stone_pick, iron_pick, wood_sword, stone_sword, iron_sword]
      can_make, can_make_labels = self._can_make()
      # near [zombie, skeleton]
      near_hostile = self._is_near_hostile()
      # [0, 9] normalized
      player_health = np.array([self._env._player.health / 9])
      # [0, 25] normalized
      player_hunger = np.array([self._env._player._hunger / 25])
            
      return np.concatenate([material_presence, can_make, near_hostile, player_health, player_hunger], dtype=np.float32)

  def step(self, action):
    if action['reset'] or self._done:
      self._episode += 1
      self._length = 0
      self._reward = 0
      self._done = False
      image = self._env.reset()
      return self._obs(image, 0.0, {}, self._get_concepts(), is_first=True)
    image, reward, self._done, info = self._env.step(action['action'])
    concepts = self._get_concepts()
    self._reward += reward
    self._length += 1
    if self._done and self._logdir:
      self._write_stats(self._length, self._reward, info)
    return self._obs(
        image, reward, info, concepts,
        is_last=self._done,
        is_terminal=info['discount'] == 0)

  def _obs(
      self, image, reward, info, concepts,
      is_first=False, is_last=False, is_terminal=False):
    obs = dict(
        image=image,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        log_reward=np.float32(info['reward'] if info else 0.0),
        concepts=concepts
    )
    if self._logs:
      log_achievements = {
          f'log_achievement_{k}': info['achievements'][k] if info else 0
          for k in self._achievements}
      obs.update({k: np.int32(v) for k, v in log_achievements.items()})
    return obs

  def _write_stats(self, length, reward, info):
    stats = {
        'episode': self._episode,
        'length': length,
        'reward': round(reward, 1),
        **{f'achievement_{k}': v for k, v in info['achievements'].items()},
    }
    filename = self._logdir / 'stats.jsonl'
    lines = filename.read() if filename.exists() else ''
    lines += json.dumps(stats) + '\n'
    filename.write(lines)
    print(f'Wrote stats: {filename}')

  def render(self):
    return self._env.render()
