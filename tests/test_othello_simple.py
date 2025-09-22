import jax
import jax.numpy as jnp
import pytest
from jaxatari.games.jax_othello import JaxOthello
from jaxatari.wrappers import FlattenObservationWrapper
import jaxatari.spaces as spaces
import numpy as np

def test_othello_flatten_observation():
    key = jax.random.PRNGKey(0)
    
    # 1. Environment instanziieren
    env = JaxOthello()
    
    # 2. Wrap mit FlattenObservationWrapper
    flat_env = FlattenObservationWrapper(env)
    
    # 3. Originale und flatten Observation-Space
    orig_space = env.observation_space()
    flat_space = flat_env.observation_space()
    
    # 4. Prüfe, dass Dict keys gleich sind
    assert orig_space.spaces.keys() == flat_space.spaces.keys()
    
    # 5. Observation generieren
    obs, _ = flat_env.reset(key)
    
    # 6. Prüfe, dass jedes Leaf 1D ist und korrekt enthalten
    def check_flat(obs_leaf, space_leaf):
        assert space_leaf.contains(obs_leaf), f"{obs_leaf.shape} not in {space_leaf.shape}"
        assert obs_leaf.ndim == 1
    
    for obs_leaf, space_leaf in zip(jax.tree_leaves(obs), jax.tree_leaves(flat_space)):
        check_flat(obs_leaf, space_leaf)


if __name__ == "__main__":
    pytest.main([__file__])
