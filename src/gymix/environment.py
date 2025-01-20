"""Protocols for reinforcement learning environments and spaces.

These are meant to be compatible with gymnax, but allow users to define environments that are not
subclasses of the gymnax `Environment` class.
"""

import typing

import jax


@typing.runtime_checkable
class Space(typing.Protocol):
    """Protocol for spaces."""

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample from the space."""
        ...

    def contains(self, x: typing.Any) -> bool:
        """Check if the space contains the given value."""
        ...


@typing.runtime_checkable
class Environment(typing.Protocol):
    """Protocol for environments."""

    def reset(self, key: jax.Array, params: typing.Any = None) -> tuple[jax.Array, typing.Any]:
        """Reset the environment.

        Args:
            key: A JAX random key.
            params: Optional parameters for the environment.

        Returns:
            A tuple containing the initial observation and the initial state.
        """
        ...

    def step(
        self,
        key: jax.Array,
        state: typing.Any,
        action: int | float | jax.Array,
        params: typing.Any = None,
    ) -> tuple[jax.Array, typing.Any, jax.Array, jax.Array, dict[typing.Any, typing.Any]]:
        """Take a step in the environment.

        Args:
            key: A JAX random key.
            state: The current state of the environment.
            action: The action to take.
            params: Optional parameters for the environment.

        Returns:
            A tuple containing the next observation, the next state, the reward, the done flag, and
            the info.
        """
        ...

    @property
    def default_params(self) -> typing.Any:
        """Default parameters for the environment."""
        ...

    def action_space(self, params: typing.Any) -> Space:
        """Action space of the environment."""
        ...

    def observation_space(self, params: typing.Any) -> Space:
        """Observation space of the environment."""
        ...
