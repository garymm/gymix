# gymix
Gymix is an interface for reinforcement learning environments in JAX.

This library is likely to be useful only for code that wants to write an training loop that is generic over environments.

Gymix uses [protocols](https://typing.readthedocs.io/en/latest/spec/protocol.html) instead of [abstract base classes](https://docs.python.org/3/library/abc.html).

It is meant to be compatible with [Gymnax](https://github.com/RobertTLange/gymnax), but allow users to define environments that are not
subclasses of the gymnax `Environment` class. While the gymnax `Environment` class is basically good, some of the types are overly prescriptive and make it hard to use those enviromnents in libraries that make use of type checking. Gymnax is unmaintained so it is not feasible to fix the types there.
