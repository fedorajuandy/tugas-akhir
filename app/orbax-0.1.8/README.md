# Orbax

**Orbax is no longer a standalone package, and should not be installed directly.**

[Orbax](https://orbax.readthedocs.io/en/latest/) is a namespace providing common
utility libraries for JAX users.

Please install [orbax-checkpoint](https://pypi.org/project/orbax-checkpoint) or
[orbax-export](https://pypi.org/project/orbax-export) instead, for model
checkpointing and exporting utilities respectively.

## Support

Contact orbax-dev@google.com for help or with any questions about Orbax!

### History

Orbax was initially published as a catch-all package itself. In order to
minimize dependency bloat for users, we have frozen that package at
`orbax-0.1.6`, and will continue to release future changes under the
domain-specific utilities listed above (e.g. `orbax-checkpoint`).

As we have preserved the orbax namespace, existing import statements can remain
unchanged (e.g. `from orbax import checkpoint`).
