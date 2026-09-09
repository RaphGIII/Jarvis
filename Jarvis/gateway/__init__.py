"""The ZEUS model gateway: one boundary between ZEUS and every model provider.

ZEUS is the product.  The models behind it are replaceable compute providers,
addressed by *role* (``reasoning.free``, ``reasoning.deep``,
``engineer.standard``, ``engineer.frontier``, ``speech.stt``, ``speech.tts``)
and bound to concrete providers in ``config/providers.json`` -- never in code.

Everything that could cost money, leak private context or pretend to be
another assistant passes through :class:`gateway.gateway.ModelGateway`:

    owner input -> chat mode -> privacy router -> model router
        -> budget reservation -> guarded transport -> persona layer

The invariants this package enforces, in order of importance:

* FREE mode makes a paid provider call impossible, not discouraged: the guard
  sits below the router, inside the only HTTP transport a provider can use.
* No paid call without a reservation against the monthly hard cap, and no
  reservation without an estimate.  A refused reservation is a refused call.
* No escalation cascade.  The router picks the cheapest route predicted to be
  reliable enough *before* anything runs; a provider outage is classified as
  an outage, never as evidence that a stronger model is needed.
* Private context never reaches a provider that may use request data for
  product improvement.
* The persona layer keeps the underlying provider out of the product identity.
"""

from gateway.modes import ChatMode

__all__ = ["ChatMode"]
