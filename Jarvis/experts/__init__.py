"""Optional stronger intelligence, recruited on Jarvis' terms.

An expert is a provider that can be *asked* to do work Jarvis cannot reliably do
locally.  Three constraints shape everything in this package:

* It is optional.  Jarvis must work with the expert absent, unauthenticated or
  out of quota, and nothing here may become a dependency of ordinary operation.
* It cannot spend money the user did not agree to.  Access is mediated by
  :mod:`runtime.cost_policy`, and quota exhaustion is a state, never a reason to
  find another way to pay.
* It cannot self-certify.  Whatever an expert reports, Jarvis re-runs the
  acceptance checks itself before believing any of it.
"""

from experts.contracts import ExpertJob, ExpertResult, ExpertStatus, QuotaState
from experts.gateway import ExpertGateway, ExpertProvider, ProviderAvailability

__all__ = [
    "ExpertGateway",
    "ExpertJob",
    "ExpertProvider",
    "ExpertResult",
    "ExpertStatus",
    "ProviderAvailability",
    "QuotaState",
]
