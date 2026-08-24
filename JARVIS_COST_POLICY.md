# Jarvis Cost Policy

**The rule: Jarvis must never silently create usage-based AI costs.**

This document explains how that is enforced, how to verify it yourself, and what
happens when the optional expert runs out of quota.

---

## How costs are modelled

Not by vendor, by **billing channel**. Two calls to the same company can sit in
different channels — one covered by a flat subscription, one metered per token —
and the difference between them is the entire point.

| Channel | What it is | Default |
|---|---|---|
| `local_model` | Models on your own hardware. Costs electricity. | **allowed** |
| `subscription_cli` | An official CLI authenticated against a subscription you already pay for. No marginal cost. | **allowed** |
| `paid_api` | Metered per-token API billing. | **denied** |
| `usage_credits` | Prepaid or auto-reloading credits. | **denied** |
| `runpod` | Rented GPU compute. | **denied** |
| `browser_ai_automation` | Driving a consumer AI web UI with a browser. | **denied** |

Browser automation of AI chat is denied on principle rather than on cost: it
generally breaches the provider's terms, it is brittle, and it disguises a
metered or prohibited service as a free one.

---

## The three guarantees

**1. A credential is not consent.**
`ANTHROPIC_API_KEY` or `OPENAI_API_KEY` sitting in your environment is a fact
about your machine, never a decision about spending. `CostPolicy.load()` reads
only explicit policy switches; it never infers permission from a key's presence.

**2. Refusal is loud.**
`CostPolicy.require()` raises `CostPolicyViolation` rather than returning
`False`. A caller that ignores a returned `False` spends the money anyway; a
caller that ignores an exception crashes and gets fixed.

**3. Exhaustion is not a licence.**
`fallbacks_for()` cannot name a metered channel — *even under a policy where
metered channels are enabled*. There is deliberately no code path from "quota
spent" to "use the API key". This is the single most dangerous route in a system
like this and it is closed structurally, not by convention.

---

## What happens when the expert runs out

```
EXPERT_UNAVAILABLE
  ├─ do NOT purchase credits
  ├─ do NOT fall back to API billing
  ├─ do NOT start rented compute
  ├─ use the local model if the task allows it
  ├─ checkpoint the work so it can resume
  └─ record why escalation was unavailable
```

The UI shows `EXPERT QUOTA EXHAUSTED`. Work continues locally or waits for the
subscription to reset.

---

## The subscription expert, specifically

The only expert adapter shipped is `experts/claude_code.py`, driving the Claude
Code CLI on your existing subscription. Two safeguards, because one was not
enough:

- **`--bare` is never passed.** Its own help text says Anthropic auth becomes
  "strictly `ANTHROPIC_API_KEY` or …" — a flag that silently switches billing
  models.
- **The child environment is scrubbed** of `ANTHROPIC_API_KEY`,
  `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, the Bedrock and Vertex switches,
  and `OPENAI_API_KEY`. Removing a variable is a far stronger guarantee than
  trusting a flag.

`total_cost_usd` is reported by the CLI even on subscription runs. It is the
notional value of the work, covered by the flat fee. Jarvis records it as a
usage signal and **never presents it as money owed**.

---

## Verify it yourself

```bash
# Every metered channel is off, and the policy says so.
python -c "from runtime.cost_policy import CostPolicy; print(CostPolicy.load().to_dict())"

# The guarantees, as executable tests.
python -m pytest tests/test_cost_policy.py tests/test_expert_gateway.py -q
```

The tests are adversarial: they try to *reach* a metered channel by the routes
that would realistically be taken — a key in the environment, an exhausted
subscription with a paid provider registered and enabled, a permissive policy.

## Turning something on deliberately

If you ever want metered billing, it must be your decision:

```json
// Jarvis/config/cost_policy.json
{ "allow_paid_api": true }
```

or `JARVIS_ALLOW_PAID_API=1`. A corrupt or unparseable config falls back to the
**strict** policy — failing open would be the worst possible default.
