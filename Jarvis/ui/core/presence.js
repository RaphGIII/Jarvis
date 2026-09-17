/* ZeusPresence: the one place that decides what the ribbon shows.

   Pages never animate the ribbon themselves.  They emit what happened on the
   bus (the server's state, speech, a receipt, streamed output) and this
   controller turns it into the ribbon's state:

     server state   ->  presence state
     idle, waiting  ->  idle
     listening, transcribing -> listening
     thinking       ->  thinking
     speaking       ->  speaking
     working, coding, researching, verifying -> executing
     error          ->  error
     offline        ->  offline

   plus one-shot pulses: a verified receipt is a success pulse, a failed one an
   error pulse.  Microphone and playback energy reach the ribbon through
   setEnergy (voice/mic.js, voice/playback.js). */

export const PRESENCE_STATES = ["idle", "listening", "thinking", "speaking", "executing", "success", "error", "offline"];

const PRESENCE_OF = {
  idle: "idle", waiting: "idle", listening: "listening", transcribing: "listening", thinking: "thinking", speaking: "speaking",
  working: "executing", coding: "executing", researching: "executing", verifying: "executing", error: "error", offline: "offline",
};

/* The presence state for a server state (unknown states are idle: the ribbon never guesses drama). */
export function presenceFor(serverState) {
  return PRESENCE_OF[String(serverState || "").toLowerCase()] || "idle";
}

/* The pulse a receipt earns: success when verified, error when it failed, none while unverified. */
export function pulseForReceipt(receipt) {
  if (!receipt) return "";
  if (receipt.verified) return "success";
  if (receipt.ok === false) return "error";
  return "";
}

export function attach(eye, bus) {
  let current = "offline";
  bus.on("state", (payload) => {
    const name = String(payload.state || "idle");
    current = presenceFor(name);
    eye.setState(name);
    document.documentElement.dataset.presence = current;
  });
  bus.on("tool", (payload) => {
    if (payload._replay || !payload.receipt) return;
    const pulse = pulseForReceipt(payload.receipt);
    if (pulse) eye.pulseOnce(pulse);
  });
  bus.on("speech", (payload) => {
    if (typeof payload.energy === "number") eye.setEnergy(payload.energy);
  });
  bus.on("jobs:active", (n) => eye.setBackgroundWork?.(Number(n) || 0));
  return { get state() { return current; } };
}
