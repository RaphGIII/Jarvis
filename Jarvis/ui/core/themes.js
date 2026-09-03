/* Visual themes: one accent system, not a random recolor.

   A theme defines the accent pair (every var(--blue)/var(--blue-deep)
   consumer follows), the eye's hue shift, and a particle-intensity factor.
   Applied instantly, persisted as a device-local preference. */

export const THEMES = {
  COSMOS:  { label: "Cosmos — Standard Blau/Cyan",  accent: "#3fa9ff", deep: "#155a92", eyeShift: 0,   particles: 1.0 },
  REACTOR: { label: "Reactor — energetisch",         accent: "#5fd8ff", deep: "#1b7ba8", eyeShift: -12, particles: 1.35 },
  VOID:    { label: "Void — minimal dunkel",         accent: "#7f96b8", deep: "#31415c", eyeShift: 6,   particles: 0.45 },
  NEBULA:  { label: "Nebula — tiefes Violett",       accent: "#a98fff", deep: "#5b3fa8", eyeShift: 62,  particles: 1.05 },
  MONO:    { label: "Mono — technisch reduziert",    accent: "#9fb2c8", deep: "#44536b", eyeShift: 8,   particles: 0.35 },
};

export const INTENSITY = { OFF: 0, LOW: 0.55, NORMAL: 1.0, HIGH: 1.35 };

export function apply(name, eye, intensityName) {
  const theme = THEMES[name] || THEMES.COSMOS;
  const root = document.documentElement;
  root.style.setProperty("--blue", theme.accent);
  root.style.setProperty("--blue-deep", theme.deep);
  root.dataset.theme = name in THEMES ? name : "COSMOS";
  const base = INTENSITY[intensityName] ?? 1.0;
  eye?.setThemeShift?.(theme.eyeShift);
  eye?.setIntensity?.(base * theme.particles);
  document.body.classList.toggle("anim-off", base === 0);
}
