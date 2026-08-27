"""The always-on-top panel in the top-left corner of the screen.

A borderless, topmost Tk window that never takes focus, so the owner keeps
using the computer.  It shows the side to move (with the reason it was
inferred), the five best lines, and the board status.  Two small buttons:
flip the side to move (the one thing a screenshot cannot prove) and stop.

Tk must live on the main thread; the analysis loop hands updates over a
queue and this window polls it.
"""

from __future__ import annotations

import queue
import tkinter as tk
from typing import Any, Callable


class OverlayWindow:
    def __init__(self, updates: "queue.Queue[dict[str, Any]]", *, on_flip: Callable[[], None], on_quit: Callable[[], None],
                 position: tuple[int, int] = (8, 8)) -> None:
        self.updates = updates
        self.on_flip = on_flip
        self.on_quit = on_quit
        self.root = tk.Tk()
        self.root.title("ZEUS Schach Analyse")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", 0.92)
        except tk.TclError:
            pass
        self.root.configure(bg="#0b1220")
        self.root.geometry(f"+{position[0]}+{position[1]}")
        font = ("Segoe UI", 10)
        self.head = tk.Label(self.root, text="Schach Analyse — suche Brett…", fg="#8fd3ff", bg="#0b1220", font=("Segoe UI", 10, "bold"), anchor="w")
        self.head.pack(fill="x", padx=10, pady=(6, 0))
        self.side = tk.Label(self.root, text="", fg="#dce5f0", bg="#0b1220", font=font, anchor="w")
        self.side.pack(fill="x", padx=10)
        self.lines = [tk.Label(self.root, text="", fg="#dce5f0", bg="#0b1220", font=("Consolas", 11), anchor="w") for _ in range(5)]
        for label in self.lines:
            label.pack(fill="x", padx=10)
        self.status = tk.Label(self.root, text="", fg="#6b7c93", bg="#0b1220", font=("Segoe UI", 8), anchor="w")
        self.status.pack(fill="x", padx=10, pady=(0, 4))
        bar = tk.Frame(self.root, bg="#0b1220")
        bar.pack(fill="x", padx=8, pady=(0, 6))
        tk.Button(bar, text="Seite wechseln", command=self.on_flip, bg="#16243a", fg="#dce5f0", relief="flat", font=("Segoe UI", 8)).pack(side="left", padx=2)
        tk.Button(bar, text="Schließen", command=self.on_quit, bg="#3a1620", fg="#dce5f0", relief="flat", font=("Segoe UI", 8)).pack(side="left", padx=2)
        # draggable by the header, so it can be moved out of the way
        self.head.bind("<ButtonPress-1>", self._drag_start)
        self.head.bind("<B1-Motion>", self._drag)
        self._drag_origin = (0, 0)
        self.root.after(150, self._poll)

    def _drag_start(self, event: Any) -> None:
        self._drag_origin = (event.x_root - self.root.winfo_x(), event.y_root - self.root.winfo_y())

    def _drag(self, event: Any) -> None:
        self.root.geometry(f"+{event.x_root - self._drag_origin[0]}+{event.y_root - self._drag_origin[1]}")

    def _poll(self) -> None:
        try:
            while True:
                self.render(self.updates.get_nowait())
        except queue.Empty:
            pass
        self.root.after(150, self._poll)

    def render(self, update: dict[str, Any]) -> None:
        if update.get("quit"):
            self.root.destroy()
            return
        self.head.config(text=update.get("head", "Schach Analyse"))
        self.side.config(text=update.get("side_text", ""))
        lines = update.get("lines", [])
        for index, label in enumerate(self.lines):
            label.config(text=lines[index] if index < len(lines) else "")
        self.status.config(text=update.get("status", ""))

    def run(self) -> None:
        self.root.mainloop()


def format_lines(lines: list[dict[str, Any]]) -> list[str]:
    return [f"{l['rank']}. {l['san']:<7} {l['score']:>6}   {l.get('pv', '')[:34]}" for l in lines]


def side_text(side: str, source: str, orientation: str) -> str:
    who = "Weiß" if side == "w" else "Schwarz"
    return f"{who} am Zug  ·  {source}  ·  du spielst {'Weiß' if orientation == 'white' else 'Schwarz'}"
