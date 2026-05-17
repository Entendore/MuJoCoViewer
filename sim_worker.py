"""Background simulation worker — offloads physics stepping to a separate thread.

Architecture:
  - SimWorker runs mj_step in a daemon thread with its own timing loop
  - A threading.RLock protects all access to MjModel / MjData
  - The main (Qt) thread acquires the lock briefly for rendering & UI refresh
  - User interactions (sliders, etc.) acquire the lock for data writes
  - NaN auto-detection pauses the sim and signals the UI
"""

import time
import threading
import numpy as np
import mujoco
from widgets import log


class SimWorker:
    """Runs MuJoCo physics in a background daemon thread."""

    def __init__(self):
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None

        self.lock = threading.RLock()

        # Internal state (accessed under lock)
        self._running = False
        self._paused = True
        self._speed_factor = 1.0
        self._sim_time_accumulator = 0.0
        self._last_real_time = time.time()
        self._step_count = 0
        self._max_steps_per_tick = 50
        self._error: str | None = None
        self._nan_detected = False

        self._thread: threading.Thread | None = None

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="MuJoCoSim"
        )
        self._thread.start()
        log.info("Simulation worker thread started")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                log.warning("Sim thread did not stop within timeout")
            self._thread = None
        log.info("Simulation worker thread stopped")

    # ── Model management ──────────────────────────────────────

    def set_model(self, model: mujoco.MjModel, data: mujoco.MjData):
        with self.lock:
            self.model = model
            self.data = data
            self._step_count = 0
            self._sim_time_accumulator = 0.0
            self._last_real_time = time.time()
            self._error = None
            self._nan_detected = False

    # ── Playback controls ─────────────────────────────────────

    def set_paused(self, paused: bool):
        with self.lock:
            self._paused = paused
            if not paused:
                self._last_real_time = time.time()
                self._sim_time_accumulator = 0.0

    def is_paused(self) -> bool:
        with self.lock:
            return self._paused

    def set_speed(self, factor: float):
        with self.lock:
            self._speed_factor = factor

    # ── Manual stepping (called from main thread) ─────────────

    def step_once(self, n: int = 1):
        with self.lock:
            if self.model is None or self.data is None:
                return
            try:
                for _ in range(n):
                    mujoco.mj_step(self.model, self.data)
                self._step_count += n
                self._check_nan()
            except Exception as e:
                self._error = str(e)
                raise

    def advance_n(self, n: int):
        with self.lock:
            if self.model is None or self.data is None:
                return
            try:
                for _ in range(n):
                    mujoco.mj_step(self.model, self.data)
                self._step_count += n
                mujoco.mj_forward(self.model, self.data)
                self._check_nan()
            except Exception as e:
                self._error = str(e)
                raise

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData):
        with self.lock:
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            self._step_count = 0
            self._sim_time_accumulator = 0.0
            self._last_real_time = time.time()
            self._error = None
            self._nan_detected = False

    # ── NaN detection ─────────────────────────────────────────

    def _check_nan(self):
        """Check for NaN and auto-pause if detected."""
        if self.data is not None and np.any(np.isnan(self.data.qpos)):
            self._nan_detected = True
            self._paused = True
            log.warning("NaN detected in qpos — simulation auto-paused")

    def has_nan(self) -> bool:
        with self.lock:
            return self._nan_detected

    # ── Status queries ────────────────────────────────────────

    def get_step_count(self) -> int:
        with self.lock:
            return self._step_count

    def get_error(self) -> str | None:
        with self.lock:
            err = self._error
            self._error = None
            return err

    # ── Background simulation loop ────────────────────────────

    def _run_loop(self):
        while self._running:
            stepped = 0
            with self.lock:
                if (
                    not self._paused
                    and self.model is not None
                    and self.data is not None
                ):
                    now = time.time()
                    real_dt = min(now - self._last_real_time, 0.1)
                    self._last_real_time = now
                    self._sim_time_accumulator += real_dt * self._speed_factor

                    timestep = self.model.opt.timestep
                    step_count = 0
                    try:
                        while (
                            self._sim_time_accumulator >= timestep
                            and step_count < self._max_steps_per_tick
                        ):
                            mujoco.mj_step(self.model, self.data)
                            self._sim_time_accumulator -= timestep
                            step_count += 1

                            # Periodic NaN check every 50 steps
                            if step_count % 50 == 0:
                                self._check_nan()
                                if self._nan_detected:
                                    break

                    except Exception as e:
                        self._error = str(e)
                        self._paused = True
                        log.error(f"Sim thread error: {e}")
                        return

                    if step_count >= self._max_steps_per_tick:
                        self._sim_time_accumulator = 0.0

                    self._step_count += step_count
                    stepped = step_count

            if stepped > 0:
                time.sleep(0.001)
            else:
                time.sleep(0.005)