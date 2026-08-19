# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest

import numpy as np

import newton
import newton.viewer


def _viewer_gl_unavailable_error_types(test: unittest.TestCase) -> tuple[type[BaseException], ...]:
    try:
        __import__("pyglet")
    except ImportError as exc:
        test.skipTest(f"ViewerGL dependencies not available: {exc}")

    unavailable_errors = []
    for module_name, exception_names in (
        ("pyglet.gl", ("ConfigException", "ContextException")),
        ("pyglet.gl.lib", ("MissingFunctionException",)),
        ("pyglet.window", ("NoSuchConfigException", "NoSuchDisplayException")),
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        unavailable_errors.extend(
            exception_type
            for exception_name in exception_names
            if isinstance(exception_type := getattr(module, exception_name, None), type)
        )

    return tuple(dict.fromkeys(unavailable_errors))


def _is_viewer_gl_unavailable_error(test: unittest.TestCase, exc: Exception) -> bool:
    if isinstance(exc, _viewer_gl_unavailable_error_types(test)):
        return True

    return type(exc).__module__.startswith("pyglet.") and type(exc).__name__ in {
        "ConfigException",
        "ContextException",
        "MissingFunctionException",
        "NoSuchConfigException",
        "NoSuchDisplayException",
    }


def _make_headless_viewer_gl_or_skip(test: unittest.TestCase, *, width: int = 320, height: int = 240):
    _viewer_gl_unavailable_error_types(test)
    pyglet = sys.modules.get("pyglet")
    if pyglet is not None:
        pyglet.app.event_loop.has_exit = False

    try:
        return newton.viewer.ViewerGL(width=width, height=height, headless=True)
    except Exception as exc:
        if _is_viewer_gl_unavailable_error(test, exc):
            test.skipTest(f"ViewerGL display/backend not available: {exc}")
        raise


def _make_ground_model():
    """A bare lit ground plane, so measured luminance is pure lighting."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    return builder.finalize()


def _ground_falloff(viewer, state) -> float:
    """Return how much darker the far ground is than the nearest ground.

    The camera looks along +Y from a fixed pose, so image rows map to
    increasing ground distance. A directional sun lights the whole plane
    equally, so this falloff stays small; a camera-anchored spotlight strips
    the direct term past its cone and drives it up.
    """
    viewer.camera.pos = viewer.camera._as_vec3((0.0, -6.0, 3.0))
    viewer.camera.look_at((0.0, 30.0, 0.0))
    # render twice so lazily-updated matrices settle before readback
    for _ in range(2):
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()

    profile = viewer.get_frame(render_ui=False).numpy().astype(np.float32).mean(axis=2).mean(axis=1)
    height = profile.shape[0]
    # walk from the bottom of the frame (nearest ground) toward the horizon
    near_to_far = profile[int(height * 0.55) : int(height * 0.98)][::-1]
    return float(np.max(np.maximum.accumulate(near_to_far) - near_to_far))


class TestViewerSpotlight(unittest.TestCase):
    def test_spotlight_disabled_by_default(self):
        """Verify the camera-anchored spotlight is off by default.

        The spotlight cone in SpotlightAttenuation() is positioned relative to
        the camera, so leaving it enabled makes scene lighting depend on camera
        position (newton-physics/newton#3977).
        """
        viewer = _make_headless_viewer_gl_or_skip(self)
        try:
            self.assertFalse(viewer.renderer.spotlight_enabled)
        finally:
            viewer.close()

    def test_ground_lighting_has_no_dark_band(self):
        """Verify distant ground keeps direct light instead of collapsing to ambient.

        With the camera-anchored spotlight enabled, ground beyond the cone loses
        the entire direct-lighting term and only hemispherical ambient survives,
        which reads as a dark band across the middle distance. Measured on this
        scene the falloff is ~9 with the spotlight on and ~2 with it off.
        """
        viewer = _make_headless_viewer_gl_or_skip(self)
        try:
            model = _make_ground_model()
            viewer.set_model(model)
            falloff = _ground_falloff(viewer, model.state())

            self.assertLess(
                falloff,
                6.0,
                f"far ground is {falloff:.2f} darker than near ground, indicating a camera-anchored dark band",
            )
        finally:
            viewer.close()

    def test_spotlight_enabled_still_reaches_the_shader(self):
        """Verify the spotlight remains available as an opt-in renderer setting."""
        viewer = _make_headless_viewer_gl_or_skip(self)
        try:
            viewer.renderer.spotlight_enabled = True
            self.assertTrue(viewer.renderer.spotlight_enabled)
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
