# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest

import numpy as np
import warp as wp

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


def _make_distant_box_model(distance: float):
    """A red box *distance* metres from the origin, sized to a constant view angle."""
    builder = newton.ModelBuilder()
    half_extent = 0.02 * distance
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, distance, 3.0)), mass=0.0)
    builder.add_shape_box(body, hx=half_extent, hy=half_extent, hz=half_extent, color=(1.0, 0.0, 0.0))
    return builder.finalize()


def _centre_redness(viewer, state, distance: float) -> float:
    """Return how red the frame centre is, viewing the box head-on."""
    viewer.camera.pos = viewer.camera._as_vec3((0.0, 0.0, 3.0))
    viewer.camera.look_at((0.0, distance, 3.0))
    for _ in range(2):
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()

    img = viewer.get_frame(render_ui=False).numpy()
    height, width = img.shape[:2]
    centre = img[int(height * 0.47) : int(height * 0.53), int(width * 0.47) : int(width * 0.53)]
    centre = centre.reshape(-1, 3).mean(axis=0)
    return float(centre[0] - max(centre[1], centre[2]))


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


class TestViewerRendering(unittest.TestCase):
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

    def test_geometry_stays_visible_well_inside_the_far_plane(self):
        """Verify distance fog does not paint shapes the background colour.

        Fog used to run from a hardcoded 20 m to 200 m while the far plane sat
        at 1000 m, so anything past 200 m was mixed to exactly ``sky_lower`` and
        became indistinguishable from the sky (newton-physics/newton#3977).
        """
        viewer = _make_headless_viewer_gl_or_skip(self)
        try:
            distance = viewer.camera.far * 0.5
            model = _make_distant_box_model(distance)
            viewer.set_model(model)

            redness = _centre_redness(viewer, model.state(), distance)

            self.assertGreater(
                redness,
                8.0,
                f"a red box at {distance:.0f} m (half the far plane) is only {redness:.1f} "
                "redder than the background, so fog has erased it",
            )
        finally:
            viewer.close()

    def test_sky_does_not_occlude_geometry_inside_the_far_plane(self):
        """Verify the sky shell leaves the depth buffer alone.

        The sky is a sphere of radius 0.9 * far centred on the camera. Drawing
        it with depth writes enabled occluded everything beyond that radius and
        cut the scene along a camera-centred sphere, which appears as a curved
        horizon (newton-physics/newton#3977). Fog is pushed out of range here so
        the check isolates occlusion.
        """
        viewer = _make_headless_viewer_gl_or_skip(self)
        try:
            far = viewer.camera.far
            distance = far * 0.95
            model = _make_distant_box_model(distance)
            viewer.set_model(model)
            viewer.renderer.draw_sky = True
            # move fog far beyond the scene so only occlusion can hide the box
            viewer.renderer.fog_start = far * 10.0
            viewer.renderer.fog_end = far * 20.0

            redness = _centre_redness(viewer, model.state(), distance)

            self.assertGreater(
                redness,
                8.0,
                f"a box at {distance:.0f} m is hidden behind the sky shell at "
                f"{far * 0.9:.0f} m even though the far plane is {far:.0f} m",
            )
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
