"""Run with python3 -B scripts/test_sun_engine.py; only NumPy required."""

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from build_sun_animation_data import _daylight_bins, _scene_building, fine_daylight_seconds
from simulate_sunlight import scenario_mesh
from sun_engine import Mesh, apparent_to_civil, lit_intervals, prism_triangles, solar_position, time_bins
from sun_model import DAYS, prisms


BEIJING = timezone(timedelta(hours=8))
# JPL Horizons observer ephemeris. Target 10 (Sun), observer coord@399,
# geodetic east lon 117.22°, lat 39.10°, elevation 0 km, quantity 4,
# APPARENT='AIRLESS'. Time column converted from UT to Beijing (UTC+8).
# https://ssd.jpl.nasa.gov/horizons/manual.html#output
# Values below are azimuth and elevation in degrees.
JPL = [
    ("2026-01-20T08:00", 120.993623, 4.859531),
    ("2026-01-20T12:00", 173.995366, 30.544699),
    ("2026-01-20T16:00", 231.376997, 11.929351),
    ("2026-03-21T08:00", 106.549682, 19.541192),
    ("2026-03-21T12:00", 172.711955, 50.889938),
    ("2026-03-21T16:00", 246.730542, 26.327869),
    ("2026-12-21T09:00", 136.160633, 13.447662),
    ("2026-12-21T12:00", 177.668463, 27.426694),
    ("2026-12-21T15:00", 220.314713, 15.792609),
]


def reference_errors():
    errors = []
    for stamp, azimuth, altitude in JPL:
        sun = solar_position(datetime.fromisoformat(stamp).replace(tzinfo=BEIJING), 39.10, 117.22)
        errors.append([abs((sun.azimuth - azimuth + 180) % 360 - 180), abs(sun.altitude - altitude)])
    return np.array(errors)


class SunTests(unittest.TestCase):
    def test_apparent_window_converts_to_matching_tianjin_civil_instants(self):
        for hour, minute, second in [(8, 21, 59), (16, 22, 5)]:
            apparent = datetime(2026, 1, 20, hour, tzinfo=BEIJING)
            civil = apparent_to_civil(apparent, 39.10, 117.22)
            expected = apparent.replace(minute=minute, second=second)
            self.assertLess(abs((civil - expected).total_seconds()), 1)
        noon = datetime(2026, 1, 20, 12, tzinfo=BEIJING)
        for zone in (BEIJING, timezone.utc):
            # The same local apparent noon refers to one physical instant
            # regardless of which civil offset is used to express the result.
            civil = apparent_to_civil(noon.replace(tzinfo=zone), 39.10, 117.22)
            sun = solar_position(civil, 39.10, 117.22)
            self.assertAlmostEqual(sun.azimuth, 180, places=5)
        with self.assertRaises(ValueError):
            apparent_to_civil(noon.replace(tzinfo=None), 39.10, 117.22)

    def test_daylight_edges_and_standard_sampling_grid(self):
        for label, (month, day, begin, end) in DAYS.items():
            with self.subTest(label):
                midnight = datetime(2026, month, day, tzinfo=BEIJING)
                bins = _daylight_bins(midnight, 39.10, 117.22)
                edges = [a for a, _, _ in bins] + [bins[-1][1]]
                altitudes = [solar_position(t, 39.10, 117.22).altitude for t in edges]
                self.assertAlmostEqual(altitudes[0], -.833, delta=.002)
                self.assertAlmostEqual(altitudes[-1], -.833, delta=.002)
                self.assertEqual(sum(abs(v) < .002 for v in altitudes), 2)
                self.assertTrue(all(a < b <= a + timedelta(minutes=5) for a, b, _ in bins))
                start, stop = midnight + timedelta(hours=begin), midnight + timedelta(hours=end)
                self.assertEqual([b for b in bins if start <= b[0] and b[1] <= stop],
                                 time_bins(start, stop))

    def test_jpl_airless_solar_coordinates(self):
        self.assertTrue(np.all(reference_errors() < .03), reference_errors())

    def test_timezone_and_east_north_up(self):
        clock = datetime(2026, 1, 20, 8, tzinfo=BEIJING)
        local = solar_position(clock, 39.10, 117.22)
        utc = solar_position(clock.astimezone(timezone.utc), 39.10, 117.22)
        np.testing.assert_allclose(utc.direction, local.direction)
        np.testing.assert_allclose(np.linalg.norm(local.direction), 1)
        self.assertTrue(local.direction[0] > 0 and local.direction[1] < 0 and local.direction[2] > 0)
        with self.assertRaises(ValueError):
            solar_position(clock.replace(tzinfo=None), 39.10, 117.22)

    def test_integration_and_clipped_bin(self):
        start = datetime(2026, 1, 20, 8, tzinfo=BEIJING)
        end = start + timedelta(hours=8)
        bins = time_bins(start, end)
        self.assertEqual(len(bins), 96)
        self.assertEqual(bins[0][2], start + timedelta(minutes=2, seconds=30))
        self.assertEqual(sum((stop - begin).total_seconds() for begin, stop, _ in bins), 8*3600)
        self.assertEqual(lit_intervals([True]*96, bins), [(start, end)])
        bins = time_bins(start, start + timedelta(minutes=17))
        self.assertEqual(lit_intervals([True, True, False, True], bins), [
            (start, start+timedelta(minutes=10)), (start+timedelta(minutes=15), start+timedelta(minutes=17))])
        self.assertEqual(lit_intervals([False]*4, bins), [])
        with self.assertRaises(ValueError):
            lit_intervals([True], bins)


def box(xmin, ymin, xmax, ymax):
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


class RayTests(unittest.TestCase):
    def test_minute_ranking_resolves_shadow_transition_inside_coarse_bin(self):
        start = datetime(2026, 1, 20, 12, 20, 20, tzinfo=BEIJING)
        stop = start + timedelta(seconds=290)
        bins = [(start, stop, start + (stop - start) / 2)]
        origins = np.array([[0, 0, 1], [-5, 0, 1]])
        normals = np.array([[0, -1], [0, -1]])
        # The wall east of the first probe blocks it until apparent noon
        # (12:22 civil); the second probe stays west of the entire wall.
        for west_edge, expected in [(0, [190, 290]), (-1, [0, 290])]:
            with self.subTest(west_edge=west_edge):
                mesh = Mesh.from_prisms([("wall", box(west_edge, -3, 10, -2), 0, 20)])
                seconds = fine_daylight_seconds(mesh, origins, normals, bins, 39.10, 117.22)
                np.testing.assert_array_equal(seconds, expected)

    def test_bridge_decks_pier_and_open_clearance(self):
        bridge = {
            "id": "桥", "is_target": False, "group": "external", "kind": "bridge",
            "estimated": True, "floors": 0, "floor_height": 0, "ground_elevation": 1.5,
            "polygon": box(5,-5,7,0), "centerline": [[6,-5],[6,0]],
            "base_height": 5, "roof_height": 7,
            "parts": [{"kind": "deck", "polygon": box(5,0,7,5), "base_height": 5, "top_height": 7,
                       "centerline": [[6,0],[6,5]]},
                      {"kind": "pier", "polygon": box(5.5,2,6.5,3), "base_height": 0, "top_height": 5}],
        }
        model = {"buildings": [bridge]}
        mesh = Mesh.from_prisms(prisms(model))
        # Light going east through a gap, into the pier, into each span, and above.
        origins = [[0,1,2],[0,2.5,2],[0,-1,6.5],[0,1,6.5],[0,1,10]]
        self.assertEqual(mesh.classify(origins, [[1,0]] * 5, [1,0,.1]).tolist(), [-1,0,0,0,-1])
        self.assertIs(scenario_mesh(model, mesh, 1), mesh)
        view = _scene_building(bridge)
        self.assertEqual((view["kind"], view["base"], view["top"]), ("bridge",6.5,8.5))
        self.assertTrue(view["estimated"])
        self.assertEqual([s["kind"] for s in view["parts"]], ["deck","pier"])
        self.assertEqual([s["centerline"] for s in (view, view["parts"][0])],
                         [bridge["centerline"], bridge["parts"][0]["centerline"]])
        self.assertEqual([(s["polygon"], s["base"], s["top"]) for s in (view, *view["parts"])],
                         [(p, lo, hi) for _, p, lo, hi in prisms(model)])

    def test_wall_height_and_nearest_owner(self):
        # At x=3 and x=8 the 45-degree ray has risen 3 m and 8 m.
        mesh = Mesh.from_prisms([("far", box(8, -2, 9, 2), 3, 16),
                                 ("near", box(3, -2, 4, 2), 3, 10)])
        result = mesh.classify([[0,0,6], [0,0,7.5], [0,0,9], [0,0,1]],
                               [[1,0], [1,0], [1,0], [-1,0]], [1,0,1])
        self.assertEqual(result.tolist(), [mesh.ids.index("near"), mesh.ids.index("far"), -1, -2])
        self.assertEqual(mesh.classify([[0,0,1]], [[1,0]], [1,0,-.1])[0], -2)

    def test_concave_roof_notch_and_self_shading(self):
        courtyard = [(0,0), (8,0), (8,8), (6,8), (6,2), (2,2), (2,8), (0,8)]
        for ring in [courtyard, courtyard[::-1]]:
            mesh = Mesh.from_prisms([("own", ring, 1, 10)])
            # Through notch from below base, then through the actual cap.
            result = mesh.classify([[4,5,0], [7,5,0]], [[0,0,1]]*2, [0,0,1])
            self.assertEqual(result.tolist(), [-1,0])
            self.assertEqual(mesh.classify([[2.1,5,2]], [[1,0]], [1,0,.2])[0], 0)
            self.assertEqual(mesh.classify([[4,-.1,2]], [[0,-1]], [0,-1,.2])[0], -1)
            # A ray can pass under an elevated solid.
            self.assertEqual(mesh.classify([[-1,.5,0]], [[1,0]], [1,0,.05])[0], -1)

    def test_polygons_and_empty_scene(self):
        polygon = [(0,0),(1,0),(2,0),(2,1),(0,1),(0,0)]
        self.assertEqual(prism_triangles(polygon, -2, 2).shape, (12,3,3))
        with self.assertRaises(ValueError):
            prism_triangles([(0,0),(2,2),(0,2),(2,0)], 0,2)
        self.assertEqual(Mesh.from_prisms([]).classify([[0,0,0]], [[0,1]], [0,1,1])[0], -1)

    def test_against_analytic_axis_aligned_slabs(self):
        # Independent geometric oracle for arbitrary rays and multiple nearest obstacles.
        rng = np.random.default_rng(2638)
        boxes = np.array([[[-3,-4,1],[1,-2,7]], [[4,-6,-1],[6,3,9]], [[-6,-5.4,5],[.8,1,8]]])
        scene = Mesh.from_prisms([(k, box(*lo[:2], *hi[:2]), lo[2], hi[2])
                                  for k, (lo, hi) in enumerate(boxes)])
        points = rng.uniform([-10,-10,-4],[10,10,12], size=(70,3))
        for ray in rng.uniform([-.9,-.9,.15],[.9,.9,1], size=(30,3)):
            ray /= np.linalg.norm(ray)
            distances = (boxes[None,:,:,:] - points[:,None,None,:]) / ray
            entering = distances.min(axis=2).max(axis=2)
            exiting = distances.max(axis=2).min(axis=2)
            valid = (entering <= exiting) & (exiting > 0)
            first = np.where(valid, np.where(entering > 0, entering, exiting), np.inf)
            expected = np.where(np.isfinite(first).any(axis=1), first.argmin(axis=1), -1)
            actual = scene.classify(points, np.tile(ray, (len(points),1)), ray)
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    max_az, max_alt = reference_errors().max(axis=0)
    print(f"JPL Horizons ({len(JPL)} timestamps), max |error|: az={max_az:.5f}°, alt={max_alt:.5f}°")
    unittest.main()
