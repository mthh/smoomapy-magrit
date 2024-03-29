#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import random
from geopandas import GeoDataFrame
from smoomacypy import (
    quick_stewart, SmoothStewart,
    head_tail_breaks, maximal_breaks)
from smoomacypy.helpers_classif import _chain


class TestSmoothStewart(unittest.TestCase):
    def setUp(self):
        pass

    def test_one_shot_stewart(self):
        # Exports correctly to `bytes`:
        res = quick_stewart(
            "misc/nuts3_data.geojson", "pop2008",
            span=65000, beta=2, resolution=80000, nb_class=8,
            mask="misc/nuts3_data.geojson")
        self.assertIsInstance(res, bytes)

        # Exports correctly to `GeoDataFrame`
        # and respects the chosen number of class:
        res = quick_stewart(
            "misc/nuts3_data.geojson", "pop2008",
            span=65000, beta=2, nb_pts=8000, nb_class=8,
            mask="misc/nuts3_data.geojson", output="GeoDataFrame")
        self.assertIsInstance(res, GeoDataFrame)
        self.assertEqual(len(res), 8)

        # Test that it works without specifying without `nb_pts`,
        # `nb_class` and `resolution`:
        res = quick_stewart(
            "misc/nuts3_data.geojson", "pop2008",
            span=65000,
            beta=2,
            mask="misc/nuts3_data.geojson",
            output="GeoDataFrame")
        self.assertIsInstance(res, GeoDataFrame)

        # Test with user defined breaks values :
        my_breaks = [0, 197000, 1295000, 2093000, 3091000,
                     5888000, 10186000, 13500000]
        res = quick_stewart(
            "misc/nuts3_data.geojson",
            "pop2008",
            span=65000,
            beta=2,
            resolution=80000,
            user_defined_breaks=my_breaks,
            mask="misc/nuts3_data.geojson",
            output="GeoDataFrame")
        self.assertIsInstance(res, GeoDataFrame)
        self.assertEqual(len(res), 7)
        # Assert these break values were actually used :
        for wanted_break, obtained_break in zip(my_breaks[1:-1], res["max"][:-1]):
            self.assertAlmostEqual(wanted_break, obtained_break)

        # Test with user defined breaks values
        # (the maximum value is voluntarily low, and the minimum voluntarily high,
        #   two new class will be created,
        #   respectively between the minimum and the first break value
        #   and between the last break value and the maximum)
        my_breaks = [1295000, 2093000, 3091000, 5888000, 10186000]
        nb_interval = len(my_breaks) - 1
        res2 = quick_stewart(
            "misc/nuts3_data.geojson",
            "pop2008",
            span=65000,
            beta=2,
            resolution=80000,
            user_defined_breaks=my_breaks,
            mask="misc/nuts3_data.geojson",
            output="GeoDataFrame")
        self.assertIsInstance(res2, GeoDataFrame)

        # We can test that there is no hole by comparing the area of these polygons
        # and the area of the previously computed result :
        self.assertAlmostEqual(res2.area.sum(), res.area.sum(), 2)
        # And by the fact that there is two extra class compared to our break values :
        self.assertEqual(len(res2), nb_interval + 2)

        # Test with break values non-unique (likely due to the discretization chosen):
        # + Not correctly ordered values
        # They should be reordered and duplicates should be removed ...
        my_breaks = [0, 0, 197000, 1295000, 3091000, 2093000,
                     5888000, 10186000, 13500000]
        res3 = quick_stewart(
            "misc/nuts3_data.geojson",
            "pop2008",
            span=65000,
            beta=2,
            resolution=80000,
            user_defined_breaks=my_breaks,
            mask="misc/nuts3_data.geojson",
            output="GeoDataFrame")
        self.assertIsInstance(res3, GeoDataFrame)
        #  ... so we should have the same class number than `res` :
        self.assertEqual(len(res3), len(res))

    def test_object_stewart(self):
        # Test the OO approach for building smoothed map with stewart potentials
        StePot = SmoothStewart("misc/nuts3_data.geojson", "pop2008",
                               span=65000, beta=2, resolution=90000,
                               mask="misc/nuts3_data.geojson")

        # Test using percentiles :
        result = StePot.render(nb_class=10,
                               disc_func="percentiles",
                               output="geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 10)

        # Test using some already chosen break values :
        my_breaks = [0, 197000, 1295000, 2093000, 3091000,
                     5888000, 10186000, 12000000]
        result = StePot.render(
            nb_class=48,  # bogus values as `nb_class` and
            disc_func="foobar",  # ... disc_func should be overridden
            user_defined_breaks=my_breaks,  # ... by the `user_defined_breaks` params
            output="geodataframe")         # ... and this is what we are testing here

        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 8)
        # Assert these break values were actually used :
        for wanted_break, obtained_break in zip(my_breaks[1:-1], result["max"][:-1]):
            self.assertAlmostEqual(wanted_break, obtained_break)

        # Test again using another discretization method : "head tail breaks"
        # (should define automatically the number of class)
        result = StePot.render(nb_class=None,
                               disc_func="head_tail",
                               output="geodataframe")
        self.assertIsInstance(result, GeoDataFrame)

        # Test that the object has a nice representation :
        a = str(StePot)
        b = repr(StePot)
        self.assertEqual(a, b)
        self.assertIn("SmoothStewart - variable :", a)
        self.assertIn("{} features".format(len(StePot.gdf)), a)

    def test_object_stewart_two_var(self):
        # Test the OO approach with two variables :
        StePot = SmoothStewart("misc/nuts3_data.geojson", "gdppps2008",
                               span=65000, beta=2, resolution=80000,
                               variable_name2="pop2008",
                               mask="misc/nuts3_data.geojson")
        result = StePot.render(8, "equal_interval", output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 8)

    def test_distance_not_geo(self):
        # First with one variable :
        StePot = SmoothStewart("misc/nuts3_data.geojson",
                               "gdppps2008",
                               resolution=100000,
                               span=65000, beta=3,
                               mask="misc/nuts3_data.geojson",
                               distGeo=False)
        result = StePot.render(8, "equal_interval", output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 8)

        # Then with two variables and a custom projection to use :
        StePot = SmoothStewart("misc/nuts3_data.geojson",
                               "gdppps2008",
                               span=65000, beta=2,
                               resolution=80000,
                               variable_name2="pop2008",
                               mask="misc/nuts3_data.geojson",
                               distGeo=False,
                               projDistance="epsg:3035")
        result = StePot.render(8, "equal_interval", output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 8)
        self.assertEqual(result.crs, 'epsg:3035')

    def test_from_gdf_with_new_mask(self):
        gdf = GeoDataFrame.from_file("misc/nuts3_data.geojson")

        # Let's use pareto function for this one :
        StePot = SmoothStewart(gdf, "gdppps2008", typefct="pareto",
                               span=65000, beta=2.33, resolution=80000,
                               mask=None)
        result = StePot.render(6, output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 6)

        # Finally, use a mask (from a file) :
        result = StePot.render(5, output="Geodataframe",
                               new_mask="misc/nuts3_data.geojson")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(StePot.use_mask, True)
        self.assertEqual(len(result), 5)

        # Or from a GeoDataFrame :
        result = StePot.render(6, output="Geodataframe",
                               new_mask=gdf)
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(StePot.use_mask, True)
        self.assertEqual(len(result), 6)

#        # Nope, no mask :
#        result = StePot.render(5, output="Geodataframe",
#                               new_mask=None)
#        self.assertIsInstance(result, GeoDataFrame)
#        self.assertEqual(StePot.use_mask, False)
#        self.assertEqual(len(result), 5)

        # Test that it skips the mask parameter if the layer provided as a mask
        # is not a Polygon/MultiPolygon layer :
        gdf_mask = gdf[1:50].copy()
        gdf_mask.geometry = gdf_mask.geometry.centroid
        result = StePot.render(5, output="Geodataframe",
                               new_mask=gdf_mask)
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(StePot.use_mask, False)
        self.assertEqual(len(result), 5)

    def test_input_with_missing_values(self):
        gdf = GeoDataFrame.from_file("misc/nuts3_data.geojson")
        gdf.loc[12:18, "gdppps2008"] = np.NaN
        StePot = SmoothStewart(gdf, "gdppps2008",
                               span=65000, beta=2, resolution=100000,
                               mask=gdf)
        result = StePot.render(9, "equal_interval", output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 9)

        gdf2 = GeoDataFrame.from_file('misc/nuts3_data.geojson').to_crs("epsg:3035")
        gdf2.loc[:, 'gdppps2008'] = gdf2['gdppps2008'].astype(object)
        gdf2.loc[15:20, 'gdppps2008'] = ""
        gdf2.loc[75:78, 'gdppps2008'] = ""
        StePot = SmoothStewart(gdf2, 'gdppps2008', span=65000, beta=2,
                               resolution=80000, mask=gdf2)
        result = StePot.render(9, 'equal_interval', output="GeoDataFrame")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 9)

    def test_wrong_dtype_missing_values(self):
        gdf = GeoDataFrame.from_file("misc/nuts3_data.geojson")
        gdf.loc[12:18, "gdppps2008"] = np.NaN
        gdf.loc[25:35, "pop2008"] = np.NaN
        gdf.loc[0:len(gdf)-1, "pop2008"] = gdf["pop2008"].astype(str)
        StePot = SmoothStewart(gdf, "gdppps2008",
                               span=65000, beta=2, resolution=100000,
                               mask="misc/nuts3_data.geojson")
        result = StePot.render(9, "equal_interval", output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 9)

        StePot = SmoothStewart(gdf, "gdppps2008", variable_name2="pop2008",
                               span=65000, beta=2, resolution=100000,
                               mask="misc/nuts3_data.geojson")
        result = StePot.render(9, "equal_interval", output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 9)

    def test_from_point_layer_and_maximal_breaks(self):
        gdf = GeoDataFrame.from_file("misc/nuts3_data.geojson").to_crs("epsg:4326")

        # Convert the input layer to a point layer :
        gdf.geometry = gdf.geometry.centroid
        StePot = SmoothStewart(gdf, "gdppps2008",
                               span=65000, beta=2, resolution=80000,
                               mask="misc/nuts3_data.geojson")

        # Use equal interval :
        result = StePot.render(9, "equal_interval", output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 9)

        # Use maximal breaks discretisation method:
        result = StePot.render(9, "maximal_breaks", output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)

    def test_from_polygon_layer_no_crs(self):
        gdf = GeoDataFrame.from_file("misc/nuts3_data.geojson")
        gdf.crs = ''

        # Convert the input layer to a polygon layer (instead of multipolygon):
        gdf.geometry = gdf.geometry.union(gdf.geometry)
        StePot = SmoothStewart(gdf, "gdppps2008",
                               span=65000, beta=2, resolution=100000,
                               mask="misc/nuts3_data.geojson")

        # Use equal interval :
        result = StePot.render(8, "equal_interval", output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 8)

    def test_errors(self):
        # Test with a wrong interaction function name :
        with self.assertRaises(ValueError):
            StePot = SmoothStewart("misc/nuts3_data.geojson", "gdppps2008",
                                   span=65000, beta=2,
                                   typefct="abcdefg")

        StePot = SmoothStewart("misc/nuts3_data.geojson", "gdppps2008",
                               span=65000, beta=2, resolution=90000)

        # Test with a wrong discretization function name :
        with self.assertRaises(ValueError):
            StePot.render(9, "foo", output="Geodataframe")
#
# Actually we are currently lowering the number of points automaticaly
# to fall behind the 'sizelimit':
#        # Test with a sizelimit and a high number of points
#        # (the nuts3 layer contains 1448 features)
#        with self.assertRaises(ValueError):
#            StePot = SmoothStewart(
#                    "misc/nuts3_data.geojson", "gdppps2008",
#                    span=65000, beta=2, typefct='pareto',
#                    nb_pts=100000, sizelimit=10000000)

class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.li = [random.random() * 1000 for i in range(1200)]

    def test_head_tail_breaks(self):
        breaks = head_tail_breaks(self.li)
        self.assertIsInstance(breaks, list)

        breaks2 = head_tail_breaks(self.li, direction="head")
        self.assertIsInstance(breaks, list)
        self.assertAlmostEqual(breaks2, sorted(breaks2))
        self.assertAlmostEqual(breaks, breaks2)

        breaks3 = head_tail_breaks(self.li, direction="tail")
        self.assertIsInstance(breaks, list)
        self.assertAlmostEqual(breaks3, sorted(breaks3))

        with self.assertRaises(ValueError):
            head_tail_breaks(self.li, direction="nope")

    def test_maximal_breaks(self):
        breaks = maximal_breaks(self.li)
        self.assertIsInstance(breaks, list)

        breaks = maximal_breaks(self.li, k=6)
        self.assertIsInstance(breaks, list)
        self.assertEqual(len(breaks), 7)

    def test_chain_list(self):
        _list = [i for i in _chain([789, 45], [78, 96], [7878, 789, 36])]
        self.assertEqual(_list, [789, 45, 78, 96, 7878, 789, 36])


if __name__ == "__main__":
    unittest.main()
