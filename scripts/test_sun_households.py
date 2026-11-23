import unittest
from build_sun_animation_data import _household_leaders, home_screening
from sun_households import _combined_intervals, summarize_households
from sun_model import facade_probes, household_addresses, model_digest


def example_row(label, intervals, minutes, unit="东户"):
    row = {"楼号":"示例", "住宅楼层":1, "户位":unit, "测点":label, "分期":"一期",
           "计算年份":2026, "模型校验码":"test"}
    for day in ("大寒","冬至","春分"):
        row[day+"时段"] = intervals
        row[day+"分钟"] = minutes
        row[day+"选层筛查分钟"] = minutes
    return row


class HouseholdTimes(unittest.TestCase):
    def test_screening_uses_the_weakest_probe_for_each_home_and_scenario(self):
        fields = ["大寒分钟", "大寒真太阳时分钟", "冬至分钟", "冬至保守分钟",
                  "冬至乐观分钟", "大寒保守分钟", "大寒乐观分钟"]
        values = [[110, 135, 180, 120, 220, 90, 140],
                  [115, 125, 200, 140, 205, 95, 130],
                  [120, 140, 165, 150, 210, 85, 145],
                  [240, 245, 310, 260, 315, 225, 250]]
        records = [dict(zip(fields, map(str, row))) for row in values]
        keys = [("楼", 1, "西户", label) for label in "ABC"] + [("楼", 2, "西户", "A")]
        result = home_screening(records, keys)["楼"]["西户"]
        self.assertEqual(result[1], {"civil":110, "apparent":125, "winter":165,
                                     "winter_high":120, "winter_low":205,
                                     "cold_high":85, "cold_low":130})
        self.assertEqual(result[2]["apparent"], 245)

    def test_legacy_screening_preserves_available_data_without_inventing_apparent_minutes(self):
        rows = [example_row("A", "08:00–10:00", 120), example_row("B", "09:00–10:00", 60)]
        keys = [(r["楼号"], r["住宅楼层"], r["户位"], r["测点"]) for r in rows]
        self.assertEqual(home_screening(rows, keys)["示例"]["东户"][1], {"civil":60, "winter":60})
        self.assertNotIn("大寒真太阳时较弱点分钟", summarize_households(rows)[0])
        for row, minutes in zip(rows, (55, 95)):
            row["大寒真太阳时分钟"] = str(minutes)
        self.assertEqual(summarize_households(rows)[0]["大寒真太阳时较弱点分钟"], 55)

    def test_whole_buildings_are_removed_before_assigning_ranks(self):
        keys = [(bid, floor, "东户", "A") for bid, floor in
                [("12号楼", 1), ("13号楼", 18), ("9号楼", 11), ("10号楼", 1)]]
        excluded = ["12号楼", "13号楼"]
        result = _household_leaders(keys, [99, 98, 70, 70], limit=1, excluded_buildings=excluded)
        self.assertEqual(result["households"], 2)
        self.assertEqual(result["excluded_buildings"], excluded)
        self.assertEqual([(row["bid"], row["rank"]) for row in result["rows"]],
                         [("9号楼", 1), ("10号楼", 1)])

    def test_ranking_scope_keeps_the_existing_sunlight_csv_valid(self):
        base = {"site": {"year": 2026}, "buildings": []}
        scoped = {**base, "site": {**base["site"], "household_ranking_excluded_buildings": ["12号楼"]}}
        self.assertEqual(model_digest(base), model_digest(scoped))
        self.assertNotEqual(model_digest({**base, "buildings": [{"id": "12号楼"}]}), model_digest(scoped))

    def test_full_day_rank_keeps_cutoff_ties_and_weights_each_probe_equally(self):
        keys, seconds = [], []
        for floor, durations in enumerate(((81, 80), (80, 40), (60, 40, 80), (59, 60)), 1):
            for label, duration in zip("ABC", durations):
                keys.append(("示例楼", floor, "东户", label))
                seconds.append(duration)
        ranking = _household_leaders(keys, seconds, limit=2)
        self.assertEqual(ranking["households"], 4)
        self.assertEqual([(r["floor"], r["rank"], r["mean_seconds"]) for r in ranking["rows"]],
                         [(1, 1, 80.5), (2, 2, 60), (3, 2, 60)])
        self.assertEqual(ranking["rows"][2]["probe_seconds"], {"A": 60, "B": 40, "C": 80})

    def test_overlap_is_distinct_from_worst_duration(self):
        a = example_row("A","09:00–11:00",120)
        b = example_row("B","10:00–12:00",120)
        r, = summarize_households([a,b])
        self.assertEqual(r["大寒全部点同时直射分钟"],60)
        self.assertEqual(r["大寒至少一点直射分钟"],180)
        self.assertEqual(r["大寒较弱点分钟"],120)
        self.assertEqual(r["大寒满2小时测点数"],2)

    def test_three_windows_with_gap_and_touching_spans(self):
        a = example_row("A","08:00–09:00 / 10:00–11:00",120)
        b = example_row("B","09:00–10:00 / 10:30–11:30",120)
        c = example_row("C","无",0)
        self.assertEqual(_combined_intervals([a,b],"大寒"),
                         ([(630,660)],[(480,690)]))
        r, = summarize_households([a,b,c])
        self.assertEqual(r["冬至全部点同时直射时段"],"无")
        self.assertEqual(r["冬至较弱测点"],"C")
        self.assertEqual(r["大寒全点均满2小时"],"否")

    def test_partial_or_duplicate_household_rejected(self):
        a = example_row("A","无",0)
        for rows in ([a],[a,a]):
            with self.assertRaises(ValueError):
                summarize_households(rows)


class HouseholdGeometry(unittest.TestCase):
    def setUp(self):
        self.building={"id":"合成矩形","units_per_floor":2,"facade_azimuth":180,
                       "polygon":[[0,0],[40,0],[40,12],[0,12]],
                       "probe_fractions_within_unit":[.25,.75]}

    def test_positions_and_unequal_unit_widths(self):
        result=facade_probes(self.building)
        self.assertEqual([(u,p) for u,p,_,_ in result],
                         [("东户","A"),("东户","B"),("西户","A"),("西户","B")])
        for (_,_,xy,_),x in zip(result,[35,25,15,5]):
            self.assertAlmostEqual(xy[0],x)
            self.assertAlmostEqual(xy[1],-.1)
        b=dict(self.building,unit_boundaries_from_east=[0,.4,1],
               unit_probe_fractions=[[.25,.75],[.1,.5,.9]])
        result=facade_probes(b)
        self.assertEqual([round(p[2][0],1) for p in result],[36,28,21.6,12,2.4])
        for shift in (-1.5,1.5):
            for unit,_,xy,_ in facade_probes(b,frontage_shift=shift):
                self.assertTrue((24<xy[0]<40) if unit=="东户" else (0<xy[0]<24))

    def test_entrances_follow_frontage_order_and_preserve_probe_keys(self):
        expected = [("东单元","东户"), ("东单元","西户"), ("西单元","东户"), ("西单元","西户")]
        wide = dict(self.building, units_per_floor=4)
        mapping = household_addresses(wide)
        self.assertEqual(list(mapping), ["东户","东中户","西中户","西户"])
        probes = facade_probes(wide)
        self.assertEqual([mapping[key] for key, label, _, _ in probes if label == "A"], expected)
        self.assertEqual(household_addresses(self.building),
                         {"东户": ("唯一单元","东户"), "西户": ("唯一单元","西户")})
        six = dict(self.building, units_per_floor=6)
        self.assertEqual(household_addresses(six)["东起3户"], ("","东起3户"))
        self.assertEqual(list(household_addresses(six,"north_six").values()),
                         [expected[0], expected[1], ("中单元","东户"), ("中单元","西户"), *expected[2:]])

    def test_single_probe_override_is_rejected(self):
        with self.assertRaises(ValueError):
            facade_probes(dict(self.building,unit_probe_fractions=[[.5],[.2,.7]]))


if __name__=="__main__":
    unittest.main()
