import os
import tempfile
import unittest
from datetime import date

os.environ["FAKE_TODAY"] = "2026-10-12"

from agent import tools as T                      # noqa: E402
from agent.llm import extract_json                # noqa: E402
from agent.memory import Memory                   # noqa: E402
from agent.rag import KnowledgeBase, chunk_text   # noqa: E402

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "data", "sample_notes")


class TestGrades(unittest.TestCase):
    def test_cgpa_10_point(self):
        r = T.calculate_cgpa([{"name": "DBMS", "credits": 4, "grade": "A"},
                              {"name": "OS", "credits": 3, "grade": "O"},
                              {"name": "ML", "credits": 3, "grade": "B+"}])
        self.assertTrue(r.ok)
        self.assertEqual(r.data["gpa"], 8.3)          # (4*8 + 3*10 + 3*7) / 10

    def test_cgpa_rejects_bad_grade(self):
        self.assertFalse(T.calculate_cgpa([{"name": "X", "credits": 3, "grade": "Z"}]).ok)

    def test_cgpa_accepts_json_string(self):
        r = T.calculate_cgpa('[{"name":"A","credits":2,"grade":"9"}]')
        self.assertTrue(r.ok)
        self.assertEqual(r.data["gpa"], 9.0)

    def test_required_gpa(self):
        r = T.required_gpa(7.5, 80, 20, 8.0)          # (8*100 - 7.5*80) / 20 = 10.0
        self.assertEqual(r.data["needed"], 10.0)
        self.assertIn("reach", r.text)
        self.assertIn("NOT reachable", T.required_gpa(7.0, 80, 20, 9.0).text)


class TestStudyPlan(unittest.TestCase):
    def setUp(self):
        self.r = T.make_study_plan(["Normalization", {"name": "Transactions", "difficulty": 5}, "Indexing"],
                                   "2026-10-30", hours_per_day=3)
        self.sessions = self.r.data

    def test_ok_and_dates_in_range(self):
        self.assertTrue(self.r.ok)
        for s in self.sessions:
            d = date.fromisoformat(s["date"])
            self.assertGreaterEqual(d, date(2026, 10, 12))
            self.assertLess(d, date(2026, 10, 30))        # nothing on exam day

    def test_daily_hours_never_exceed_limit(self):
        per_day = {}
        for s in self.sessions:
            per_day[s["date"]] = per_day.get(s["date"], 0) + s["hours"]
        self.assertTrue(all(h <= 3.02 for h in per_day.values()))

    def test_harder_topic_gets_more_time_and_revision_is_last(self):
        study = {}
        for s in self.sessions:
            if s["type"] == "study":
                study[s["topic"]] = study.get(s["topic"], 0) + s["hours"]
        self.assertGreater(study["Transactions"], study["Indexing"])
        self.assertEqual(self.sessions[-1]["type"], "revision")
        self.assertIn("mock test", self.sessions[-1]["title"])

    def test_errors(self):
        self.assertFalse(T.make_study_plan([], "2026-10-30").ok)
        self.assertFalse(T.make_study_plan(["A"], "2026-10-01").ok)     # exam in the past
        self.assertFalse(T.make_study_plan(["A"], "not-a-date").ok)

    def test_short_timeline_is_all_revision(self):
        r = T.make_study_plan(["A", "B"], "2026-10-14")
        self.assertTrue(r.ok)
        self.assertTrue(all(s["type"] == "revision" for s in r.data))


class TestCalendarAndMemory(unittest.TestCase):
    def test_ics_file_created_from_plan(self):
        plan = T.make_study_plan(["A", "B"], "2026-10-25").data
        with tempfile.TemporaryDirectory() as d:
            r = T.create_calendar_file(plan, "my plan.ics", outdir=d)
            self.assertTrue(r.ok)
            text = open(r.files[0], encoding="utf-8").read()
            self.assertIn("BEGIN:VCALENDAR", text)
            self.assertEqual(text.count("BEGIN:VEVENT"), len(plan))
            self.assertTrue(os.path.basename(r.files[0]).endswith(".ics"))

    def test_ics_rejects_bad_event(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(T.create_calendar_file([{"title": "x", "date": "garbage"}], outdir=d).ok)

    def test_ics_filename_cannot_escape_folder(self):
        with tempfile.TemporaryDirectory() as d:
            r = T.create_calendar_file([{"title": "x", "date": "2026-10-20"}], "../../evil.ics", outdir=d)
            self.assertEqual(os.path.dirname(r.files[0]), d)

    def test_deadlines_persist_and_sort(self):
        with tempfile.TemporaryDirectory() as d:
            m = Memory(os.path.join(d, "m.json"))
            m.add_deadline("Lab report", date(2026, 10, 20), "OS")
            m.add_deadline("DBMS exam", date(2026, 10, 15))
            m.add_deadline("Old", date(2026, 9, 1))
            items = Memory(os.path.join(d, "m.json")).list_deadlines(date(2026, 10, 12), 30)
            self.assertEqual([i["title"] for i in items], ["DBMS exam", "Lab report"])
            self.assertEqual(items[0]["days_left"], 3)

    def test_parse_date_variants(self):
        self.assertEqual(T.parse_date("20/11/2026"), date(2026, 11, 20))
        self.assertEqual(T.parse_date("tomorrow"), date(2026, 10, 13))
        self.assertEqual(T.parse_date("in 3 days"), date(2026, 10, 15))


class TestRAG(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.kb = KnowledgeBase(self.tmp.name, use_dense=False)
        self.kb.add_folder(SAMPLES)

    def tearDown(self):
        self.tmp.cleanup()

    def test_indexed_three_sources(self):
        self.assertEqual(len(self.kb.sources()), 3)

    def test_retrieval_finds_right_document(self):
        top = self.kb.search("round robin time quantum", k=2)[0]
        self.assertEqual(top["source"], "os_scheduling.md")
        top = self.kb.search("what is BCNF and 3NF", k=2)[0]
        self.assertEqual(top["source"], "dbms_normalization.md")

    def test_irrelevant_query_returns_nothing(self):
        self.assertEqual(self.kb.search("zebra giraffe savannah migration", k=3), [])

    def test_no_duplicate_chunks_on_reindex(self):
        before = len(self.kb.chunks)
        self.kb.add_folder(SAMPLES)
        self.assertEqual(len(self.kb.chunks), before)

    def test_index_persists(self):
        again = KnowledgeBase(self.tmp.name, use_dense=False)
        self.assertEqual(len(again.chunks), len(self.kb.chunks))

    def test_chunking_respects_size(self):
        chunks = chunk_text("word " * 1000, size=300, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 360 for c in chunks))


class TestJson(unittest.TestCase):
    def test_extract_from_fenced_and_chatty(self):
        self.assertEqual(extract_json('Sure!\n```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('blah {"a": {"b": "}"}} trailing'), {"a": {"b": "}"}})
        with self.assertRaises(ValueError):
            extract_json("no json here")


if __name__ == "__main__":
    unittest.main()
