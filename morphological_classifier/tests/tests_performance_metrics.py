import unittest
from morphological_classifier import classifier

class TestPerformanceMetrics(unittest.TestCase):
    def setUp(self):
        self.tags = [0, 1, 2]
        self.perf_metrics = classifier.PerformanceMetrics(self.tags)

        actual = [0, 1, 0, 1, 0, 0, 2, 2]
        pred = [0, 1, 1, 1, 1, 0, 0, 1]
        for a, p in zip(actual, pred):
            self.perf_metrics.update_predicted(a, p)

    def test_update_predicted(self):
        pm = self.perf_metrics
        self.assertEqual(pm[0, 0], 2)
        self.assertEqual(pm[0, 1], 2)
        self.assertEqual(pm[0, 2], 0)
        self.assertEqual(pm[1, 0], 0)
        self.assertEqual(pm[1, 1], 2)
        self.assertEqual(pm[1, 2], 0)
        self.assertEqual(pm[2, 0], 1)
        self.assertEqual(pm[2, 1], 1)
        self.assertEqual(pm[2, 2], 0)

    def test_total_count(self):
        pm = self.perf_metrics
        self.assertEqual(pm._total_count(0), 4)
        self.assertEqual(pm._total_count(1), 2)
        self.assertEqual(pm._total_count(2), 2)

    def test_tags_accuracies(self):
        TP = self.perf_metrics.tag_accuracies()
        self.assertEqual(TP[0], 0.5)
        self.assertEqual(TP[1], 1)
        self.assertEqual(TP[2], 0)

    def test_normalize(self):
        pm = self.perf_metrics
        self.perf_metrics.normalize()
        self.assertEqual(pm[0, 0], 0.5)
        self.assertEqual(pm[0, 1], 0.5)
        self.assertEqual(pm[0, 2], 0)
        self.assertEqual(pm[1, 0], 0)
        self.assertEqual(pm[1, 1], 1)
        self.assertEqual(pm[1, 2], 0)
        self.assertEqual(pm[2, 0], 0.5)
        self.assertEqual(pm[2, 1], 0.5)
        self.assertEqual(pm[2, 2], 0)



if __name__ == '__main__':
    unittest.main()
