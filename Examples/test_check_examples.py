from unittest import TestCase



class TestExampleCases(TestCase):
    # Run the quickstart example to ensure that it is up to date with the API
    def test_run_quick_start(self):
        try:
            from . import quick_start

        except Exception as e:
            self.fail(e)



