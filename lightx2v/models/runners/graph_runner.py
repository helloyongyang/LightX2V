from lightx2v.utils.profiler import ProfilingContext4Debug


class GraphRunner:
    def __init__(self, runner):
        self.runner = runner
        self.compile()

    def compile(self):
        print("start compile...")
        with ProfilingContext4Debug("compile"):
            self.runner.run_step()
        print("end compile...")

    def run_pipeline(self):
        return self.runner.run_pipeline()
