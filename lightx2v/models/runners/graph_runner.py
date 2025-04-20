import copy
from lightx2v.utils.profiler import ProfilingContext4Debug


class GraphRunner:
    def __init__(self, runner):
        self.runner = runner
        self.compile()

    def compile(self):
        scheduler = copy.deepcopy(self.runner.model.scheduler)
        inputs = copy.deepcopy(self.runner.inputs)

        print("start compile...")
        with ProfilingContext4Debug("compile"):
            self.runner.run_step()
        print("end compile...")

        self.runner.model.set_scheduler(scheduler)
        setattr(self.runner, "inputs", inputs)

    def run(self):
        return self.runner.run()
