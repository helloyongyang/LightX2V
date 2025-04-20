from lightx2v.utils.profiler import ProfilingContext4Debug


class DefaultRunner:
    def __init__(self, model, inputs):
        self.model = model
        self.inputs = inputs

    def run(self):
        for step_index in range(self.model.scheduler.infer_steps):
            print(f"==> step_index: {step_index + 1} / {self.model.scheduler.infer_steps}")

            with ProfilingContext4Debug("step_pre"):
                self.model.scheduler.step_pre(step_index=step_index)

            with ProfilingContext4Debug("infer"):
                self.model.infer(self.inputs)

            with ProfilingContext4Debug("step_post"):
                self.model.scheduler.step_post()

        return self.model.scheduler.latents, self.model.scheduler.generator

    def run_step(self, step_index=0):
        self.model.scheduler.step_pre(step_index=step_index)
        self.model.infer(self.inputs)
        self.model.scheduler.step_post()
