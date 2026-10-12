class DefaultPredictor:
    def __init__(self, cfg): self.cfg = cfg
    def __call__(self, img): return {"instances": None}
