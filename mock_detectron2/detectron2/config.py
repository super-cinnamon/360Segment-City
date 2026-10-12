class MockCfg:
    def __init__(self):
        self.DATASETS = type('obj', (object,), {'TRAIN': ["coco"]})
        self.MODEL = type('obj', (object,), {'DEVICE': "cuda"})
        self.INPUT = type('obj', (object,), {'MIN_SIZE_TEST': 640, 'MAX_SIZE_TEST': 10000})
        self.MODEL.ROI_HEADS = type('obj', (object,), {'SCORE_THRESH_TEST': 0.4, 'NMS_THRESH_TEST': 0.45})
    def merge_from_file(self, path): pass

def get_cfg():
    return MockCfg()
