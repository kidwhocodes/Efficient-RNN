def subplots(*args, **kwargs):
    class DummyFig:
        def savefig(self, *a, **k): pass
        def tight_layout(self): pass
        def colorbar(self, *a, **k): pass
    class DummyAx:
        def plot(self, *a, **k): pass
        def set_title(self, *a, **k): pass
        def set_xlabel(self, *a, **k): pass
        def set_ylabel(self, *a, **k): pass
        def grid(self, *a, **k): pass
        def legend(self, *a, **k): pass
        def set_ylim(self, *a, **k): pass
    return DummyFig(), DummyAx()

def colorbar(*args, **kwargs): pass
def close(*args, **kwargs): pass
