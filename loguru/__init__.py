try:
    from ._colorizer import AnsiParser  # type: ignore
except ImportError:
    class AnsiParser:
        def feed(self, *args, **kwargs):
            pass

        def get_style_str(self):
            return ""

else:
    class _ColorizerModule:
        AnsiParser = AnsiParser

    _colorizer = _ColorizerModule()


class _DummyLogger:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass

        return method


logger = _DummyLogger()
