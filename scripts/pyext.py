import inspect
import types

# 修复 Python 3.11 缺失接口
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

class RuntimeModule(types.ModuleType):
    def __init__(self, name, code, doc=None):
        super(RuntimeModule, self).__init__(name, doc)
        self.__dict__.update(code if isinstance(code, dict) else {})
        self.__file__ = "<runtime_module:%s>" % name

# 这里可以根据需要补充 pyext 的其他简单方法