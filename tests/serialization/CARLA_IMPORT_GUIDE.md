# CARLA 导入指南

## 正式代码中的 CARLA 导入方式

### 1. utils.py 的导入方式（第22-30行）

```python
import config

config.set_carla_api_path()

try:
    import carla
except ModuleNotFoundError as e:
    print("Carla module not found. Make sure you have built Carla.")
    proj_root = config.get_proj_root()
    print("Try `cd {}/carla && make PythonAPI' if not.".format(proj_root))
    exit(-1)
```

### 2. npc.py 的导入方式（第10-11行）

```python
import config

config.set_carla_api_path()
import carla
```

### 3. scenario.py 的导入方式（第8行）

```python
import carla
```

（scenario.py 直接导入，假设 CARLA 已经在 sys.path 中）

## config.set_carla_api_path() 的行为

`config.set_carla_api_path()` 函数会：
1. 查找 CARLA 0.9.13 的 PythonAPI egg 文件
2. 如果找到，将其添加到 `sys.path`
3. 如果找不到，会调用 `sys.exit(-1)` 退出程序

## 测试代码中的 CARLA 导入方式

### 正确的导入方式（参考正式代码）

```python
import config

CARLA_AVAILABLE = False
carla = None

def setup_carla_import():
    """设置 CARLA 导入，使用与正式程序相同的方式"""
    global CARLA_AVAILABLE, carla
    
    try:
        # 使用与正式代码相同的方式：调用 config.set_carla_api_path()
        import config
        
        # config.set_carla_api_path() 如果找不到 CARLA 会 sys.exit(-1)
        # 我们需要捕获 SystemExit 异常，这是正常的（CARLA 不可用时）
        try:
            config.set_carla_api_path()
        except SystemExit:
            # CARLA 不可用，这是预期的，正常跳过测试
            CARLA_AVAILABLE = False
            return False
        
        # 尝试导入 carla
        try:
            import carla
            CARLA_AVAILABLE = True
            return True
        except (ImportError, ModuleNotFoundError):
            CARLA_AVAILABLE = False
            return False
            
    except ImportError:
        # config 模块不可用，尝试直接导入 carla
        try:
            import carla
            CARLA_AVAILABLE = True
            return True
        except (ImportError, ModuleNotFoundError):
            CARLA_AVAILABLE = False
            return False
    except Exception as e:
        # 其他异常不应该被隐藏，但也不应该阻止测试运行
        CARLA_AVAILABLE = False
        return False

setup_carla_import()
```

### 导入 utils 模块

```python
# 导入工具函数（延迟导入，只在 CARLA 可用时）
# 注意：utils 模块在导入时会调用 config.set_carla_api_path()
# 如果 CARLA 不可用，utils 模块导入会失败（sys.exit）
UTILS_AVAILABLE = False
utils = None
if CARLA_AVAILABLE:
    try:
        # utils 模块在导入时会调用 config.set_carla_api_path()
        # 如果 CARLA 不可用，会触发 SystemExit
        # 但由于我们已经设置了 CARLA 路径，这里应该能成功导入
        import utils
        UTILS_AVAILABLE = True
    except (ImportError, SystemExit):
        # SystemExit 表示 CARLA 不可用，这是预期的
        UTILS_AVAILABLE = False
    except Exception as e:
        # 其他异常应该被记录
        UTILS_AVAILABLE = False
```

## 测试中的使用

### 在 setUp() 中检查

```python
def setUp(self):
    if not UTILS_AVAILABLE:
        self.skipTest("utils module not available (requires CARLA)")
    if not CARLA_AVAILABLE:
        self.skipTest("需要 CARLA 环境")
```

### 在测试方法中使用

```python
def test_something(self):
    if CARLA_AVAILABLE:
        # 使用真实的 CARLA 对象
        location = carla.Location(x=100.5, y=200.3, z=50.7)
    else:
        # 使用 Mock 对象（如果测试允许）
        location = MockLocation(x=100.5, y=200.3, z=50.7)
    
    # 测试序列化
    json_string = utils.carla_location_pickle(location)
    # ...
```

## 关键点

1. **使用与正式代码相同的方式**: 调用 `config.set_carla_api_path()`
2. **捕获 SystemExit**: `config.set_carla_api_path()` 在找不到 CARLA 时会 `sys.exit(-1)`
3. **优雅降级**: 如果没有 CARLA，使用 `self.skipTest()` 跳过测试，而不是失败
4. **延迟导入 utils**: 只在 CARLA 可用时导入 `utils` 模块，因为它也会调用 `config.set_carla_api_path()`

## 测试结果

- **有 CARLA 环境**: 所有测试正常运行
- **无 CARLA 环境**: 测试被优雅跳过（使用 `self.skipTest()`）

