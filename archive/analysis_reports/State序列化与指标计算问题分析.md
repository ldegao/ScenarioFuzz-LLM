# State序列化与指标计算问题详细分析

## 一、问题概述

根据您的反馈，所有场景都正确执行了，但存在以下问题：
1. **JSON中state的序列化/反序列化失败**：场景JSON文件缺少完整的state数据
2. **场景行为统计出现问题**：所有场景的BCM值完全相同，PC/PEC/TCD为0

## 二、State序列化问题分析

### 2.1 问题根源：`dump_states()` 方法未保存完整state

**关键代码位置**：`scenario.py` 第305-340行

```python
def dump_states(self, state, log_type):
    # ...
    event_dict = {
        "crash": state.crashed,
        "stuck": state.stuck,
        "lane_invasion": state.laneinvaded,
        "red": state.red_violation,
        "speeding": state.speeding,
        "other": state.other_error,
        "other_error_val": state.other_error_val
    }
    config_dict = {
        "fps": c.FRAME_RATE,
        "max_dist_from_player": c.MAX_DIST_FROM_PLAYER,
        "min_dist_from_player": c.MIN_DIST_FROM_PLAYER,
        "abort_seconds": self.conf.timeout,
        "wait_autoware_num_topics": c.WAIT_AUTOWARE_NUM_NODES
    }

    # ⚠️ 问题：只保存了events和config，没有保存完整的state数据！
    state_dict = {"events": event_dict, "config": config_dict}
    filename = "gid:{}_sid:{}.json".format(self.generation_id, self.scenario_id)
    if log_type == "queue":
        out_dir = self.conf.queue_dir
    with open(os.path.join(out_dir, filename), "w") as fp:
        json.dump(state_dict, fp)
    return filename
```

**问题分析**：
- `dump_states()` 方法只保存了 `events` 和 `config` 两个字段
- **没有保存完整的 `state` 数据**，包括：
  - `speed`, `speed_lim`
  - `yaw_list`, `yaw_rate_list`
  - `lon_speed_list`, `lat_speed_list`
  - `steer_angle_list`
  - `min_dist`, `min_dist_frame`
  - 等等所有用于指标计算的数据

### 2.2 State数据的实际存储位置

**State数据在内存中**：
- 场景执行后，`ind.state` 对象包含完整的执行数据
- 这些数据在内存中可用，用于指标计算

**State数据未保存到JSON**：
- `dump_states()` 方法没有将 `state` 数据序列化到JSON文件
- 导致JSON文件中只有 `events` 和 `config`

### 2.3 为什么需要保存State数据

1. **指标计算需要**：
   - PC需要：`a_long`, `a_lat`, `jerk_long`, `jerk_lat`, `yaw_rate`, `ttc`
   - PEC需要：`speed`, `a_long`, `yaw_rate`, `lat_speed_list` 等
   - TCD需要：`lon_speed_list`, `lat_speed_list`, `yaw_list`, `speed` 等
   - BCM需要：`crashed`, `laneinvaded`, `stuck`, `speed`, `speed_lim` 等

2. **离线分析需要**：
   - 如果场景JSON文件没有state数据，无法进行离线指标计算
   - 无法重新分析历史场景

3. **数据完整性**：
   - 场景执行的所有数据应该被完整保存
   - 便于后续分析和调试

## 三、指标计算问题分析

### 3.1 指标计算时机

**代码位置**：`fuzzer.py` 第955-1009行

```python
# Calculate additional metrics if enabled
if conf and getattr(conf, "enable_rag_metrics", False):
    try:
        from metrics import ParameterCoverage, BehaviorCoverage, TrajectoryDiversity, BehaviorMatrix
        
        # Parameter Coverage (PC)
        pc_calculator = ParameterCoverage()
        pc_score = pc_calculator.calculate_coverage([ind])  # ← 传入单个场景
        
        # Behavior Coverage (PEC)
        pec_calculator = BehaviorCoverage()
        pec_score = pec_calculator.calculate_coverage([ind])  # ← 传入单个场景
        
        # Trajectory Diversity (TCD)
        tcd_calculator = TrajectoryDiversity()
        tcd_results = tcd_calculator.calculate_coverage([ind])  # ← 传入单个场景
        
        # Behavior Matrix Coverage (BCM)
        bcm_calculator = BehaviorMatrix()
        bcm_results = bcm_calculator.calculate_coverage([ind])  # ← 传入单个场景
```

**关键问题**：
- 指标计算时传入的是**单个场景** `[ind]`
- 但指标计算逻辑期望的是**累积的所有场景**

### 3.2 PC/PEC/TCD为0的原因

#### PC (Parameter Coverage) 为0的原因：

```python
# metrics/parameter_coverage.py
def calculate_coverage(self, scenarios: List[Scenario]) -> float:
    self.covered_combinations.clear()  # ← 每次计算都清空
    
    for scenario in scenarios:
        combination = self._extract_parameter_combination(scenario)
        if combination is not None:
            self.covered_combinations.add(combination)
    
    covered_count = len(self.covered_combinations)
    if covered_count == 0:
        return 0.0  # ← 如果没有有效组合，返回0
```

**问题分析**：
1. **每次计算都清空**：`self.covered_combinations.clear()` 导致每次只计算当前场景
2. **单个场景无法计算覆盖率**：覆盖率需要累积多个场景的数据
3. **State数据可能不完整**：如果 `scenario.state` 的某些字段为空，`_extract_parameter_combination()` 返回 `None`

#### PEC (Behavior Coverage) 为0的原因：

```python
# metrics/driving_behavior_class_coverage.py
def calculate_coverage(self, scenarios: List[Scenario]) -> float:
    self.covered_classes.clear()  # ← 每次计算都清空
    
    for scenario in scenarios:
        if not hasattr(scenario, 'state') or not scenario.state:
            continue  # ← 跳过没有state的场景
        
        behaviors = self.classify_behavior(scenario.state)
        self.covered_classes.update(behaviors)
    
    coverage = len(self.covered_classes) / total_classes
    return float(coverage)
```

**问题分析**：
1. **每次计算都清空**：`self.covered_classes.clear()` 导致每次只计算当前场景
2. **单个场景可能无法触发所有行为**：行为覆盖率需要累积多个场景
3. **State数据可能不完整**：如果 `scenario.state` 数据不完整，`classify_behavior()` 可能返回空集合

#### TCD (Trajectory Diversity) 为0的原因：

```python
# metrics/trajectory_diversity.py
def calculate_coverage(self, scenarios: List[Scenario]) -> Dict[str, float]:
    trajectories = self.extract_trajectories(scenarios)
    
    if len(trajectories) < 2:  # ← 需要至少2个轨迹
        return {
            'diversity_score': 0.0  # ← 如果轨迹数<2，返回0
        }
```

**问题分析**：
1. **需要至少2个轨迹**：多样性计算需要比较多个轨迹
2. **单个场景无法计算多样性**：传入单个场景 `[ind]` 时，只能提取1个轨迹
3. **直接返回0**：当轨迹数 < 2 时，直接返回 `diversity_score: 0.0`

### 3.3 BCM值完全相同的原因

#### BCM计算逻辑：

```python
# metrics/behavior_matrix.py
def calculate_coverage(self, scenarios: List[Scenario]) -> Dict[str, float]:
    matrix = self.build_matrix(scenarios)
    
    # 计算唯一行为组合数
    unique_combinations = set()
    for j in range(matrix.shape[1]):
        combination = tuple(matrix[:, j])
        unique_combinations.add(combination)
    
    covered_count = len(unique_combinations)
    total_possible = 2 ** 12  # 12种行为，2^12 = 4096
    
    # 对数归一化
    coverage_ratio = np.log1p(covered_count) / np.log1p(total_possible)
    return {'coverage_ratio': coverage_ratio}
```

**问题分析**：
1. **每次计算都重新构建矩阵**：`build_matrix()` 每次只处理传入的场景列表
2. **单个场景只有1种行为组合**：如果所有场景的行为标签都相同，`unique_combinations` 只有1个
3. **固定值计算**：`log(1+1)/log(1+4096) = 0.0833308877281607`

**为什么所有场景的行为标签相同**：
- 如果所有场景都在早期就crash了，行为标签都是 `['collision']`
- 或者所有场景都执行了相同的简单行为
- 导致所有场景的行为组合相同

## 四、单元测试代码检查

### 4.1 现有的序列化测试

**文件位置**：
- `tests/test_serialization.py`：测试JSON序列化（但不包括Scenario state）
- `tests/serialization/test_json_serialization.py`：测试JSON序列化
- `tests/test_states_and_constants.py`：测试ScenarioState初始化

**测试覆盖情况**：
- ✅ 测试了JSON基本序列化
- ✅ 测试了Token tracker、Progress tracker等
- ❌ **没有测试Scenario state的序列化/反序列化**
- ❌ **没有测试指标计算时state数据的完整性**

### 4.2 缺失的测试

**应该添加的测试**：

1. **Scenario state序列化测试**：
   ```python
   def test_scenario_state_serialization():
       """测试Scenario state的序列化/反序列化"""
       scenario = Scenario(conf, seed_data)
       scenario.state.speed = [50, 60, 70]
       scenario.state.yaw_list = [0, 10, 20]
       # ... 设置其他state数据
       
       # 测试序列化
       state_dict = scenario.dump_states(scenario.state, "queue")
       # 验证state数据是否被保存
   ```

2. **指标计算时state数据完整性测试**：
   ```python
   def test_metrics_calculation_with_state():
       """测试指标计算时state数据的完整性"""
       scenario = Scenario(conf, seed_data)
       # 设置完整的state数据
       scenario.state.speed = [50, 60, 70]
       scenario.state.a_long = [1.0, 2.0, 3.0]
       # ...
       
       # 测试指标计算
       pc_calculator = ParameterCoverage()
       pc_score = pc_calculator.calculate_coverage([scenario])
       # 验证指标值是否正常
   ```

3. **累积指标计算测试**：
   ```python
   def test_cumulative_metrics_calculation():
       """测试累积指标计算"""
       scenarios = [scenario1, scenario2, scenario3, ...]
       
       # 测试累积计算
       pc_calculator = ParameterCoverage()
       pc_score = pc_calculator.calculate_coverage(scenarios)
       # 验证累积覆盖率是否正确
   ```

## 五、问题修复建议

### 5.1 修复State序列化问题

**方案1：修改 `dump_states()` 方法**

```python
def dump_states(self, state, log_type):
    # ... 现有的events和config代码 ...
    
    # 添加完整的state数据序列化
    state_data = {
        "speed": state.speed,
        "speed_lim": state.speed_lim,
        "yaw_list": state.yaw_list,
        "yaw_rate_list": state.yaw_rate_list,
        "lon_speed_list": state.lon_speed_list,
        "lat_speed_list": state.lat_speed_list,
        "steer_angle_list": state.steer_angle_list,
        "min_dist": state.min_dist,
        "min_dist_frame": state.min_dist_frame,
        "num_frames": state.num_frames,
        "elapsed_time": state.elapsed_time,
        # ... 其他需要的字段 ...
    }
    
    state_dict = {
        "events": event_dict,
        "config": config_dict,
        "state": state_data  # ← 添加完整的state数据
    }
    
    filename = "gid:{}_sid:{}.json".format(self.generation_id, self.scenario_id)
    if log_type == "queue":
        out_dir = self.conf.queue_dir
    with open(os.path.join(out_dir, filename), "w") as fp:
        json.dump(state_dict, fp, ensure_ascii=False, indent=2)
    return filename
```

**方案2：使用pickle保存完整Scenario对象**

```python
def save_scenario(self, scenario, log_type):
    """保存完整的Scenario对象（包括state）"""
    filename = "gid:{}_sid:{}.pkl".format(scenario.generation_id, scenario.scenario_id)
    if log_type == "queue":
        out_dir = self.conf.queue_dir
    
    with open(os.path.join(out_dir, filename), "wb") as fp:
        pickle.dump(scenario, fp)
    
    # 同时保存JSON格式（用于快速查看）
    self.dump_states(scenario.state, log_type)
```

### 5.2 修复指标计算问题

**方案1：累积计算指标**

```python
# 在fuzzer.py中，维护一个累积的场景列表
accumulated_scenarios = []  # 全局或类属性

# 在指标计算时
if conf and getattr(conf, "enable_rag_metrics", False):
    # 将当前场景添加到累积列表
    accumulated_scenarios.append(ind)
    
    # 使用累积的场景列表计算指标
    pc_calculator = ParameterCoverage()
    pc_score = pc_calculator.calculate_coverage(accumulated_scenarios)
    
    # 但记录的是当前场景的增量值
    # 或者记录累积值（取决于需求）
```

**方案2：修改指标计算逻辑，支持单场景计算**

```python
# 修改指标计算类，支持增量计算
class ParameterCoverage:
    def __init__(self):
        self.covered_combinations: Set[Tuple] = set()  # 不每次清空
    
    def calculate_coverage(self, scenarios: List[Scenario], reset: bool = False):
        if reset:
            self.covered_combinations.clear()
        
        # 累积计算
        for scenario in scenarios:
            combination = self._extract_parameter_combination(scenario)
            if combination is not None:
                self.covered_combinations.add(combination)
        
        # ... 计算覆盖率 ...
```

### 5.3 修复行为统计问题

**检查行为标签提取逻辑**：

```python
# 在behavior_matrix.py中，添加调试信息
def label_behaviors(self, scenario_state: ScenarioState) -> List[str]:
    behaviors = []
    
    # 添加调试日志
    print(f"[DEBUG] Labeling behaviors for scenario {scenario_state.scenario_id}")
    print(f"[DEBUG] crashed: {scenario_state.crashed}")
    print(f"[DEBUG] stuck: {scenario_state.stuck}")
    # ...
    
    # 检查为什么所有场景的行为标签相同
    # 可能需要调整阈值或逻辑
```

## 六、验证步骤

### 6.1 验证State序列化

1. **检查场景JSON文件**：
   ```bash
   # 查看场景文件是否包含state数据
   cat experiment_results/SimilarityComparison/.../queue/gid:1_sid:106.json | jq '.state'
   ```

2. **验证State数据完整性**：
   ```python
   # 加载场景JSON文件
   with open('scenario.json', 'r') as f:
       data = json.load(f)
   
   # 检查state字段是否存在
   assert 'state' in data
   assert 'speed' in data['state']
   assert 'yaw_list' in data['state']
   # ...
   ```

### 6.2 验证指标计算

1. **测试单个场景的指标计算**：
   ```python
   from scenario import Scenario
   from metrics import ParameterCoverage
   
   scenario = Scenario(conf, seed_data)
   # 设置完整的state数据
   scenario.state.speed = [50, 60, 70]
   # ...
   
   pc_calculator = ParameterCoverage()
   pc_score = pc_calculator.calculate_coverage([scenario])
   print(f"PC score: {pc_score}")
   ```

2. **测试累积指标计算**：
   ```python
   scenarios = [scenario1, scenario2, scenario3, ...]
   pc_calculator = ParameterCoverage()
   pc_score = pc_calculator.calculate_coverage(scenarios)
   print(f"Cumulative PC score: {pc_score}")
   ```

## 七、总结

### 7.1 核心问题

1. **State序列化不完整**：`dump_states()` 方法只保存了 `events` 和 `config`，没有保存完整的 `state` 数据
2. **指标计算逻辑错误**：每次计算都传入单个场景，但指标需要累积计算
3. **行为统计问题**：所有场景的行为标签相同，导致BCM值固定

### 7.2 修复优先级

1. **高优先级**：修复指标计算逻辑，支持累积计算
2. **中优先级**：修复State序列化，保存完整数据
3. **低优先级**：添加单元测试，验证修复效果

### 7.3 建议的修复顺序

1. 先修复指标计算逻辑（累积计算）
2. 再修复State序列化（保存完整数据）
3. 最后添加单元测试（验证修复）

---

**报告生成时间**: 2025-12-13  
**分析依据**: 
- `scenario.py` 中的 `dump_states()` 方法
- `fuzzer.py` 中的指标计算代码
- `metrics/` 目录下的指标计算类
- 现有的单元测试代码

