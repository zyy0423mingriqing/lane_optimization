"""
配置文件：定义模型参数和常量
Configuration file: Model parameters and constants
"""

# 车辆类型 Vehicle types
VEHICLE_TYPES = ['CAV', 'CHV', 'AV', 'HV']

# 车道管理策略 Lane management strategies
STRATEGIES = ['AL', 'CL', 'CAL', 'NCAL', 'GL']
DEDICATED_STRATEGIES = ['AL', 'CL', 'CAL', 'NCAL']
GENERAL_STRATEGY = 'GL'

# 车辆-策略兼容性矩阵 Vehicle-strategy eligibility matrix
# E[v,k] = 1 表示车辆类型v可以使用策略k
ELIGIBILITY_MATRIX = {
    'CAV': {'AL': 1, 'CL': 1, 'CAL': 1, 'NCAL': 0, 'GL': 1},
    'CHV': {'AL': 0, 'CL': 1, 'CAL': 0, 'NCAL': 1, 'GL': 1},
    'AV':  {'AL': 1, 'CL': 0, 'CAL': 0, 'NCAL': 1, 'GL': 1},
    'HV':  {'AL': 0, 'CL': 0, 'CAL': 0, 'NCAL': 1, 'GL': 1}
}

# 车头时距参数 (秒) Headway parameters (seconds)
class HeadwayParams:
    """车头时距参数类"""
    h_P = 0.6   # 车队内部时距 Platoon headway
    h_C = 1.0   # 切入时距 Cut-in headway
    h_L = 1.5   # 领队时距 Leader headway
    h_A = 1.6   # 自动驾驶时距 Autonomous headway
    h_I = 2.0   # 人工驾驶时距 Human/independent headway


# 优化算法参数 Optimization algorithm parameters
class OptimizationParams:
    """优化算法参数类"""

    # 约束处理 Constraint handling
    BIG_M = 10.0  # 大M常数
    PENALTY_WEIGHT_HARD = 1e8  # 硬约束违反惩罚权重（车辆准入性、渗透率守恒）
    PENALTY_WEIGHT_SOFT = 1000.0  # 软约束违反惩罚权重（策略一致性等）
    BALANCE_WEIGHT = 1e8  # 车道平衡权重μ（用户均衡硬约束）

    # 专用车道优化求解参数 Dedicated-lane optimization parameters
    UE_TOLERANCE = 1e-4  # 用户均衡收敛容差
    MAX_ITERATIONS = 10  # 单车道配置最大迭代次数
    ROUND_DECIMALS = 4  # 归一化渗透率保留小数位
    DEFAULT_VERBOSE = False  # 是否默认输出逐步日志

# 数值稳定性参数 Numerical stability parameters
class NumericalParams:
    """数值计算参数类"""
    EPSILON = 1e-6  # 数值零阈值
    MIN_HEADWAY = 0.5  # 最小车头时距
    MAX_PLATOON_LENGTH = 12  # 最大车队长度限制（用于数值稳定）
