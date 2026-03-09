"""
容量计算模块：计算不同车道管理策略下的通行能力
Capacity calculation module: Calculate lane capacity under different management strategies
"""

from typing import Dict
from config import HeadwayParams, NumericalParams


class LaneCapacityCalculator:
    """车道容量计算器 Lane capacity calculator"""

    def __init__(self, h_P: float = None, h_C: float = None, h_L: float = None,
                 h_A: float = None, h_I: float = None):
        self.h_P = h_P if h_P is not None else HeadwayParams.h_P
        self.h_C = h_C if h_C is not None else HeadwayParams.h_C
        self.h_L = h_L if h_L is not None else HeadwayParams.h_L
        self.h_A = h_A if h_A is not None else HeadwayParams.h_A
        self.h_I = h_I if h_I is not None else HeadwayParams.h_I

    def capacity_AL(self, P_CAV_norm: float, L_max: int) -> float:
        if P_CAV_norm < NumericalParams.EPSILON:
            return 3600 / self.h_A

        if abs(P_CAV_norm - 1.0) < NumericalParams.EPSILON:
            L_max_safe = min(L_max, NumericalParams.MAX_PLATOON_LENGTH)
            h_bar = (self.h_P * (L_max_safe - 1) + self.h_C) / L_max_safe
        else:
            P_pow_2 = P_CAV_norm ** 2
            P_pow_Lmax = P_CAV_norm ** L_max
            P_pow_Lmax_plus_1 = P_CAV_norm ** (L_max + 1)
            term1 = self.h_P * (P_pow_2 - P_pow_Lmax_plus_1) / (1 - P_pow_Lmax + NumericalParams.EPSILON)
            term2 = self.h_C * P_pow_Lmax_plus_1 * (1 - P_CAV_norm) / (1 - P_pow_Lmax + NumericalParams.EPSILON)
            term3 = self.h_A * (1 - P_pow_2)
            h_bar = term1 + term2 + term3

        return 3600 / max(h_bar, NumericalParams.MIN_HEADWAY)

    def capacity_CL(self, P_CAV_norm: float, L_max: int) -> float:
        P_pow_2 = P_CAV_norm ** 2
        P_pow_Lmax = P_CAV_norm ** L_max
        P_pow_Lmax_plus_1 = P_CAV_norm ** (L_max + 1)
        term1 = self.h_L * (1 - P_CAV_norm)
        term2 = self.h_A * (P_CAV_norm - P_pow_2)
        term3 = self.h_P * (P_pow_2 - P_pow_Lmax_plus_1) / (1 - P_pow_Lmax + NumericalParams.EPSILON)
        term4 = self.h_C * P_pow_Lmax_plus_1 * (1 - P_CAV_norm) / (1 - P_pow_Lmax + NumericalParams.EPSILON)
        h_bar = term1 + term2 + term3 + term4
        return 3600 / max(h_bar, NumericalParams.MIN_HEADWAY)

    def capacity_CAL(self, L_max: int) -> float:
        h_bar = ((L_max - 1) * self.h_P + self.h_C) / L_max
        return 3600 / max(h_bar, NumericalParams.MIN_HEADWAY)

    def capacity_NCAL(self, P_AV_norm: float, P_HV_norm: float, P_CHV_norm: float) -> float:
        h_bar = self.h_A * P_AV_norm + self.h_I * (P_CHV_norm + P_HV_norm)
        return 3600 / max(h_bar, NumericalParams.MIN_HEADWAY)

    def capacity_GL(self, P_CAV_norm: float, P_CHV_norm: float,
                    P_AV_norm: float, P_HV_norm: float, L_max: int) -> float:
        if abs(P_CAV_norm - 1.0) < NumericalParams.EPSILON:
            h_bar = ((L_max - 1) * self.h_P + self.h_C) / L_max
            return 3600 / max(h_bar, NumericalParams.MIN_HEADWAY)

        if abs(P_CHV_norm - 1.0) < NumericalParams.EPSILON:
            return 3600 / self.h_I

        if P_CAV_norm < NumericalParams.EPSILON:
            h_bar = self.h_A * P_AV_norm + self.h_I * (P_CHV_norm + P_HV_norm)
            return 3600 / max(h_bar, NumericalParams.MIN_HEADWAY)

        P_pow_2 = P_CAV_norm ** 2
        P_pow_Lmax = P_CAV_norm ** L_max
        P_pow_Lmax_plus_1 = P_CAV_norm ** (L_max + 1)

        term1 = self.h_P * (P_pow_2 - P_pow_Lmax_plus_1) / (1 - P_pow_Lmax + NumericalParams.EPSILON)
        term2 = self.h_C * P_pow_Lmax_plus_1 * (1 - P_CAV_norm) / (1 - P_pow_Lmax + NumericalParams.EPSILON)

        if P_CHV_norm < NumericalParams.EPSILON:
            term3 = 0.0
            term5_part = 0.0
        else:
            P_CHV_pow_Lmax = P_CHV_norm ** L_max
            term3 = self.h_L * (P_CAV_norm * P_CHV_norm - P_CAV_norm * P_CHV_pow_Lmax) / (1 - P_CHV_norm + NumericalParams.EPSILON)
            term5_part = (P_CAV_norm * P_CHV_norm - P_CAV_norm * P_CHV_pow_Lmax) / (1 - P_CHV_norm + NumericalParams.EPSILON)

        term4 = self.h_A * (P_AV_norm + P_CAV_norm * (1 - P_CAV_norm))
        term5 = self.h_I * (P_HV_norm + P_CHV_norm - term5_part)
        h_bar = term1 + term2 + term3 + term4 + term5
        return 3600 / max(h_bar, NumericalParams.MIN_HEADWAY)

    def calculate_capacity(self, strategy: str, P_lane: Dict[str, float], L_max: int) -> float:
        total_P = sum(P_lane.values())
        if total_P < NumericalParams.EPSILON:
            return 0.0

        P_norm = {vehicle_type: value / total_P for vehicle_type, value in P_lane.items()}

        if strategy == 'AL':
            return self.capacity_AL(P_norm['CAV'], L_max)
        if strategy == 'CL':
            return self.capacity_CL(P_norm['CAV'])
        if strategy == 'CAL':
            return self.capacity_CAL(L_max)
        if strategy == 'NCAL':
            return self.capacity_NCAL(P_norm['AV'], P_norm['HV'], P_norm['CHV'])
        if strategy == 'GL':
            return self.capacity_GL(P_norm['CAV'], P_norm['CHV'], P_norm['AV'], P_norm['HV'], L_max)
        raise ValueError(f"Unknown strategy: {strategy}")
