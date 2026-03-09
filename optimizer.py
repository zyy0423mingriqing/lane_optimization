"""
专用车道优化求解器
Dedicated lane optimization solver
"""

import math
from typing import Dict, List, Optional

from capacity_calculator import LaneCapacityCalculator
from config import NumericalParams, OptimizationParams


class LaneOptimizer:
    """按照给定流程求解最优车道配置"""

    def __init__(self, calculator: Optional[LaneCapacityCalculator] = None):
        self.calculator = calculator or LaneCapacityCalculator()

    def _log(self, logs: List[str], verbose: bool, message: str) -> None:
        logs.append(message)
        if verbose:
            print(message)

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        if abs(denominator) < NumericalParams.EPSILON:
            return 0.0
        return numerator / denominator

    def _round_value(self, value: float) -> float:
        return round(float(value), OptimizationParams.ROUND_DECIMALS)

    def _validate_inputs(self, P_CAV: float, P_CHV: float, P_AV: float, P_HV: float,
                         n: int, L_max: int) -> Dict[str, float]:
        values = {
            'CAV': float(P_CAV),
            'CHV': float(P_CHV),
            'AV': float(P_AV),
            'HV': float(P_HV),
        }

        for vehicle_type, value in values.items():
            if value < -NumericalParams.EPSILON:
                raise ValueError(f"{vehicle_type} penetration rate must be non-negative, got {value}")

        total = sum(values.values())
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Vehicle penetration rates must sum to 1.0, got {total:.6f}")

        if n <= 1:
            raise ValueError(f"Total lane count n must be greater than 1, got {n}")
        if L_max <= 0:
            raise ValueError(f"L_max must be positive, got {L_max}")

        return values

    def _normalize_penetrations(self, penetrations: Dict[str, float]) -> Dict[str, float]:
        total = sum(penetrations.values())
        if total < NumericalParams.EPSILON:
            raise ValueError("Remaining GL penetration total is zero; cannot normalize")

        normalized = {
            key: self._round_value(max(0.0, value) / total)
            for key, value in penetrations.items()
        }

        rounded_sum = sum(normalized.values())
        diff = self._round_value(1.0 - rounded_sum)
        if abs(diff) >= 10 ** (-OptimizationParams.ROUND_DECIMALS):
            largest_key = max(normalized, key=normalized.get)
            normalized[largest_key] = self._round_value(normalized[largest_key] + diff)

        return normalized

    def _build_strategy_meta(self, strategy: str, P_CAV: float, P_CHV: float,
                             P_AV: float, P_HV: float, L_max: int) -> Optional[Dict]:
        if strategy == 'CL':
            eligible_total = P_CAV + P_CHV
            if eligible_total < NumericalParams.EPSILON:
                return None
            return {
                'strategy': 'CL',
                'eligible_total': eligible_total,
                'capacity': self.calculator.capacity_CL(self._safe_ratio(P_CAV, eligible_total), L_max),
                'mix': {
                    'CAV': self._safe_ratio(P_CAV, eligible_total),
                    'CHV': self._safe_ratio(P_CHV, eligible_total),
                    'AV': 0.0,
                    'HV': 0.0,
                },
                'rounding': 'floor_minus_one_if_integer',
            }

        if strategy == 'AL':
            eligible_total = P_CAV + P_AV
            if eligible_total < NumericalParams.EPSILON:
                return None
            return {
                'strategy': 'AL',
                'eligible_total': eligible_total,
                'capacity': self.calculator.capacity_AL(self._safe_ratio(P_CAV, eligible_total), L_max),
                'mix': {
                    'CAV': self._safe_ratio(P_CAV, eligible_total),
                    'CHV': 0.0,
                    'AV': self._safe_ratio(P_AV, eligible_total),
                    'HV': 0.0,
                },
                'rounding': 'floor_minus_one_if_integer',
            }

        if strategy == 'CAL':
            eligible_total = P_CAV
            if eligible_total < NumericalParams.EPSILON:
                return None
            return {
                'strategy': 'CAL',
                'eligible_total': eligible_total,
                'capacity': self.calculator.capacity_CAL(L_max),
                'mix': {'CAV': 1.0, 'CHV': 0.0, 'AV': 0.0, 'HV': 0.0},
                'rounding': 'floor_minus_one_if_integer',
            }

        if strategy == 'NCAL':
            eligible_total = P_AV + P_CHV + P_HV
            if eligible_total < NumericalParams.EPSILON:
                return None
            return {
                'strategy': 'NCAL',
                'eligible_total': eligible_total,
                'capacity': self.calculator.capacity_NCAL(
                    self._safe_ratio(P_AV, eligible_total),
                    self._safe_ratio(P_HV, eligible_total),
                    self._safe_ratio(P_CHV, eligible_total),
                ),
                'mix': {
                    'CAV': 0.0,
                    'CHV': self._safe_ratio(P_CHV, eligible_total),
                    'AV': self._safe_ratio(P_AV, eligible_total),
                    'HV': self._safe_ratio(P_HV, eligible_total),
                },
                'rounding': 'ceil_plus_one_if_integer',
            }

        raise ValueError(f"Unsupported strategy: {strategy}")

    def _initial_lane_count(self, strategy_meta: Dict, n: int) -> int:
        raw_lane_count = strategy_meta['eligible_total'] * n
        rounded_raw = round(raw_lane_count)

        if strategy_meta['rounding'] == 'floor_minus_one_if_integer':
            if abs(raw_lane_count - rounded_raw) < NumericalParams.EPSILON:
                return int(rounded_raw)
            return int(math.floor(raw_lane_count))

        if strategy_meta['rounding'] == 'ceil_plus_one_if_integer':
            if abs(raw_lane_count - rounded_raw) < NumericalParams.EPSILON:
                return int(rounded_raw)
            return int(math.ceil(raw_lane_count))

        raise ValueError(f"Unsupported rounding rule: {strategy_meta['rounding']}")

    def _validate_capacity_order(self, P_CAV: float, P_CHV: float, P_AV: float, P_HV: float,
                                 L_max: int) -> Dict[str, Optional[float]]:
        GL_origin = self.calculator.capacity_GL(P_CAV, P_CHV, P_AV, P_HV, L_max)
        capacities = {
            'GL': GL_origin,
            'AL': None,
            'CL': None,
            'CAL': None,
            'NCAL': None,
        }

        if P_CAV + P_AV > NumericalParams.EPSILON:
            capacities['AL'] = self.calculator.capacity_AL(self._safe_ratio(P_CAV, P_CAV + P_AV), L_max)
        if P_CAV + P_CHV > NumericalParams.EPSILON:
            capacities['CL'] = self.calculator.capacity_CL(self._safe_ratio(P_CAV, P_CAV + P_CHV), L_max)
        if P_CAV > NumericalParams.EPSILON:
            capacities['CAL'] = self.calculator.capacity_CAL(L_max)
        if P_AV + P_CHV + P_HV > NumericalParams.EPSILON:
            capacities['NCAL'] = self.calculator.capacity_NCAL(
                self._safe_ratio(P_AV, P_AV + P_CHV + P_HV),
                self._safe_ratio(P_HV, P_AV + P_CHV + P_HV),
                self._safe_ratio(P_CHV, P_AV + P_CHV + P_HV),
            )

        checks = []
        if capacities['AL'] is not None and not (capacities['GL'] <= capacities['AL']):
            checks.append(f"GL <= AL violated: {capacities['GL']:.4f} !<= {capacities['AL']:.4f}")
        if capacities['CL'] is not None and not (capacities['GL'] <= capacities['CL']):
            checks.append(f"GL <= CL violated: {capacities['GL']:.4f} !<= {capacities['CL']:.4f}")
        if capacities['CAL'] is not None and not (capacities['GL'] <= capacities['CAL']):
            checks.append(f"GL <= CAL violated: {capacities['GL']:.4f} !<= {capacities['CAL']:.4f}")
        if capacities['NCAL'] is not None and not (capacities['GL'] >= capacities['NCAL']):
            checks.append(f"GL >= NCAL violated: {capacities['GL']:.4f} !>= {capacities['NCAL']:.4f}")

        if checks:
            raise ValueError("Capacity relationship check failed: " + "; ".join(checks))

        return capacities

    def _iterate_strategy_for_lane_count(self, strategy_meta: Dict, n_special: int, n: int,
                                         L_max: int, original_penetrations: Dict[str, float],
                                         verbose: bool, logs: List[str]) -> Optional[Dict]:
        n_GL = n - n_special
        if n_special <= 0 or n_GL <= 0:
            self._log(logs, verbose, f"[{strategy_meta['strategy']}] invalid lane count n_special={n_special}, skip")
            return None

        GL_current = self.calculator.capacity_GL(
            original_penetrations['CAV'],
            original_penetrations['CHV'],
            original_penetrations['AV'],
            original_penetrations['HV'],
            L_max,
        )

        iterations = []
        previous_gap = None
        for iteration in range(1, OptimizationParams.MAX_ITERATIONS + 1):
            alpha = strategy_meta['capacity'] / GL_current
            denominator = alpha * n_special + n_GL
            if denominator < NumericalParams.EPSILON:
                break

            raw_P_special = alpha / denominator
            P_special = min(raw_P_special, strategy_meta['eligible_total'] / n_special)
            assigned = {
                key: P_special * strategy_meta['mix'][key]
                for key in ['CAV', 'CHV', 'AV', 'HV']
            }
            remaining = {
                key: max(0.0, original_penetrations[key] - assigned[key] * n_special)
                for key in ['CAV', 'CHV', 'AV', 'HV']
            }

            total_remaining = sum(remaining.values())
            if total_remaining < NumericalParams.EPSILON:
                self._log(logs, verbose, f"[{strategy_meta['strategy']}] remaining GL flow is zero at iteration {iteration}")
                break

            normalized_GL = self._normalize_penetrations(remaining)
            GL_new = self.calculator.capacity_GL(
                normalized_GL['CAV'],
                normalized_GL['CHV'],
                normalized_GL['AV'],
                normalized_GL['HV'],
                L_max,
            )

            U_special = P_special * 3600 / strategy_meta['capacity']
            U_GL = total_remaining * 3600 / (GL_new * n_GL)
            gap = abs(U_special - U_GL)
            gap_improvement = None if previous_gap is None else previous_gap - gap

            iteration_result = {
                'iteration': iteration,
                'alpha': alpha,
                'P_special': P_special,
                'assigned_per_special_lane': assigned,
                'remaining_GL_penetrations': remaining,
                'remaining_total': total_remaining,
                'normalized_GL_penetrations': normalized_GL,
                'GL_new': GL_new,
                'U_special': U_special,
                'U_GL': U_GL,
                'gap': gap,
                'gap_improvement': gap_improvement,
            }
            iterations.append(iteration_result)

            self._log(
                logs,
                verbose,
                f"[{strategy_meta['strategy']}] n_special={n_special}, iter={iteration}, "
                f"alpha={alpha:.6f}, P_special={P_special:.6f}, GL_new={GL_new:.4f}, "
                f"U_special={U_special:.6f}, U_GL={U_GL:.6f}, gap={gap:.6f}"
            )

            if gap < OptimizationParams.UE_TOLERANCE:
                total_capacity = strategy_meta['capacity'] * n_special + GL_new * n_GL
                self._log(logs, verbose, f"[{strategy_meta['strategy']}] converged at n_special={n_special}, iter={iteration}, best={total_capacity:.4f}")
                return {
                    'strategy': strategy_meta['strategy'],
                    'feasible': True,
                    'n_special': n_special,
                    'n_GL': n_GL,
                    'special_capacity': strategy_meta['capacity'],
                    'total_capacity': total_capacity,
                    'iterations': iterations,
                    'final_iteration': iteration_result,
                }

            if previous_gap is not None and abs(previous_gap - gap) <= NumericalParams.EPSILON:
                self._log(logs, verbose, f"[{strategy_meta['strategy']}] gap improvement is zero at n_special={n_special}, iter={iteration}")
                break

            previous_gap = gap
            GL_current = GL_new

        self._log(logs, verbose, f"[{strategy_meta['strategy']}] did not converge for n_special={n_special}")
        return {
            'strategy': strategy_meta['strategy'],
            'feasible': False,
            'n_special': n_special,
            'n_GL': n_GL,
            'special_capacity': strategy_meta['capacity'],
            'iterations': iterations,
            'final_iteration': iterations[-1] if iterations else None,
        }

    def _solve_single_strategy(self, strategy: str, P_CAV: float, P_CHV: float,
                               P_AV: float, P_HV: float, n: int, L_max: int,
                               verbose: bool, logs: List[str]) -> Dict:
        original_penetrations = {
            'CAV': P_CAV,
            'CHV': P_CHV,
            'AV': P_AV,
            'HV': P_HV,
        }
        strategy_meta = self._build_strategy_meta(strategy, P_CAV, P_CHV, P_AV, P_HV, L_max)
        if strategy_meta is None:
            self._log(logs, verbose, f"[{strategy}] ineligible because eligible penetration is zero")
            return {'strategy': strategy, 'feasible': False, 'reason': 'eligible penetration is zero', 'attempts': [], 'best': None}

        initial_lane_count = self._initial_lane_count(strategy_meta, n)
        self._log(logs, verbose, f"[{strategy}] initial lane count = {initial_lane_count}")

        attempts = []
        n_special = initial_lane_count
        while n_special > 0:
            result = self._iterate_strategy_for_lane_count(
                strategy_meta=strategy_meta,
                n_special=n_special,
                n=n,
                L_max=L_max,
                original_penetrations=original_penetrations,
                verbose=verbose,
                logs=logs,
            )
            if result is None:
                break

            attempts.append(result)
            if result['feasible']:
                return {
                    'strategy': strategy,
                    'feasible': True,
                    'reason': None,
                    'initial_lane_count': initial_lane_count,
                    'attempts': attempts,
                    'best': result,
                }

            n_special -= 1
            self._log(logs, verbose, f"[{strategy}] reduce dedicated lanes to {n_special} and retry")

        return {
            'strategy': strategy,
            'feasible': False,
            'reason': 'no converged lane allocation found',
            'initial_lane_count': initial_lane_count,
            'attempts': attempts,
            'best': None,
        }

    def optimize(self, P_CAV: float, P_CHV: float, P_AV: float, P_HV: float,
                 n: int, L_max: int, verbose: bool = None) -> Dict:
        verbose = OptimizationParams.DEFAULT_VERBOSE if verbose is None else verbose
        logs: List[str] = []

        validated = self._validate_inputs(P_CAV, P_CHV, P_AV, P_HV, n, L_max)
        P_CAV = validated['CAV']
        P_CHV = validated['CHV']
        P_AV = validated['AV']
        P_HV = validated['HV']

        GL_origin = self.calculator.capacity_GL(P_CAV, P_CHV, P_AV, P_HV, L_max)
        best_origin = GL_origin * n

        capacities = self._validate_capacity_order(P_CAV, P_CHV, P_AV, P_HV, L_max)

        cl_denominator = P_CHV + P_CAV
        cl_ratio = self._safe_ratio(P_CHV, cl_denominator)
        cl_threshold = 0.05 ** (1/L_max)
        candidate_strategies = ['CL', 'AL', 'CAL', 'NCAL'] if cl_denominator > NumericalParams.EPSILON and cl_ratio < cl_threshold else ['AL', 'CAL', 'NCAL']

        strategy_results = {}
        best_result = {
            'strategy': 'GL',
            'lane_allocation': {'GL': n},
            'total_capacity': best_origin,
            'improvement': 0.0,
            'improvement_percent': 0.0,
        }

        for strategy in candidate_strategies:
            result = self._solve_single_strategy(strategy, P_CAV, P_CHV, P_AV, P_HV, n, L_max, verbose, logs)
            strategy_results[strategy] = result

            if not result['feasible'] or result['best'] is None:
                continue

            total_capacity = result['best']['total_capacity']
            if total_capacity > best_result['total_capacity']:
                best_result = {
                    'strategy': strategy,
                    'lane_allocation': {strategy: result['best']['n_special'], 'GL': result['best']['n_GL']},
                    'total_capacity': total_capacity,
                    'improvement': total_capacity - best_origin,
                    'improvement_percent': (total_capacity - best_origin) / best_origin * 100,
                }

        if verbose:
            print(f"最优策略: {best_result['strategy']}, 车道分配: {best_result['lane_allocation']}")
            print(f"总容量: {best_result['total_capacity']:.2f} veh/h, 提升: {best_result['improvement_percent']:.2f}%")

        return {
            'inputs': {
                'P_CAV': P_CAV,
                'P_CHV': P_CHV,
                'P_AV': P_AV,
                'P_HV': P_HV,
                'n': n,
                'L_max': L_max,
            },
            'baseline': {
                'GL_origin': GL_origin,
                'best_origin': best_origin,
            },
            'capacities': capacities,
            'candidate_strategies': candidate_strategies,
            'strategy_results': strategy_results,
            'best_result': best_result,
            'improvement': {
                'absolute': best_result['improvement'],
                'percent': best_result['improvement_percent'],
            },
            'logs': logs,
        }
