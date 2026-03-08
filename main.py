from pprint import pprint

from optimizer import LaneOptimizer


def main() -> None:
    optimizer = LaneOptimizer()

    result = optimizer.optimize(
        P_CAV=0.3,
        P_CHV=0.3,
        P_AV=0.2,
        P_HV=0.2,
        n=5,
        L_max=5,
        verbose=True,
    )

    print("\n===== 最优结果 =====")
    pprint(result['best_result'])

    print("\n===== 基线结果 =====")
    pprint(result['baseline'])

    print("\n===== 候选策略 =====")
    pprint(result['candidate_strategies'])


if __name__ == '__main__':
    main()
