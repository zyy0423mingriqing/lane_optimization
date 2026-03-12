import pandas as pd
import numpy as np
from optimizer import LaneOptimizer
import warnings
warnings.filterwarnings('ignore')

def run_optimization_survey():
    optimizer = LaneOptimizer()
    L_MAX = 4
    N_LANES = 4
    step = 0.05
    
    results = []
    
    for p_cav in np.arange(0.0, 1.05, step):
        for p_chv in np.arange(0.0, 1.05 - p_cav, step):
            for p_av in np.arange(0.0, 1.05 - p_cav - p_chv, step):
                p_hv = 1.0 - p_cav - p_chv - p_av
                if p_hv < -0.01:
                    continue
                p_hv = max(p_hv, 0.0)
                
                automation = p_cav + p_av
                connectivity = p_cav + p_chv
                
                try:
                    result = optimizer.optimize(
                        P_CAV=p_cav,
                        P_CHV=p_chv,
                        P_AV=p_av,
                        P_HV=p_hv,
                        n=N_LANES,
                        L_max=L_MAX,
                        verbose=False,
                    )
                    
                    optimal_capacity = result['best_result']['total_capacity']
                    gl_capacity = result['baseline']['best_origin']
                    
                    if gl_capacity > 0:
                        improvement_pct = (optimal_capacity - gl_capacity) / gl_capacity * 100
                        improvement_abs = optimal_capacity - gl_capacity
                    else:
                        improvement_pct = 0
                        improvement_abs = 0
                    
                    results.append({
                        'P_CAV': p_cav,
                        'P_CHV': p_chv,
                        'P_AV': p_av,
                        'P_HV': p_hv,
                        'Automation': int(automation * 100),
                        'Connectivity': int(connectivity * 100),
                        'Strategy': result['best_result']['strategy'],
                        'Lane_Allocation': str(result['best_result']['lane_allocation']),
                        'Optimal_Capacity': optimal_capacity,
                        'GL_Capacity': gl_capacity,
                        'Improvement_Abs': improvement_abs,
                        'Improvement_Pct': improvement_pct,
                    })
                    
                except Exception as e:
                    print(f"错误: p_cav={p_cav}, p_chv={p_chv}, p_av={p_av}: {e}")
                    continue
        
        print(f"  p_cav={p_cav:.2f} done")
    
    df = pd.DataFrame(results)
    
    output_path_all = './plot1_strategy/improvement_data_all.csv'
    df.to_csv(output_path_all, index=False)
    print(f"\n所有数据已保存到: {output_path_all}")
    print(f"总共 {len(df)} 条记录")
    
    grouped = df.groupby(['Automation', 'Connectivity']).agg({
        'Improvement_Pct': ['mean', 'max', 'std', 'count'],
        'Improvement_Abs': ['mean', 'max'],
        'Optimal_Capacity': ['mean', 'max'],
        'GL_Capacity': ['mean', 'max'],
    }).reset_index()
    
    grouped.columns = ['Automation', 'Connectivity', 
                      'Improvement_Pct_Mean', 'Improvement_Pct_Max', 'Improvement_Pct_Std', 'Sample_Count',
                      'Improvement_Abs_Mean', 'Improvement_Abs_Max',
                      'Optimal_Capacity_Mean', 'Optimal_Capacity_Max',
                      'GL_Capacity_Mean', 'GL_Capacity_Max']
    
    output_path_max = './plot1_strategy/improvement_data_max.csv'
    df_max = grouped[['Automation', 'Connectivity', 'Improvement_Pct_Max', 'Improvement_Abs_Max', 
                      'Optimal_Capacity_Max', 'GL_Capacity_Max', 'Sample_Count']].copy()
    df_max.columns = ['Automation', 'Connectivity', 'Improvement_Pct', 'Improvement_Abs', 
                      'Optimal_Capacity', 'GL_Capacity', 'Sample_Count']
    df_max.to_csv(output_path_max, index=False)
    print(f"最大值数据已保存到: {output_path_max}")
    
    output_path_mean = './plot1_strategy/improvement_data_mean.csv'
    df_mean = grouped[['Automation', 'Connectivity', 'Improvement_Pct_Mean', 'Improvement_Abs_Mean',
                       'Optimal_Capacity_Mean', 'GL_Capacity_Mean', 'Sample_Count']].copy()
    df_mean.columns = ['Automation', 'Connectivity', 'Improvement_Pct', 'Improvement_Abs',
                       'Optimal_Capacity', 'GL_Capacity', 'Sample_Count']
    df_mean.to_csv(output_path_mean, index=False)
    print(f"平均值数据已保存到: {output_path_mean}")
    
    return df, df_max, df_mean

if __name__ == '__main__':
    df_all, df_max, df_mean = run_optimization_survey()
