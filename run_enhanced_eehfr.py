"""
运行Enhanced EEHFR WSN系统
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_eehfr_wsn_system import EnhancedEEHFRSystem

def main():
    print("🚀 启动Enhanced EEHFR WSN智能路由系统")
    print("📋 系统特点:")
    print("   ✅ 基于Intel Berkeley Lab真实数据集")
    print("   ✅ EEHFR协议：融合模糊逻辑与混合元启发式优化")
    print("   ✅ PSO+ACO混合路由优化算法")
    print("   ✅ LSTM时序预测模块")
    print("   ✅ 多维信任评估机制")
    print("   ✅ 能耗-精度-可靠性联合优化")
    print("   ✅ 完整的性能评估和可视化")
    print()
    
    # 数据路径
    data_path = "D:/lkr_wsn/WSN-Intel-Lab-Project/data/data.txt"
    
    try:
        # 创建并运行系统
        eehfr_system = EnhancedEEHFRSystem(data_path)
        performance, lstm_metrics, overall_score = eehfr_system.run_complete_system()
        
        print(f"\n🎉 系统运行成功！综合评分: {overall_score:.3f}")
        
    except Exception as e:
        print(f"❌ 系统运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()