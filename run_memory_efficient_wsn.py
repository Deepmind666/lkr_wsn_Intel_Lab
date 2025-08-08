"""
运行内存优化的WSN系统
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("🚀 内存优化的真实数据WSN路由系统")
    print("=" * 60)
    
    print("\n📋 系统特点:")
    print("✅ 使用真实Intel Berkeley数据集")
    print("✅ 内存优化：数据采样 + 批处理")
    print("✅ 限制节点数量和样本数量")
    print("✅ 真实的LSTM训练和预测")
    print("✅ 诚实的性能评估")
    
    print("\n🔧 优化措施:")
    print("• 数据采样比例: 10%")
    print("• 最大节点数: 20个")
    print("• 每节点最大样本: 1000条")
    print("• 批处理大小: 512")
    print("• 最大数据量: 50万条记录")
    
    print("\n" + "=" * 60)
    
    try:
        # 切换到项目目录
        project_dir = Path(__file__).parent
        os.chdir(project_dir)
        
        # 运行内存优化的WSN系统
        script_path = "src/advanced_algorithms/memory_efficient_wsn.py"
        
        print(f"🔄 执行: python {script_path}")
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              text=True,
                              timeout=600)  # 10分钟超时
        
        if result.returncode == 0:
            print("\n✅ 内存优化WSN系统执行成功!")
            
            # 检查结果文件
            results_dir = project_dir / "results" / "memory_efficient_wsn"
            expected_files = [
                "memory_efficient_results.png",
                "memory_efficient_metrics.json",
                "memory_efficient_training_history.json",
                "memory_efficient_lstm_model.pth"
            ]
            
            print("\n📁 生成的文件:")
            for file_name in expected_files:
                file_path = results_dir / file_name
                if file_path.exists():
                    print(f"✅ {file_name}")
                else:
                    print(f"❌ {file_name} (未找到)")
        else:
            print(f"\n❌ 执行失败，返回码: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("\n⏰ 执行超时 (10分钟)")
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")

if __name__ == "__main__":
    main()