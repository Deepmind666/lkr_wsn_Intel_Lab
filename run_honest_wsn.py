"""
运行诚实的WSN系统 - 无虚假宣传
位置：WSN-Intel-Lab-Project/
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🚀 启动诚实的WSN路由系统")
    print("⚠️  重要声明：")
    print("   - 这是仿真系统，不是真实硬件")
    print("   - GAT只做特征重构，不是复杂图学习")
    print("   - LSTM使用模拟数据，不是真实传感器数据")
    print("   - 路由算法是简化模型，不是完整协议实现")
    print("   - 性能指标仅供参考，不代表真实网络表现")
    print()
    
    # 正确的项目路径
    project_dir = Path(__file__).parent
    script_path = project_dir / "src" / "advanced_algorithms" / "honest_wsn_routing.py"
    
    print(f"📁 项目目录: {project_dir}")
    print(f"📄 脚本路径: {script_path}")
    
    if not script_path.exists():
        print(f"❌ 脚本文件不存在: {script_path}")
        return
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print("📊 执行输出:")
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("⚠️ 错误信息:")
            print(result.stderr)
        
        print(f"✅ 执行完成，返回码: {result.returncode}")
        
        # 检查结果文件
        results_dir = project_dir / "results" / "honest_wsn"
        if results_dir.exists():
            print("\n📋 生成的文件:")
            for file in results_dir.glob("*"):
                print(f"  ✅ {file.name}")
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")

if __name__ == "__main__":
    main()