"""
ULTIMATE XGBoost Annihilator Training Monitor
Real-time monitoring of the 1.5B parameter beast
"""

import time
import os
import json
import subprocess

def monitor_ultimate_training():
    """Monitor the ULTIMATE training progress"""
    
    print("🔍 MONITORING ULTIMATE XGBOOST ANNIHILATOR")
    print("="*60)
    print("🎯 ULTIMATE TARGETS TO BEAT:")
    print("XGBoost Target: R² > 0.8760")
    print("Model Size: 1.5 BILLION parameters")
    print("Expected Improvement: +15-25% over XGBoost")
    print("="*60)
    
    results_path = '../../results/ultimate_annihilation/annihilation_results.json'
    monitor_count = 0
    
    while True:
        monitor_count += 1
        print(f"\n⏳ Monitor Check #{monitor_count} - {time.strftime('%H:%M:%S')}")
        
        # Check if results file exists
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                print("🏆 ULTIMATE TRAINING COMPLETED!")
                print("="*60)
                
                final_comparison = results.get('final_comparison', {})
                ultimate_results = final_comparison.get('ultimate_xgboost_destroyer', {})
                xgboost_results = final_comparison.get('xgboost', {})
                
                our_r2 = ultimate_results.get('r2', 0)
                xgb_r2 = xgboost_results.get('r2', 0)
                
                print(f"ULTIMATE XGBoost Destroyer: R² = {our_r2:.4f}")
                print(f"XGBoost Target:             R² = {xgb_r2:.4f}")
                
                if results.get('xgboost_annihilated', False):
                    improvement = results.get('improvement_over_xgboost', 0)
                    annihilation_epoch = results.get('annihilation_epoch', 'N/A')
                    print(f"\n🔥🔥🔥 XGBOOST ANNIHILATED! 🔥🔥🔥")
                    print(f"🚀 IMPROVEMENT: +{improvement:.2f}% over XGBoost!")
                    print(f"🏆 ANNIHILATION EPOCH: {annihilation_epoch}")
                    print("✅ ULTIMATE MISSION ACCOMPLISHED!")
                else:
                    gap = xgb_r2 - our_r2
                    print(f"\n❌ Still {gap:.4f} R² points from annihilation")
                    print("🔄 May need even more ULTIMATE optimization...")
                
                print(f"\nModel Parameters: {results.get('model_parameters', 'Unknown'):,}")
                print(f"Training Completed: {results.get('timestamp', 'Unknown')}")
                
                break
                
            except Exception as e:
                print(f"Error reading results: {e}")
        
        # Check if training process is still running
        try:
            result = subprocess.run(['pgrep', '-f', 'ultimate_annihilator'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("🔥 ULTIMATE training still in progress...")
                print("📊 Waiting for the 1.5B parameter beast to complete...")
            else:
                print("⚠️  No ULTIMATE training process found")
                print("Training may have completed or failed")
                # Wait a bit more in case results are being written
                time.sleep(10)
                if not os.path.exists(results_path):
                    print("❌ No results file found - training may have failed")
                    break
        except Exception as e:
            print(f"Error checking process: {e}")
        
        # Wait before next check
        time.sleep(30)  # Check every 30 seconds
        
        # Safety timeout after 2 hours
        if monitor_count > 240:  # 240 * 30 seconds = 2 hours
            print("⏰ Monitor timeout reached (2 hours)")
            print("Training taking longer than expected")
            break

if __name__ == "__main__":
    monitor_ultimate_training()