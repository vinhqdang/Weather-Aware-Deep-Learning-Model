"""
Quick monitoring script to track our training progress
"""

import json
import os
import time

def monitor_progress():
    results_file = 'results/superior_training/superior_results.json'
    
    print("🔍 MONITORING SUPERIOR TRAINING PROGRESS")
    print("="*50)
    
    target_rf = 0.8567
    target_xgb = 0.8760
    target_best = max(target_rf, target_xgb)
    
    print(f"🎯 TARGETS TO BEAT:")
    print(f"Random Forest: R² = {target_rf:.4f}")
    print(f"XGBoost:       R² = {target_xgb:.4f}")
    print(f"Best Target:   R² = {target_best:.4f}")
    print()
    
    while True:
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                our_r2 = results['final_comparison']['weather_aware_stgat']['r2']
                
                print(f"⚡ CURRENT STATUS:")
                print(f"Weather-Aware STGAT: R² = {our_r2:.4f}")
                
                if our_r2 > target_best:
                    print("🔥🔥🔥 SUCCESS! BEAT BOTH BASELINES! 🔥🔥🔥")
                    break
                elif our_r2 > target_rf:
                    print("🎉 BEAT RANDOM FOREST!")
                    if our_r2 > target_xgb:
                        print("🔥🔥🔥 BEAT XGBOOST TOO! 🔥🔥🔥")
                        break
                else:
                    progress = (our_r2 / target_best) * 100
                    print(f"📈 Progress: {progress:.1f}% toward target")
                
                print(f"Gap to beat XGBoost: {target_xgb - our_r2:.4f}")
                print()
                
            except Exception as e:
                print(f"Error reading results: {e}")
        else:
            print("⏳ Waiting for training to complete...")
        
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    monitor_progress()