#!/usr/bin/env python3
"""Quick validation - skips slow training tests"""

import subprocess
import requests
import sys
from pathlib import Path

def main():
    print("\n" + "="*60)
    print("QUICK VALIDATION")
    print("="*60)
    
    results = []
    
    # Check 1: Health
    print("\n[CHECK 1/4] /health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("✅ PASS: /health returns 200")
            results.append(True)
        else:
            print(f"❌ FAIL: /health returns {response.status_code}")
            results.append(False)
    except Exception as e:
        print(f"❌ FAIL: {e}")
        results.append(False)
    
    # Check 2: Reset
    print("\n[CHECK 2/4] /reset endpoint...")
    try:
        response = requests.post(
            "http://localhost:8000/reset",
            params={"difficulty": "easy"},
            timeout=30
        )
        if response.status_code == 200 and "env_id" in response.json():
            print("✅ PASS: /reset works")
            results.append(True)
        else:
            print("❌ FAIL: /reset invalid")
            results.append(False)
    except Exception as e:
        print(f"❌ FAIL: {e}")
        results.append(False)
    
    # Check 3: inference.py format
    print("\n[CHECK 3/4] inference.py format...")
    try:
        with open("inference.py", "r") as f:
            content = f.read()
        if "[START]" in content and "[STEP]" in content and "[END]" in content:
            print("✅ PASS: inference.py format correct")
            results.append(True)
        else:
            print("❌ FAIL: Missing required format")
            results.append(False)
    except Exception as e:
        print(f"❌ FAIL: {e}")
        results.append(False)
    
    # Check 4: openenv.yaml
    print("\n[CHECK 4/4] openenv.yaml exists...")
    if Path("openenv.yaml").exists():
        print("✅ PASS: openenv.yaml exists")
        results.append(True)
    else:
        print("❌ FAIL: openenv.yaml missing")
        results.append(False)
    
    # Summary
    passed = sum(results)
    print("\n" + "="*60)
    print(f"PASSED: {passed}/4")
    print("="*60)
    
    if passed == 4:
        print("\n🎉 ALL CHECKS PASSED! Ready for submission!\n")
        return 0
    else:
        print(f"\n❌ {4-passed} checks failed\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())