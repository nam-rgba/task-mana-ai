#!/usr/bin/env python3
"""
Script để test sync endpoint
Chạy: python test_sync.py
"""

import requests
import json

# Configuration
AI_SERVICE_URL = "http://localhost:8000"
BACKEND_URL = "http://localhost:3000"

def test_backend_connection():
    """Kiểm tra kết nối đến backend"""
    print("🔍 Testing backend connection...")
    try:
        response = requests.get(f"{BACKEND_URL}/aidata/tasks", params={"page": 1, "limit": 1}, timeout=5)
        if response.status_code == 200:
            print(f"✅ Backend is reachable: {BACKEND_URL}")
            data = response.json()
            if isinstance(data, dict):
                print(f"   Format: Object with 'data' field")
                print(f"   Sample keys: {list(data.keys())}")
            elif isinstance(data, list):
                print(f"   Format: Array")
                if len(data) > 0:
                    print(f"   Sample task keys: {list(data[0].keys())}")
            return True
        else:
            print(f"❌ Backend returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return False

def test_ai_service_connection():
    """Kiểm tra kết nối đến AI service"""
    print("\n🔍 Testing AI service connection...")
    try:
        response = requests.get(f"{AI_SERVICE_URL}/ai/sync/status", timeout=5)
        if response.status_code == 200:
            print(f"✅ AI service is reachable: {AI_SERVICE_URL}")
            data = response.json()
            print(f"   Status: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ AI service returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to AI service: {e}")
        return False

def run_sync(page_limit=50, max_pages=2):
    """Chạy sync với giới hạn nhỏ để test"""
    print(f"\n🚀 Starting sync (page_limit={page_limit}, max_pages={max_pages})...")
    print("=" * 70)
    
    try:
        response = requests.post(
            f"{AI_SERVICE_URL}/ai/sync/tasks-from-backend",
            params={
                "page_limit": page_limit,
                "max_pages": max_pages,
                "force": True
            },
            timeout=120  # 2 minutes timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Sync completed successfully!")
            print("=" * 70)
            print(json.dumps(data, indent=2))
            print("=" * 70)
            
            stats = data.get("stats", {})
            print(f"\n📊 Summary:")
            print(f"   Total fetched: {stats.get('total_fetched', 0)}")
            print(f"   Total synced: {stats.get('total_synced', 0)}")
            print(f"   Total failed: {stats.get('total_failed', 0)}")
            print(f"   Pages processed: {stats.get('pages_processed', 0)}")
            
            if stats.get('errors'):
                print(f"\n⚠️  Errors ({len(stats['errors'])}):")
                for err in stats['errors'][:5]:  # Show first 5 errors
                    print(f"   - Task {err.get('task_id')}: {err.get('error')}")
            
            return True
        else:
            print(f"❌ Sync failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Sync error: {e}")
        return False

def main():
    print("=" * 70)
    print("🧪 SYNC ENDPOINT TEST")
    print("=" * 70)
    
    # Step 1: Test backend
    if not test_backend_connection():
        print("\n❌ Backend is not available. Please start your Node.js backend first.")
        return
    
    # Step 2: Test AI service
    if not test_ai_service_connection():
        print("\n❌ AI service is not available. Please start AI service first:")
        print("   cd ai_service")
        print("   uvicorn app.main:app --reload --port 8000")
        return
    
    # Step 3: Run sync
    print("\n" + "=" * 70)
    choice = input("Do you want to run sync now? (y/n): ").strip().lower()
    if choice == 'y':
        run_sync(page_limit=50, max_pages=2)
    else:
        print("\n✅ Test completed. You can run sync manually:")
        print(f"   curl -X POST \"{AI_SERVICE_URL}/ai/sync/tasks-from-backend?page_limit=100&force=true\"")
    
    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
