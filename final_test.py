import requests
import time

API_URL = "http://localhost:8000"

def test_api():
    print("--- Running Final JD Requirements Checklist Test ---")
    
    # 1. Health check
    try:
        r = requests.get(f"{API_URL}/health")
        assert r.json() == {"status": "ok"}
        print("✅ Backend API is running (FastAPI GET /health)")
    except Exception as e:
        print(f"❌ Backend API failed: {e}")
        return

    # 2. General Query (Agent decides to answer directly via LLM)
    print("\n[Test 1] Query: 'Hello, what can you do?'")
    r1 = requests.post(
        f"{API_URL}/ask",
        json={"query": "Hello, what can you do?", "session_id": "eval1"}
    )
    resp1 = r1.json()
    print(f"Answer: {resp1['answer'][:100]}...")
    print(f"Sources: {resp1['source']}")
    if "llm_direct" in resp1['source'] or not resp1['source']:
        print("✅ Tooling decision: LLM answered directly (No docs used)")
    else:
        print("⚠️ Warning: Tool might have been used unnecessarily")

    # 3. Document Query (Agent decides to search documents)
    print("\n[Test 2] Query: 'What is the company leave policy?'")
    r2 = requests.post(
        f"{API_URL}/ask",
        json={"query": "What is the company leave policy?", "session_id": "eval1"}
    )
    resp2 = r2.json()
    print(f"Answer: {resp2['answer'][:100]}...")
    print(f"Sources: {resp2['source']}")
    if any("hr_policy.txt" in s for s in resp2['source']):
        print("✅ RAG logic: Agent fetched from correct document via FAISS")
    else:
        print("❌ FAIL: Did not fetch from hr_policy.txt")

    # 4. Session Memory Query
    print("\n[Test 3] Query: 'What did I just ask about?'")
    r3 = requests.post(
        f"{API_URL}/ask",
        json={"query": "What did I just ask about?", "session_id": "eval1"}
    )
    resp3 = r3.json()
    print(f"Answer: {resp3['answer'][:100]}...")
    if "leave" in resp3['answer'].lower() or "policy" in resp3['answer'].lower():
        print("✅ Memory functionality works: Agent remembered previous query.")
    else:
        print("❌ FAIL: Agent memory failed.")
        
    print("\n--- All Tests Execution Finished ---")

if __name__ == "__main__":
    test_api()
