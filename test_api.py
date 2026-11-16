"""
Simple API test with httpx.

  python test_api.py                                  -> tests the local server (http://127.0.0.1:8000)
  python test_api.py https://YOUR-SPACE.hf.space      -> tests the live Hugging Face Space
"""
import sys

import httpx

BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")

SAMPLE_TICKETS = [
    "e-poçtuma daxil olmaqda çətinlik çəkirəm. hər dəfə daxil olmaq istəyəndə xəta verir",
    "veb saytım çox yavaş yüklənir və bunun səbəbini bilmirəm",
    "men coxlu spam mekdublar aliram ve onlari neje dayandirajaghimi bilmirem",
]

failed = False
print(f"Testing {BASE_URL}\n")

with httpx.Client(timeout=60) as client:
    # 1. health check
    r = client.get(f"{BASE_URL}/health")
    print("GET /health ->", r.status_code, r.json())
    failed |= r.status_code != 200

    # 2. normal predictions
    for text in SAMPLE_TICKETS:
        r = client.post(f"{BASE_URL}/predict", json={"description": text})
        print(f"\nPOST /predict -> {r.status_code}")
        print("  ticket :", text[:70])
        print("  answer :", r.json())
        failed |= r.status_code != 200

    # 3. bad input must give a clean 422 error, not a crash
    r = client.post(f"{BASE_URL}/predict", json={"description": "   "})
    print("\nBlank ticket ->", r.status_code, r.json())
    failed |= r.status_code != 422

    r = client.post(f"{BASE_URL}/predict", json={"wrong_field": "hello"})
    print("Missing field ->", r.status_code, r.json())
    failed |= r.status_code != 422

print("\nRESULT:", "SOME TESTS FAILED" if failed else "ALL TESTS PASSED")
sys.exit(1 if failed else 0)
