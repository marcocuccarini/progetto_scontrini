import time
import subprocess
import os

# === CONFIG ===
INTERVAL_HOURS = 12

# Two different API environments
APIS = [
    {
        "API_KEY": "FIRST_API_KEY",
        "MODEL_NAME": "gemini-2.0-pro",
        "ENV": "AIzaSyBymZe7dunVj3j76DqA9q6VuDgimwYmryA"
    },
    {
        "API_KEY": "SECOND_API_KEY",
        "MODEL_NAME": "gemini-2.0-pro",
        "ENV": "AIzaSyBisxgy9D6A-EOAD5HulVouMIU56qLROe8"
    }
]

def run_processing(api_cfg):
    print(f"\n🚀 Running processor with {api_cfg['ENV']} ({api_cfg['MODEL_NAME']})...\n")

    env = os.environ.copy()
    env["GEMINI_API_KEY"] = api_cfg["API_KEY"]
    env["MODEL_NAME"] = api_cfg["MODEL_NAME"]

    subprocess.run(
        ["python", "process_receipts.py"],
        env=env
    )

while True:
    for api in APIS:
        run_processing(api)

    print(f"⏳ Waiting {INTERVAL_HOURS} hours before next run...")
    time.sleep(INTERVAL_HOURS * 3600)
