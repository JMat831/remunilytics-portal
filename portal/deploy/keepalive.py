"""
Keep the Streamlit Community Cloud app awake.

Free-tier apps go to sleep after a period with no visitors, and a prospect who
clicks an outreach link then sees "Zzzz... Yes, get this app back up!" and waits
a minute or more (found 2026-10-23). This visits the app's HOME page with a real
browser -- no portal token, so nothing secret -- clicks the wake button if the
app is asleep, and waits until the app itself has rendered. Exits 1 if it never
loads, so a failure shows up in GitHub Actions.

Run by .github/workflows/keepalive.yml in the deploy repo (every 4 hours).
"""

import os
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL", "https://remunilytics-app-mzqfjqvfdhe3vtmrurykkk.streamlit.app/")
WAKE_BUTTON = "Yes, get this app back up!"


def app_rendered(page) -> bool:
    # The app runs inside an iframe whose URL contains "/~/+/"; it is up when
    # that frame has rendered text.
    for f in page.frames:
        if "/~/+/" in f.url:
            try:
                if len(f.inner_text("body").strip()) > 30:
                    return True
            except Exception:
                pass
    return False


def main(timeout_s: int = 420) -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=120000)
        t0 = time.time()
        woke = False
        while time.time() - t0 < timeout_s:
            if app_rendered(page):
                print(f"app is up ({'woken' if woke else 'was already awake'}; {time.time() - t0:.0f}s)")
                browser.close()
                return 0
            btn = page.get_by_role("button", name=WAKE_BUTTON)
            if not woke and btn.count():
                btn.first.click()
                woke = True
                print("app was asleep -- clicked the wake button")
            time.sleep(5)
        print(f"app did not render within {timeout_s}s", file=sys.stderr)
        page.screenshot(path="keepalive_failure.png")
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
