"""Step 6: screenshots of the running app for the video (needs app.py running on PORT, default 7860).

    python 06_screenshots.py

results/app_answer.png   Staff asks the winter-crop question
results/app_access.png   Public visitor vs Executive on a restricted-document question (side by side)
"""
import os
import time

from PIL import Image
from playwright.sync_api import sync_playwright

from common import RESULTS

URL = f"http://127.0.0.1:{os.environ.get('PORT', 7860)}"


def ask(page, role, question, wait_s=120):
    page.goto(URL)
    page.wait_for_selector("textarea", timeout=60000)
    dd = page.get_by_label("Signed in as (access level)")
    dd.click()
    page.get_by_role("option", name=role).click()
    box = page.get_by_placeholder("Ask a question about the reports...")
    box.fill(question)
    box.press("Enter")
    # wait until the meta line (written when the answer finishes) appears
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if "total" in (page.locator("#meta").inner_text() or ""):
            break
        time.sleep(1)
    time.sleep(1.5)


def main():
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        page = b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        ask(page, "Staff", "What is ABARES forecasting for Australia's winter crop production in 2026-27, "
                           "and how does it compare with the 10-year average?")
        page.screenshot(path=os.path.join(RESULTS, "app_answer.png"))

        q = "How much has winter rainfall declined in south-western Australia since 1970?"
        ask(page, "Public visitor", q)
        page.screenshot(path=os.path.join(RESULTS, "_access_public.png"))
        ask(page, "Executive", q)
        page.screenshot(path=os.path.join(RESULTS, "_access_exec.png"))
        b.close()

    # side by side: left = public visitor, right = executive (crop the chat + sources area)
    box = (292, 120, 1462, 1000)                      # chat column + sources column
    a = Image.open(os.path.join(RESULTS, "_access_public.png")).crop(box)
    c = Image.open(os.path.join(RESULTS, "_access_exec.png")).crop(box)
    a, c = a.resize((940, int(940 * a.height / a.width))), c.resize((940, int(940 * c.height / c.width)))
    sheet = Image.new("RGB", (1920, 1080), (11, 18, 32))
    sheet.paste(a, (10, 140))
    sheet.paste(c, (970, 140))
    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 40)
    d.text((40, 50), "Signed in as: Public visitor", font=f, fill=(248, 113, 113))
    d.text((1000, 50), "Signed in as: Executive", font=f, fill=(74, 222, 128))
    d.line((960, 30, 960, 1060), fill=(56, 189, 248), width=3)
    sheet.save(os.path.join(RESULTS, "app_access.png"))
    print("screenshots written")


if __name__ == "__main__":
    main()
