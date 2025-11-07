import re
from playwright.sync_api import Playwright, sync_playwright, expect


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://missav.ai/dm19/ms")
    page.get_by_role("link", name="Melayu").click()
    page.get_by_role("link", name="简体中文 简体中文").dblclick()
    page.screenshot(path="pyout.png")
    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
