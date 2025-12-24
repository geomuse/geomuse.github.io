import re
from playwright.sync_api import Playwright, sync_playwright, expect

def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://www.sinchew.com.my/")
    page.get_by_role("link", name="国际", exact=True).click()
    links = page.query_selector_all("div.title")
    print(f"抓到 {len(links)} 条链接\n")
    for i, el in enumerate(links):
        a = el.query_selector("a")
        if not a : 
            continue
        print(el.inner_text(),a.get_attribute('href'))

    # ---------------------
    context.close()
    browser.close()

with sync_playwright() as playwright:
    run(playwright)