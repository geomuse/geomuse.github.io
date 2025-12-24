from playwright.sync_api import sync_playwright
import time 
with sync_playwright() as p:
    # 启动 Firefox
    browser = p.firefox.launch(headless=True)  # headless=True 表示无头模式
    page = browser.new_page()
    # 打开网页
    page.goto("https://www.wikipedia.org")
    # 输入搜索关键字并提交
    page.fill("input[name='search']", "Python_(programming_language)")
    page.press("input[name='search']", "Enter")
    # 等待页面加载完成
    page.wait_for_selector("p")

    paragraphs = page.query_selector_all("p")

    # for i, p_tag in enumerate(paragraphs, 0):
    for p_tag in paragraphs : 
        text = p_tag.inner_text().strip() # 取得 HTML 元素 里面可见的文字
        if text:
            print(f"{text}\n")

    browser.close()