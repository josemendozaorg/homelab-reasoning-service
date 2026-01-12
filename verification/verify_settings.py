from playwright.sync_api import sync_playwright

def verify_settings():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("Navigating to app...")
        page.goto("http://localhost:8000")

        # 1. Click Settings Button
        print("Opening Settings...")
        page.click("#settingsBtn")

        # 2. Check Provider Dropdown
        print("Checking Dropdown Options...")
        dropdown = page.locator("#searchProvider")
        options = dropdown.locator("option").all_inner_texts()
        print(f"Options found: {options}")

        if "Auto (Smart Router)" not in options:
            print("ERROR: Auto option missing!")
        if "Exa (Semantic/Neural)" not in options:
            print("ERROR: Exa option missing!")

        # 3. Select Exa and check config visibility
        print("Selecting Exa...")
        dropdown.select_option("exa")

        exa_config = page.locator("#exaConfig")
        if not exa_config.is_visible():
            print("ERROR: Exa config not visible after selection!")
        else:
            print("SUCCESS: Exa config visible.")

        # 4. Screenshot
        page.screenshot(path="verification/settings_modal.png")
        print("Screenshot saved to verification/settings_modal.png")

        browser.close()

if __name__ == "__main__":
    verify_settings()
