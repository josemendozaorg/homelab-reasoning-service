from playwright.sync_api import Page, expect

def test_homepage_loads(page: Page):
    """Verify homepage loads and has correct title."""
    page.goto("http://localhost:8080/")
    
    # Check title
    expect(page).to_have_title("DeepSeek Reasoning Service")
    
    # Check Header
    header = page.locator("h1")
    expect(header).to_contain_text("Reasoning Service")

def test_input_elements(page: Page):
    """Verify input form elements are present."""
    page.goto("http://localhost:8080/")
    
    # Check input area
    input_box = page.locator("#queryInput")
    expect(input_box).to_be_visible()
    expect(input_box).to_be_enabled()
    
    # Check submit button (initially disabled or enabled depending on input)
    submit_btn = page.locator("#submitBtn")
    expect(submit_btn).to_be_visible()

def test_interaction_enables_button(page: Page):
    """Verify typing enables the submit button."""
    page.goto("http://localhost:8080/")
    
    input_box = page.locator("#queryInput")
    submit_btn = page.locator("#submitBtn")
    
    # Initially disabled (per app.js logic)
    expect(submit_btn).to_be_disabled()
    
    # Type something
    input_box.fill("Hello World")
    
    # Should be enabled
    expect(submit_btn).to_be_enabled()
