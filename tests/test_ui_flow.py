import pytest
from playwright.sync_api import Page, expect

def test_reasoning_submission(page: Page):
    """
    Test that the reasoning flow can be INITIATED:
    1. Load Page
    2. Enter Query
    3. Submit
    4. Verify UI enters "processing" state (Button Disabled)
    5. Verify Backend Stream Starts (Trace wrapper appears)
    
    NOTE: We do NOT wait for the final answer because local inference
    is too slow for reasonable test timeouts (>120s). Verification
    of the start of the stream is sufficient for integration testing.
    """
    page.goto("http://localhost:8080/")
    
    input_box = page.locator("#queryInput")
    submit_btn = page.locator("#submitBtn")
    
    # 2. Enter Query
    input_box.fill("What is 2 + 2?")
    
    # 3. Submit
    submit_btn.click()
    
    # Check that button disables (Client logic working)
    expect(submit_btn).to_be_disabled()
    
    # 4. Wait for processing signal (Backend stream valid)
    trace_wrapper = page.locator(".trace-wrapper")
    expect(trace_wrapper).to_be_visible(timeout=10000)
    
    # 5. Success - The loop is running.
