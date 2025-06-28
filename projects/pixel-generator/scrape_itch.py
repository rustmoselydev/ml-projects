# This needs some work. Itch has so many possible page types that makes it pretty annoying to scrape

import os
import time
import shutil
import traceback
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException, NoSuchElementException

asset_types = ["characters"]
# --- Configuration ---
download_dir = os.path.abspath("../../raw-data/pixel-art")

url = "https://itch.io/game-assets/free/tag-$/tag-pixel-art"
download_link_text = "Download"
driver_wait_time = 30

# --- Chrome options for headless download ---
chrome_options = Options()
#chrome_options.add_argument("--headless")
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})

def wait_for_download(download_dir, timeout=driver_wait_time, extensions=(".png", ".zip", ".rar")):
    """
    Waits for a new file with a given extension to appear and finish downloading.
    Skips partial or temp files (e.g., .crdownload).
    Returns the full path to the downloaded file, or None if timed out.
    """
    start_time = time.time()
    already_existing = set(os.listdir(download_dir))  # To avoid grabbing leftovers

    while time.time() - start_time < timeout:
        time.sleep(1)
        current_files = set(os.listdir(download_dir))
        new_files = current_files - already_existing

        for file in new_files:
            if file.endswith(extensions) and not file.endswith(".crdownload"):
                full_path = os.path.join(download_dir, file)
                # Ensure the download has finished
                if not os.path.exists(full_path + ".crdownload"):
                    return file

    return None

def scroll_to_latest(driver):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    return driver.find_elements(By.CLASS_NAME, "game_link")

for tag in asset_types:
    # --- Start WebDriver ---
    driver = webdriver.Chrome(service=Service(), options=chrome_options)
    driver.get(url.replace('$', tag))

    try:
        driver.execute_cdp_cmd(
            "Page.setDownloadBehavior",
            {
                "behavior": "allow",
                "downloadPath": download_dir
            }
        )
        game_links = driver.find_elements(By.CLASS_NAME, "game_link")
        i = 0
        while i < len(game_links):
            try:
                game_links = driver.find_elements(By.CLASS_NAME, "game_link")
                while i > len(game_links):
                    scroll_to_latest(driver)
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", game_links[i])
                WebDriverWait(driver, driver_wait_time).until(
                        EC.element_to_be_clickable(game_links[i])
                    )
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", game_links[i])
                time.sleep(3)
                buy_btn_info = None, None
                is_download_btn = False
                try:
                    btn_type = None
                    buy_btn = None

                    try:
                        buy_btn = WebDriverWait(driver, driver_wait_time).until(
                            EC.presence_of_element_located((By.CLASS_NAME, "buy_btn"))
                        )
                        if buy_btn.is_displayed() and buy_btn.is_enabled():
                            btn_type = "buy"
                    except TimeoutException:
                        pass

                    if btn_type is None:
                        try:
                            buy_btn = WebDriverWait(driver, driver_wait_time).until(
                                EC.presence_of_element_located((By.CLASS_NAME, "download_btn"))
                            )
                            if buy_btn.is_displayed() and buy_btn.is_enabled():
                                btn_type = "download"
                        except TimeoutException:
                            pass

                    if not btn_type or not buy_btn:
                        raise TimeoutException("Neither 'buy_btn' nor 'download_btn' were found or clickable.")
                    else:
                        buy_btn_info = btn_type, buy_btn
                except TimeoutException:
                    buy_btn_info =  None, None
                    
                btn_type, buy_btn = buy_btn_info
                    
                if btn_type == "buy":
                    buy_btn = WebDriverWait(driver, driver_wait_time).until(
                        EC.element_to_be_clickable((By.CLASS_NAME, "buy_btn"))
                    )
                elif btn_type == "download":
                    buy_btn = WebDriverWait(driver, driver_wait_time).until(
                        EC.element_to_be_clickable((By.CLASS_NAME, "download_btn"))
                    )
                    is_download_btn = True
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", buy_btn)
                original_window = driver.current_window_handle
                current_window = original_window
                if is_download_btn == False:
                    WebDriverWait(driver, driver_wait_time).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "direct_download_btn"))
                    )
                    direct_download_btn = WebDriverWait(driver, driver_wait_time).until(
                    EC.element_to_be_clickable(driver.find_element(By.CLASS_NAME, "direct_download_btn")))
                    driver.execute_script("arguments[0].click();", direct_download_btn)
                    # Check if it opened a new tab, if so, switch to that tab
                    time.sleep(3)
                    if len(driver.window_handles) > 1:
                        new_window = [w for w in driver.window_handles if w != original_window][0]
                        driver.switch_to.window(new_window)
                        current_window = new_window
                    WebDriverWait(driver, driver_wait_time).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "download_btn"))
                    )
                    download_btn = WebDriverWait(driver, driver_wait_time).until(
                        EC.element_to_be_clickable(driver.find_elements(By.CLASS_NAME, "download_btn")[0]))
                    driver.execute_script("arguments[0].click();", download_btn)
                # --- Wait for download to complete ---
                downloaded = wait_for_download(download_dir)

                i += 2

                if downloaded:
                    
                    print(f"Downloaded")
                else:
                    print("Download not found within timeout.")
                if original_window != current_window:
                    # Close tab and switch back
                    driver.close()
                    driver.switch_to.window(original_window)
                else:
                    driver.back()
                
                driver.back()
                time.sleep(3)
                print("completing cycle")
                while (i >= len(game_links)):
                    scroll_to_latest(driver)
            except (TimeoutException, ElementClickInterceptedException, NoSuchElementException) as e:
                print(f"[{tag}] Skipping game #{i} due to error: {type(e).__name__} - {str(e)}")
                driver.get(url.replace('$', tag))
                pass
            except Exception as e:
                print(f"[{tag}] Unexpected error on game #{i}: {str(e)}")
                driver.get(url.replace('$', tag))
                pass 
            # except Exception as e:
            #     print(f"[{tag}] ERROR on game #{i}: {type(e).__name__}: {e}")
            #     traceback.print_exc()
            #     break
    finally:
        driver.quit()