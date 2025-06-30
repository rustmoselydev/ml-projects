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

# --- Configuration ---
download_dir = os.path.abspath("../../raw-data/pixel-art/characters/2")

url = "https://spritedatabase.net/system/genesis"
driver_wait_time = 15

# If it crashes out, set it to where it crashed - 1
# To start at the beginning, set it to 0
resume_iteration = 0

# --- Chrome options for headless download ---
chrome_options = Options()
#chrome_options.add_argument("--headless")
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})

def wait_for_download(download_dir, timeout=5):
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
            if not file.endswith(".crdownload"):
                full_path = os.path.join(download_dir, file)
                # Ensure the download has finished
                if not os.path.exists(full_path + ".crdownload"):
                    return file

    return None

# --- Start WebDriver ---
driver = webdriver.Chrome(service=Service(), options=chrome_options)
driver.get(url)

# --- Click download link ---
try:
    game_links = driver.find_elements(By.CLASS_NAME, "textview")
    i = resume_iteration
    while i < len(game_links):
        try:
            print("Game " + str(i))
            game_links = driver.find_elements(By.CLASS_NAME, "textview")
            time.sleep(0.5)
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", game_links[i])
            WebDriverWait(driver, driver_wait_time).until(
                    EC.element_to_be_clickable(game_links[i]))
            driver.execute_script("arguments[0].click();", game_links[i])
            time.sleep(3)
            sprite_links = driver.find_elements(By.CLASS_NAME, "textview")
            for link in sprite_links:
                sprite_links = driver.find_elements(By.CLASS_NAME, "textview")
                WebDriverWait(driver, driver_wait_time).until(
                    EC.element_to_be_clickable(link)
                )
                driver.execute_script("arguments[0].click();", link)
                time.sleep(3)
                download = driver.find_element(By.CLASS_NAME, "selectedview")
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", download)
                WebDriverWait(driver, driver_wait_time).until(
                    EC.element_to_be_clickable(download))
                driver.execute_script("arguments[0].click();", download)
                downloaded = wait_for_download(download_dir)
                time.sleep(3)
                if downloaded:
                    print(f"Downloaded")
                else:
                    print("Download not found within timeout.")
                driver.back()

            i += 1
            driver.back()
            time.sleep(3)
        # except (TimeoutException, ElementClickInterceptedException, NoSuchElementException) as e:
        #     print(f"[{tag}] Skipping game #{i} due to error: {type(e).__name__} - {str(e)}")
        #     pass
        # except Exception as e:
        #     print(f"[{tag}] Unexpected error on game #{i}: {str(e)}")
        #     pass 
        except Exception as e:
            print(f"ERROR on game #{i}: {type(e).__name__}: {e}")
            traceback.print_exc()
            driver.get(url)
            pass
finally:
    driver.quit()