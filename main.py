import asyncio
import json
import os
import re
import shutil
import sys
import requests
import aiohttp
import argparse
import cv2
import time

import numpy as np

from paddleocr import PaddleOCR
from PIL import Image

from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.pdfmetrics import stringWidth
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte


from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Set the appropriate event loop policy for Windows platforms
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def download_image(session, url, page_number, max_retries=None, cooldown=5):
    """
    Asynchronously downloads an image from a modified URL and saves it locally.
    Retries indefinitely or up to `max_retries` times if the download fails.

    Args:
        session (aiohttp.ClientSession): The HTTP session to use for the request.
        url (str): The base URL to modify and download the image from.
        page_number (int): The page number to replace in the URL.
        max_retries (int, optional): Maximum number of retries. If None, retries indefinitely.
        cooldown (int, optional): Seconds to wait before retrying after a failure.
    """
    modified_url = url.replace("page=0", f"page={page_number}")
    final_url = re.sub(r'w=\d+', 'w=3200', modified_url)
    
    attempt = 0
    while True:
        try:
            async with session.get(final_url) as response:
                if response.status == 200:
                    content = await response.read()
                    # Ensure the directory exists
                    os.makedirs("images", exist_ok=True)
                    file_path = f"images/image_page_{page_number}.jpg"
                    with open(file_path, "wb") as f:
                        f.write(content)
                    break  # Exit the loop on success
                else:
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [WARNING] Failed to download image from page {page_number}, status: {response.status}")
        except aiohttp.ClientError as e:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [ERROR] Client error while downloading image from page {page_number}: {e}")
        except Exception as e:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [ERROR] Unexpected error while downloading image from page {page_number}: {e}")
        
        attempt += 1
        if max_retries is not None and attempt >= max_retries:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [ERROR] Exceeded maximum retries for image page {page_number}")
            break  # Exit the loop after reaching max retries
        
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [INFO] Retrying image download for page {page_number} after {cooldown} seconds...")
        await asyncio.sleep(cooldown)

def replace_romanian_special_chars(text):
    """
    Replaces Romanian special characters with their base Latin equivalents.
    
    Args:
        text (str): The input string containing Romanian characters.
        
    Returns:
        str: The modified string with special characters replaced.
    """
    # Define a translation table
    translation_table = str.maketrans({
        'Ă': 'A', 'ă': 'a',
        'Â': 'A', 'â': 'a',
        'Î': 'I', 'î': 'i',
        'Ș': 'S', 'ș': 's',
        'Ş': 'S', 'ş': 's',
        'Ț': 'T', 'ț': 't',
        'Ţ': 'T', 'ţ': 't',
    })
    
    # Replace characters using the translation table
    return text.translate(translation_table)

async def download_text(session, url, page_number, max_retries=None, cooldown=5):
    """
    Asynchronously downloads a text file from a modified URL, processes it, and saves it locally.
    Retries indefinitely or up to `max_retries` times if the download fails.

    Args:
        session (aiohttp.ClientSession): The HTTP session to use for the request.
        url (str): The base URL to modify and download the text from.
        page_number (int): The page number to replace in the URL.
        max_retries (int, optional): Maximum number of retries. If None, retries indefinitely.
        cooldown (int, optional): Seconds to wait before retrying after a failure.
    """
    modified_url = url.replace("page=0", f"page={page_number}")
    
    attempt = 0
    while True:
        try:
            async with session.get(modified_url) as response:
                if response.status == 200:
                    content = await response.text()
                    lines = content.splitlines()
                    if len(lines) > 1:
                        content_without_first_line = "\n".join(lines[1:])
                    else:
                        content_without_first_line = ""
                        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [WARNING] Text content from page {page_number} has only one line.")
                    
                    # Replace Romanian special characters
                    cleared_content = replace_romanian_special_chars(content_without_first_line)

                    # Ensure the directory exists
                    os.makedirs("texts", exist_ok=True)
                    file_path = f"texts/text_page_{page_number}.txt"
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(cleared_content)
                    
                    break 
        except aiohttp.ClientError as e:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [ERROR] Client error while downloading text from page {page_number}: {e}")
        except Exception as e:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [ERROR] Unexpected error while downloading text from page {page_number}: {e}")
        
        attempt += 1
        if max_retries is not None and attempt >= max_retries:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [ERROR] Exceeded maximum retries for text page {page_number}")
            break  # Exit the loop after reaching max retries
        
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [INFO] Retrying text download for page {page_number} after {cooldown} seconds...")
        await asyncio.sleep(cooldown)

async def download_content(num_pages, img_default_link, json_default_link, max_retries=None, cooldown=5):
    """
    Orchestrates the asynchronous downloading of images and text files for all pages.
    Retries each download until successful or until `max_retries` is reached.

    Args:
        num_pages (int): The total number of pages to download.
        img_default_link (str): The base URL for downloading images.
        json_default_link (str): The base URL for downloading text files.
        max_retries (int, optional): Maximum number of retries for each download. If None, retries indefinitely.
        cooldown (int, optional): Seconds to wait before retrying after a failure.
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_pages):
            tasks.append(download_image(session, img_default_link, i, max_retries, cooldown))
            tasks.append(download_text(session, json_default_link, i, max_retries, cooldown))
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks)


def process_image(input_image_path, thickness=1):
    """
    Processes the input image by extracting the table structure, thickening the lines,
    and thinning them to one pixel width.

    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the processed image.
        thickness (int): Thickness for line dilation. Defaults to 1.
    """
    # Load the image
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read the image at path: {input_image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(~gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, -2)

    # Detect horizontal lines
    horizontal = binary.copy()
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)

    # Detect vertical lines
    vertical = binary.copy()
    rows = vertical.shape[0]
    vertical_size = rows // 30
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)

    # Combine horizontal and vertical lines
    table_structure = cv2.add(horizontal, vertical)

    # Convert to binary format suitable for skeletonization
    _, binary = cv2.threshold(table_structure, 127, 1, cv2.THRESH_BINARY)
    binary_bool = binary.astype(bool)

    # Apply skeletonization to thin the lines to one pixel width
    skeleton = skeletonize(binary_bool)
    skeleton_uint8 = img_as_ubyte(skeleton)

    # Save the processed (thinned) image
    success = cv2.imwrite(input_image_path, skeleton_uint8)
    if not success:
        raise IOError(f"Could not write the image to path: {input_image_path}")

def add_image_to_existing_pdf(image_path, pdf_canvas):
    """
    Processes an image and draws its structure onto an existing PDF canvas.

    Args:
        image_path (str): Path to the processed image.
        pdf_canvas (canvas.Canvas): Existing ReportLab canvas object to draw on.
        text_data_path (str, optional): Path to the JSON text data file for overlaying text. Defaults to None.

    Returns:
        canvas.Canvas: The modified PDF canvas object with the image structure drawn on it.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read the image at path: {image_path}")

    # Threshold the image to make it binary (white lines on black background)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Get the dimensions of the existing PDF canvas
    width, height = pdf_canvas._pagesize

    # Define the scale factor to convert image coordinates to PDF coordinates
    scale_x = width / binary_image.shape[1]
    scale_y = height / binary_image.shape[0]

    # List to store lines to be drawn in the PDF
    lines_to_draw = []

    # Draw horizontal lines by detecting contiguous segments using numpy
    for y in range(binary_image.shape[0]):
        white_pixels = np.where(binary_image[y] == 255)[0]
        if len(white_pixels) == 0:
            continue

        diffs = np.diff(white_pixels)
        segments = np.split(white_pixels, np.where(diffs > 1)[0] + 1)

        for segment in segments:
            if len(segment) > 1:
                x_start = segment[0]
                x_end = segment[-1]
                lines_to_draw.append(
                    (x_start * scale_x, height - y * scale_y, x_end * scale_x, height - y * scale_y)
                )

    # Draw vertical lines by detecting contiguous segments using numpy
    for x in range(binary_image.shape[1]):
        white_pixels = np.where(binary_image[:, x] == 255)[0]
        if len(white_pixels) == 0:
            continue

        diffs = np.diff(white_pixels)
        segments = np.split(white_pixels, np.where(diffs > 1)[0] + 1)

        for segment in segments:
            if len(segment) > 1:
                y_start = segment[0]
                y_end = segment[-1]
                lines_to_draw.append(
                    (x * scale_x, height - y_start * scale_y, x * scale_x, height - y_end * scale_y)
                )

    # Draw all collected lines in the PDF
    pdf_canvas.setStrokeColorRGB(0, 0, 0)
    pdf_canvas.setLineWidth(0.1)
    for line in lines_to_draw:
        pdf_canvas.line(*line)



def calculate_font_size(text, box_width, box_height, font_name, max_font_size=100, min_font_size=6):
    """
    Calculate the maximum font size that allows the text to fit within the bounding box.

    Args:
        text (str): The text to fit.
        box_width (float): The width of the bounding box.
        box_height (float): The height of the bounding box.
        font_name (str): The name of the font.
        max_font_size (int, optional): The starting font size to try. Defaults to 100.
        min_font_size (int, optional): The minimum font size allowed. Defaults to 6.

    Returns:
        int: The optimal font size.
    """
    if not text:
        return min_font_size

    for font_size in range(max_font_size, min_font_size - 1, -1):
        text_width = stringWidth(text, font_name, font_size)
        # Assuming that the font's ascent and descent roughly fit within box_height
        if text_width <= box_width and font_size <= box_height:
            return font_size
    return min_font_size


def process_page_with_ocr(image_path, c):
    """
    Creates a PDF page from an image by performing OCR to extract text,
    including URL detection and link annotations.

    Args:
        image_path (str): The file path to the input image.
        c (canvas.Canvas): The ReportLab canvas object where the content is added.
    """
    # Initialize PaddleOCR with angle classification and English language support
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

    # Perform OCR on the image to extract text and their bounding boxes
    result = ocr_model.ocr(image_path, cls=True)
    text_positions = []
    for res in result:
        for line in res:
            bbox = line[0]  # Bounding box coordinates
            text = line[1][0]  # Extracted text
            text_positions.append((bbox, text))

    # Define the PDF page size
    page_width, page_height = landscape(A4)

    # Open the image to get its dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size

    process_image(image_path)

    # Draw the image onto the PDF
    add_image_to_existing_pdf(image_path, c)

    # Initialize text object for invisible text overlay
    text_writer = c.beginText()

    # Calculate scaling factors based on image and page dimensions
    img_aspect = img_width / img_height
    page_aspect = page_width / page_height

    if img_aspect > page_aspect:
        scaled_width = page_width
        scaled_height = page_width / img_aspect
    else:
        scaled_height = page_height
        scaled_width = page_height * img_aspect

    x_offset = (page_width - scaled_width) / 2
    y_offset = (page_height - scaled_height) / 2

    scale_x = scaled_width / img_width
    scale_y = scaled_height / img_height

    # Compile URL pattern for detection
    url_pattern = re.compile(r'(https?://\S+|www\.\S+|bit\.ly/\S+)', re.IGNORECASE)
    link_annotations = []

    # Iterate through each detected text block
    for bbox, text in text_positions:
        # Adjust bounding box coordinates based on scaling and positioning
        adjusted_bbox = []
        for point in bbox:
            adjusted_x = point[0] * scale_x + x_offset
            adjusted_y = (img_height - point[1]) * scale_y + y_offset
            adjusted_bbox.append((adjusted_x, adjusted_y))
        
        x_coords = [pt[0] for pt in adjusted_bbox]
        y_coords = [pt[1] for pt in adjusted_bbox]
        bbox_width = max(x_coords) - min(x_coords)
        bbox_height = max(y_coords) - min(y_coords)
        
        # Dynamically adjust font size to fit within the bounding box
        font_size = bbox_height
        text_width = c.stringWidth(text, 'Helvetica', font_size)
        while text_width > bbox_width and font_size > 1:
            font_size -= 0.5
            text_width = c.stringWidth(text, 'Helvetica', font_size)
        
        # Set font and position for the text overlay
        text_writer.setFont("Helvetica", font_size)
        min_x = min(x_coords)
        min_y = min(y_coords)
        text_writer.setTextOrigin(min_x, min_y)
        text_writer.textOut(text)

        # Detect URLs within the text
        match = url_pattern.search(text)
        if match:
            url_text = match.group(0)
            # Ensure the URL has a scheme
            if not re.match(r'^https?://', url_text):
                actual_url = 'http://' + url_text
            else:
                actual_url = url_text

            # Define the rectangle area for the link
            rect = (min_x, min_y, min_x + bbox_width, min_y + bbox_height)
            link_annotations.append({
                'rect': rect,
                'url': actual_url
            })

    # Draw the text onto the PDF
    c.drawText(text_writer)

    # Add URL links as annotations
    for link in link_annotations:
        rect = link['rect']
        actual_url = link['url']
        c.linkURL(actual_url, rect, relative=0, thickness=0)

    # Finalize the page
    c.showPage()

    # Clean up the image file after processing
    try:
        os.remove(image_path)
    except FileNotFoundError:
        pass



def adjust_y(y, height):
    """
    Adjusts the y-coordinate based on the page height.

    Args:
        y (float): The original y-coordinate.
        height (float): The total height of the page.
    """
    return height - y


def process_page(counter, c):
    """
    Processes a single page by either generating a PDF with OCR text or overlaying existing text data.

    Args:
        counter (int): The page number to process.

        c (canvas.Canvas): The ReportLab canvas object to draw on.
    """
    text_path = f"texts/text_page_{counter}.txt"
    image_path = f"images/image_page_{counter}.jpg"

    try:
        with open(text_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        unknown, width, height, tree = data[:4]
    except Exception as e:
        process_page_with_ocr(image_path, c)
        print(f"Error loading JSON data: {e}")
        return
    
    process_image(image_path)
    
    # Draw the image structure onto the PDF
    add_image_to_existing_pdf(image_path, c)

    # Initialize text overlay
    text_writer = c.beginText()

    # Define URL pattern
    url_pattern = re.compile(r'(https?://\S+|www\.\S+|bit\.ly/\S+)', re.IGNORECASE)
    link_annotations = []

    # Define a style for the text
    styles = getSampleStyleSheet()

    # Customize the style to include a background color
    highlighted_style = styles["Normal"].clone('highlighted')
    highlighted_style.backColor = colors.yellow

    _, page_height = landscape(A4)


    # Iterate through the tree structure
    for _, ((_, children),) in tree:
        for (y, x, h, w), node in children:
            text = node.strip()
            if not text:
                continue  # Skip empty text

            font_name = "Helvetica-Bold"

            # Calculate optimal font size
            font_size = calculate_font_size(text, w, h, font_name)

            text_writer.setFont(font_name, font_size)

            # Adjust Y-coordinate
            adjusted_y = adjust_y(y, page_height) - 7.5 # Magic number
            text_writer.setTextOrigin(x, adjusted_y)
            text_writer.textOut(text)

            # Detect URLs
            match = url_pattern.search(text)
            if match:
                url_text = match.group(0)
                link_annotations.append({
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'url': url_text
                })

    c.drawText(text_writer)

    # Add URL links
    for link in link_annotations:
        x = link['x']
        y = link['y']
        w = link['w']
        h = link['h']
        actual_url = link['url']

        if not re.match(r'^https?://', actual_url):
            actual_url = 'http://' + actual_url

        x1 = x
        y1 = adjust_y(y + h, page_height)
        x2 = x + w
        y2 = adjust_y(y, page_height)

        rect = (x1, y1, x2, y2)
        c.linkURL(actual_url, rect, relative=0, thickness=0)

    # Finalize the page
    c.showPage()

    # Clean up the image and text files for the processed page
    try:
        os.remove(image_path)
    except FileNotFoundError:
        pass
    try:
        os.remove(text_path)
    except FileNotFoundError:
        pass


def pdf_generator(num_pages, path):
    """
    Processes all PDF pages, adds them to a single PDF.

    Args:
        num_pages (int): The total number of PDF pages to process.
        path (str): The file path for the final PDF.
    """
    # Initialize the ReportLab canvas with landscape A4 size
    c = canvas.Canvas(path, pagesize=landscape(A4))

    
    c.setPageCompression(True)


    for counter in range(num_pages):
        try:
            process_page(counter, c)
        except Exception as e:
            print(f"Error processing page {counter}: {e}")

    # Save the single PDF after all pages have been added
    c.save()


def download_schedule(path, num_pages, img_default_link, json_default_link):
    """
    Manages the entire download and PDF generation schedule, including directory setup and cleanup.

    Args:
        path (str): The file path for the final merged PDF.
        num_pages (int): The total number of pages to download and process.
        img_default_link (str): The base URL for downloading images.
        json_default_link (str): The base URL for downloading text files.
    """
    os.makedirs("images", exist_ok=True)
    os.makedirs("texts", exist_ok=True)

    asyncio.run(download_content(num_pages, img_default_link, json_default_link))
    
    try:
        with open("texts/text_page_0.txt", "r", encoding="utf-8") as f:
            data = f.read()
        pattern = r'Actualizat:.*?(\d{2}\.\d{2}\.\d{4})'
        date = re.search(pattern, data, re.DOTALL)
        path = path + "-" + date.group(1) + ".pdf"
    except:
        path = path + ".pdf"
        print("An error occurred while trying to extract the date from the text file.")

    pdf_generator(num_pages, path)

    # Remove the images and texts directories after processing
    shutil.rmtree("images")
    shutil.rmtree("texts")

def setup_selenium():
    """
    Sets up the Selenium WebDriver with Chrome and enables logging of network requests.

    Returns:
        webdriver.Chrome: Configured Selenium Chrome WebDriver instance.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    # Enable performance logging
    chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    # Initialize the Chrome WebDriver
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def get_specific_network_requests(driver, timeout=30):
    """
    Retrieves the first occurrence of specific network request URLs captured by Selenium.

    Args:
        driver (webdriver.Chrome): Selenium WebDriver instance.
        timeout (int, optional): Maximum time to wait for network activity to complete in seconds.

    Returns:
        dict: A dictionary containing the first URLs matching each specified substring.
    """
    # Define the substrings to search for and their corresponding keys
    desired_substrings = {
        "json_default_link": "presspage?ck",
        "img_default_link": "img",
        "presspage_link": "meta?ck"
    }

    # Initialize the dictionary to store the first occurrence of each substring
    network_requests = {key: None for key in desired_substrings}

    start_time = time.time()

    while True:
        # Fetch performance logs
        try:
            logs = driver.get_log('performance')
        except Exception as e:
            print(f"    Error fetching performance logs: {e}")
            break

        for entry in logs:
            try:
                log = json.loads(entry['message'])['message']
                if log['method'] == 'Network.requestWillBeSent':
                    request_url = log['params']['request']['url']
                    for key, substr in desired_substrings.items():
                        if network_requests[key] is None and substr in request_url:
                            network_requests[key] = request_url
                            break  # Prevent checking other substrings for the same URL
            except (json.JSONDecodeError, KeyError) as e:
                # Skip malformed log entries
                continue

        # Check if all desired URLs have been found
        if all(value is not None for value in network_requests.values()):
            break

        # Break the loop if timeout is reached
        if time.time() - start_time > timeout:
            print("    Timeout reached while capturing network requests.")
            break

    return network_requests

def get_num_pages(meta_ck_url):
    """
    Retrieves the number of pages from the provided meta?ck URL.

    Args:
        meta_ck_url (str): The URL containing the meta?ck parameter.

    Returns:
        int: The number of pages extracted from the JSON response.
    """
    if not meta_ck_url:
        return 0

    try:
        response = requests.get(meta_ck_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        content = response.text
        cleaned_content = content.lstrip(")]}'")
        data = json.loads(cleaned_content)
        num_pages = data.get('pages', 0)
        return num_pages
    except Exception as e:
        print(f"    Error fetching num_pages from {meta_ck_url}: {e}")
        return 0

def enrich_link_data(driver, links):
    """
    Processes a single Bit.ly link by visiting it, capturing specific network requests,
    and extracting additional information.

    Args:
        driver (webdriver.Chrome): Selenium WebDriver instance.
        links (dict): Dictionary containing 'url', 'text', and 'semester'.

    Returns:
        dict: The enriched link_data dictionary with added 'json_default_link',
              'img_default_link', and 'num_pages'.
    """
    results = []
    for link_data in links:
        url = link_data['url']
        text = link_data['text']
        semester = link_data['semester']
        year = link_data['year']

        specific_network_urls = {}

        try:
            driver.get(url)
            # Wait for the page to load and network requests to complete
            time.sleep(5)  # Adjust sleep time as needed based on page complexity
            specific_network_urls = get_specific_network_requests(driver)
        except Exception as e:
            print(f"    Error visiting {url}: {e}")

        # Extract the desired URLs
        json_default_link = specific_network_urls.get('json_default_link')
        img_default_link = specific_network_urls.get('img_default_link')
        meta_ck_url = specific_network_urls.get('presspage_link')

        # Get num_pages from json_default_link
        num_pages = get_num_pages(meta_ck_url)

        # Enrich the link_data dictionary
        enriched_link_data = {
            'text': text,
            'year': year,
            'semester': semester,
            'num_pages': num_pages,
            'img_default_link': img_default_link,
            'json_default_link': json_default_link
        }
        results.append(enriched_link_data)

    return results

def get_hrefs_with_data_type(urls, filter_words):
    """
    Retrieves Bit.ly links from the specified URLs, assigning semester numbers and extracting the academic year.

    Args:
        urls (list): A list of URLs of the webpages to scrape.
        filter_words (list, optional): List of words to filter the link texts.

    Returns:
        list: A list of dictionaries, each containing 'url', 'text', 'semester', and 'year'.
    """
    flattened_links = []

    # Define the regex pattern for year extraction
    year_pattern = re.compile(r'(\d{4}-\d{4})')

    for url in urls:
        try:
            # Fetch the web page
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP errors
        except requests.exceptions.RequestException as e:
            print(f"Error fetching the URL '{url}': {e}")
            continue  # Skip to the next URL instead of returning

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the title and find the year
        page_text = soup.get_text(separator=' ', strip=True)
        year_match = year_pattern.search(page_text)
        if year_match:
            academic_year = year_match.group(1)

        # Find all <h2> elements with class 'wp-block-heading' containing 'Semestrul'
        h2_elements = soup.find_all('h2', string=re.compile(r'Semestrul', re.IGNORECASE))

        if not h2_elements:
            print(f"No semester headers found in '{url}'.")
            continue  # Skip to the next URL

        semester = len(h2_elements)

        for h2 in h2_elements:
            # Find all sibling elements after the current <h2> until the next <h2>
            siblings = h2.find_next_siblings()
            for sibling in siblings:
                if sibling.name == 'h2' and 'semestrul' in sibling.get_text(strip=True).lower():
                    break  # Stop if the next semester <h2> is encountered

                # Within the sibling, find all <a> tags with 'bit.ly' in href
                a_tags = sibling.find_all('a', href=True)
                for a_tag in a_tags:
                    href = a_tag['href']
                    if 'bit.ly' in href or 'drive.google.com' in href:
                        link_text = a_tag.get_text(strip=True)
                        link_text.replace(" ", "")

                        if any(word.lower() in link_text.lower() for word in filter_words):
                            flattened_links.append({
                                'url': href,
                                'text': link_text,
                                'semester': semester,
                                'year': academic_year
                            })
            semester -= 1

    return flattened_links

def parse_year_range(year_range_str):
    # Split the string "year-year" into two integers
    start_year, end_year = map(int, year_range_str.split('-'))
    
    # Ensure that the difference between years is at most 1
    if end_year - start_year > 1:
        year_range_list = [f"{start_year}-{start_year+1}", f"{start_year+1}-{end_year}"]
    else:
        year_range_list = [year_range_str]
    
    return year_range_list

def filter_links_by_semester_and_type(links, semesters=None, schedule_type=None):
    filtered_links = []
    for link in links:
        # Filter by semesters if provided
        if semesters is not None and link['semester'] not in semesters:
            continue
        # Filter by schedule type if provided
        if schedule_type is not None and schedule_type not in link['text'].lower():
            continue
        filtered_links.append(link)
    return filtered_links

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download schedules based on year range, semester, and type filter.")
    
    parser.add_argument("--year_range", help="Year range of the form 'year-year' (e.g., 2022-2024). If not provided, downloads all years.")
    
    parser.add_argument("--semester", type=int, choices=[1, 2], help="Which semester to download: 1 for first, 2 for second.")
    
    parser.add_argument("--schedule_type", choices=['grupe', 'profesori'], help="Type of schedule to download: grupe, profesori.")
    
    args = parser.parse_args()
    
    filter_words = ["grupe", "grupelor", "profesori", "profesorilor"]
    
    # Define the URLs to search
    urls = ['https://fmi.unibuc.ro/orar-2020-2021/', 'https://fmi.unibuc.ro/orar-2021-2022/', 
            'https://fmi.unibuc.ro/orar-2022-2023/', 'https://fmi.unibuc.ro/orar/orar-2023-2024/', 
            'https://fmi.unibuc.ro/orar/']
    
    bitly_links = get_hrefs_with_data_type(urls, filter_words)
    
    # If a year range is passed, process it to create the list of years
    if args.year_range:
        year_list = parse_year_range(args.year_range)
    else:
        # If no year range is passed, use all available years from the links
        year_list = list(set([link['year'] for link in bitly_links]))

    # Map semester argument to appropriate semester filter, if passed
    semesters = [args.semester] if args.semester else None

    # Schedule type filter (grupe, profesori), if passed
    schedule_type = args.schedule_type

    driver = setup_selenium()

    # Filter links by year, semester, and schedule type
    filtered_links = []
    for link in bitly_links:
        if link['year'] in year_list:
            filtered_links.extend(filter_links_by_semester_and_type([link], semesters, schedule_type))
    
    try:
        enriched_results = enrich_link_data(driver, filtered_links)
    finally:
        driver.quit()

    # Download the schedules based on the filtered results
    for result in enriched_results:
        path = f"{result['year']}/sem{result['semester']}"
        os.makedirs(path, exist_ok=True)
        path += "/" + result['text']
        download_schedule(path, result['num_pages'], result['img_default_link'], result['json_default_link'])
