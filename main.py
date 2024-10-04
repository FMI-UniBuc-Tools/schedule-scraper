import asyncio
import json
import os
import re
import shutil
import sys

from multiprocessing import Pool

import requests
import aiohttp

from paddleocr import PaddleOCR
from PIL import Image
from PyPDF2 import PdfMerger

from reportlab.lib import pagesizes
from reportlab.lib.pagesizes import landscape
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException


# Set the appropriate event loop policy for Windows platforms
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def download_image(session, url, page_number):
    """
    Asynchronously downloads an image from a modified URL and saves it locally.

    Args:
        session (aiohttp.ClientSession): The HTTP session to use for the request.
        url (str): The base URL to modify and download the image from.
        page_number (int): The page number to replace in the URL.
    """
    modified_url = url.replace("page=0", f"page={page_number}")
    async with session.get(modified_url) as response:
        if response.status == 200:
            content = await response.read()
            with open(f"images/image_page_{page_number}.jpg", "wb") as f:
                f.write(content)
        else:
            print(f"Failed to download image from page {page_number}")

async def download_text(session, url, page_number):
    """
    Asynchronously downloads a text file from a modified URL, processes it, and saves it locally.

    Args:
        session (aiohttp.ClientSession): The HTTP session to use for the request.
        url (str): The base URL to modify and download the text from.
        page_number (int): The page number to replace in the URL.
    """
    modified_url = url.replace("page=0", f"page={page_number}")
    async with session.get(modified_url) as response:
        if response.status == 200:
            content = await response.text()
            lines = content.splitlines()[1:]
            content_without_first_line = "\n".join(lines)
            with open(f"texts/text_page_{page_number}.txt", "w", encoding="utf-8") as f:
                f.write(content_without_first_line)
        else:
            print(f"Failed to download text from page {page_number}")

async def download_content(num_pages, img_default_link, json_default_link):
    """
    Orchestrates the asynchronous downloading of images and text files for all pages.

    Args:
        num_pages (int): The total number of pages to download.
        img_default_link (str): The base URL for downloading images.
        json_default_link (str): The base URL for downloading text files.
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_pages):
            tasks.append(download_image(session, img_default_link, i))
            tasks.append(download_text(session, json_default_link, i))
        await asyncio.gather(*tasks)

def create_pdf_with_ocr_text(image_path, pdf_path):
    """
    Creates a PDF from an image by performing OCR to extract text and overlaying the text on the image.

    Args:
        image_path (str): The file path to the input image.
        pdf_path (str): The file path where the output PDF will be saved.
    """
    # Initialize PaddleOCR with angle classification and English language support
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

    # Perform OCR on the image to extract text and their bounding boxes
    result = ocr_model.ocr(image_path, cls=True)
    text_positions = []
    for res in result:
        for line in res:
            bbox = line[0]
            text = line[1][0]
            text_positions.append((bbox, text))

    # Initialize a ReportLab canvas with landscape A4 size
    c = canvas.Canvas(pdf_path, pagesize=landscape(pagesizes.A4))
    page_width, page_height = landscape(pagesizes.A4)

    # Open the image to get its dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Draw the image onto the PDF canvas, preserving aspect ratio
    c.drawImage(
        image_path,
        0,
        0,
        width=page_width,
        height=page_height,
        preserveAspectRatio=True
    )

    # Initialize text object for invisible text overlay
    text_writer = c.beginText()
    text_writer.setTextRenderMode(3)  # 3 = invisible text

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

    # Overlay each detected text onto the PDF
    for bbox, text in text_positions:
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
        
        text_writer.setFont("Helvetica", font_size)
        min_x = min(x_coords)
        min_y = min(y_coords)
        text_writer.setTextOrigin(min_x, min_y)
        text_writer.textOut(text)
    
    # Draw the text overlay onto the PDF
    c.drawText(text_writer)

    # Finalize and save the PDF
    c.showPage()
    c.save()

def adjust_y(y, height):
    """
    Adjusts the y-coordinate based on the page height.

    Args:
        y (float): The original y-coordinate.
        height (float): The total height of the page.

    Returns:
        float: The adjusted y-coordinate.
    """
    return height - y

def process_page(counter):
    """
    Processes a single page by either generating a PDF with OCR text or overlaying existing text data.

    Args:
        counter (int): The page number to process.
    """
    if not os.path.exists(f"texts/text_page_{counter}.txt"):
        create_pdf_with_ocr_text(f"images/image_page_{counter}.jpg", f"out_page_{counter}.pdf")
        return

    with open(f"texts/text_page_{counter}.txt", "r", encoding="utf-8") as f:
        data = json.load(f)

    unknown, width, height, tree = data[:4]
    c = canvas.Canvas(f"out_page_{counter}.pdf", pagesize=landscape(pagesizes.A4))
    page_width, page_height = landscape(pagesizes.A4)

    c.drawImage(
        f"images/image_page_{counter}.jpg",
        0,
        0,
        width=page_width,
        height=page_height,
        preserveAspectRatio=True
    )

    text_writer = c.beginText()
    text_writer.setTextRenderMode(3)

    url_pattern = re.compile(
        r'(https?://\S+|www\.\S+|bit\.ly/\S+)', re.IGNORECASE)
    link_annotations = []

    for _, ((_, children),) in tree:
        for (y, x, h, w), node in children:
            font_size = 9
            while stringWidth(node, text_writer._fontname, font_size) < w:
                font_size += 0.5
            text_writer.setFont(text_writer._fontname, font_size)
            text_writer.setTextOrigin(x, adjust_y(y, height) - 10)
            text_writer.textOut(node)

            match = url_pattern.search(node)
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

    for link in link_annotations:
        x = link['x']
        y = link['y']
        w = link['w']
        h = link['h']
        actual_url = link['url']

        if not re.match(r'^https?://', actual_url):
            actual_url = 'http://' + actual_url

        x1 = x
        y1 = adjust_y(y + h, height)
        x2 = x + w
        y2 = adjust_y(y, height)

        rect = (x1, y1, x2, y2)
        c.linkURL(actual_url, rect, relative=0, thickness=0)

    c.save()

    # Clean up the image and text files for the processed page
    try:
        os.remove(f"images/image_page_{counter}.jpg")
    except FileNotFoundError:
        pass
    try:
        os.remove(f"texts/text_page_{counter}.txt")
    except FileNotFoundError:
        pass

def merge_pdfs(num_pages, path):
    """
    Merges individual PDF pages into a single consolidated PDF file.

    Args:
        num_pages (int): The total number of PDF pages to merge.
        path (str): The file path for the final merged PDF.
    """
    merger = PdfMerger()
    for counter in range(num_pages):
        pdf_path = f'out_page_{counter}.pdf'
        if os.path.exists(pdf_path):
            merger.append(pdf_path)
        else:
            print(f"Warning: {pdf_path} does not exist and will be skipped.")
    merger.write(path)
    merger.close()

def clean_up(num_pages):
    """
    Removes individual PDF page files after they have been merged.

    Args:
        num_pages (int): The total number of PDF pages to remove.
    """
    for counter in range(num_pages):
        pdf_path = f'out_page_{counter}.pdf'
        try:
            os.remove(pdf_path)
        except FileNotFoundError:
            print(f"Warning: {pdf_path} not found and cannot be removed.")
        except Exception as e:
            print(f"Error removing {pdf_path}: {e}")

def pdf_generator(num_pages, path):
    """
    Processes all PDF pages, merges them into a single PDF, and performs cleanup.

    Args:
        num_pages (int): The total number of PDF pages to process.
        path (str): The file path for the final merged PDF.
    """
    for counter in range(num_pages):
        try:
            process_page(counter)
        except Exception as e:
            print(f"Error processing page {counter}: {e}")
    merge_pdfs(num_pages, path)
    clean_up(num_pages)

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
    pdf_generator(num_pages, path)

    # Remove the images and texts directories after processing
    shutil.rmtree("images")
    shutil.rmtree("texts")

def get_hrefs_with_data_type(url):
    """
    Extracts specific href links from a webpage using Selenium WebDriver and processes each link.

    Args:
        url (str): The target URL to scrape for href links.
        data_type (str, optional): The type of data to filter hrefs. Defaults to "URL".
        wait_time (int, optional): Time to wait for page elements to load. Defaults to 5 seconds.
    """
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--ignore-certificate-errors")
        chrome_options.add_argument("--enable-logging")
        chrome_options.add_argument("--log-level=0")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        bitly_xpath = '//a[contains(@href, "bit.ly") and (contains(text(), "grupelor") or contains(text(), "profesorilor"))]'
        bitly_links = driver.find_elements(By.XPATH, bitly_xpath)
        hrefs = [link.get_attribute('href') for link in bitly_links]

        print(f"Found {len(hrefs)} bitly links.")
        counter = 0

        for bitly_href in hrefs:
            try:
                print(f"\nProcessing {bitly_href}...")
                counter += 1
                link_text = f"orar_{counter}"

                with webdriver.Chrome(options=chrome_options) as new_driver:
                    new_driver.get(bitly_href)
                    logs = new_driver.get_log("performance")

                    img_default_link = None
                    json_default_link = None
                    pages_json = None

                    for entry in logs:
                        log = json.loads(entry["message"])["message"]
                        if log["method"] == "Network.requestWillBeSent":
                            request_url = log["params"]["request"]["url"]
                            if "meta?ck" in request_url:
                                pages_json = request_url
                            if "img" in request_url:
                                img_default_link = request_url
                                json_default_link = request_url.replace("img", "presspage")
                                break
                            if "presspage" in request_url:
                                json_default_link = request_url
                                img_default_link = request_url.replace("presspage", "img")
                                break

                    if pages_json:
                        response = requests.get(pages_json)
                        content = response.text
                        cleaned_content = content.lstrip(")]}'")
                        data = json.loads(cleaned_content)
                        num_pages = data.get('pages')
                        download_schedule(f"{link_text}.pdf", num_pages, img_default_link, json_default_link)
                    else:
                        print(f"No pages_json found for {bitly_href}")

            except WebDriverException as e:
                print(f"WebDriverException occurred while processing {bitly_href}: {e}")
            except Exception as e:
                print(f"An error occurred while processing {bitly_href}: {e}")

    except WebDriverException as e:
        print(f"WebDriverException occurred while accessing {url}: {e}")
    except Exception as e:
        print(f"An error occurred while accessing {url}: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    target_url = "https://fmi.unibuc.ro/orar/"
    # target_url = "https://fmi.unibuc.ro/orar/orar-2023-2024/"
    print(f"Fetching all hrefs with data-type='URL' from {target_url}...\n")
    get_hrefs_with_data_type(
        url=target_url,
    )
