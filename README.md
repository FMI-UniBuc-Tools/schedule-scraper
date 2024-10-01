# PDF Generator with OCR and Link Extraction

## Overview

This project is a Python-based tool designed to automate the process of downloading images and text files from specified URLs, performing Optical Character Recognition (OCR) to extract text from images, and compiling the results into consolidated PDF documents. It leverages web scraping techniques to extract specific links from a target webpage, processes each link to gather necessary data, and generates PDFs with embedded OCR text and clickable links.

The implemented logic can work with any PDF, but this project specifically focuses on the timetables of the Faculty of Mathematics and Computer Science.

## Features

- **Web Scraping**: Utilizes Selenium WebDriver to extract specific bit.ly links containing "grupe" or "teacher" from a target webpage.
- **Asynchronous Downloads**: Employs `aiohttp` to concurrently download images and text files for efficient data retrieval.
- **OCR Processing**: Integrates PaddleOCR to perform text extraction from images, enabling searchable and selectable text within PDFs.
- **PDF Generation**: Uses ReportLab to create PDF documents with images and overlaid OCR text, maintaining the layout and enabling clickable URLs.
- **PDF Merging**: Combines individual PDF pages into a single consolidated PDF using PyPDF2.
- **Cleanup**: Automatically removes temporary files post-processing to maintain a clean working directory.

## Prerequisites

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.7 or higher
- **Chrome Browser**: Latest version recommended for compatibility with ChromeDriver

### Dependencies

The project relies on the following Python libraries:

- `aiohttp==3.8.1`
- `PyPDF2==3.0.1`
- `PaddleOCR==2.8.1`
- `paddlepaddle==2.4.2`
- `reportlab==3.6.12`
- `requests==2.31.0`
- `selenium==4.10.0`
- `Pillow==10.0.1`
- `numpy==1.23.5`
- `opencv-python-headless==4.7.0.72`

### Usage Tutorial

- Install the dependencies by running the following command in the terminal of the project:

```console
pip install -r dependencies.txt
```

- Update the target_url variable with the latest url of the page. The target_url is located inside the within the `__main__` block.
- Run the code and wait for the pdf files to be generated. It may take up to a couple of minutes, based on the quality of the internet connection and the size of the files.


