# PDF Generator with OCR and Link Extraction

## Overview

This is a Python project that utilizes various tools and libraries to locally reconstruct a PDF from the Google Drive platform, for which download permissions have been disabled.

The implemented logic can work with any PDF, but this project specifically focuses on the timetables of the Faculty of Mathematics and Computer Science.

## Features

- **Web Scraping**: Utilizes methods from the Selenium WebDriver library to navigate the structure of the timetable page and extract links from the site's requests, which are necessary for reconstructing the PDF.
- **Asynchronous Downloads**: Since the timetables in question contain hundreds of pages, file downloads are performed asynchronously to optimize the efficiency of the script.
- **OCR Processing**: In some cases, files may contain pages with only images and no text. To allow content to be searchable throughout the entire PDF, we use the PaddleOCR model to recognize the text that appears in images.
- **PDF Generation**: Uses ReportLab to create PDF documents with images and overlaid OCR text, maintaining the layout and enabling clickable URLs.
- **PDF Merging**: Once the necessary images and texts have been downloaded, we overlay the images with the corresponding texts and create the PDF.
- **Cleanup**: Upon completion of the reconstruction process, the script deletes the temporary files that were generated or downloaded.

## Why It Was Implemented This Way:

- It may seem like too many tools are being used for web page manipulation and content downloading, but this is due to the limitations of the requests library (which can retrieve the content of a response from a request but cannot identify the requests made by a site) and the low efficiency of the Selenium library (which is not fast enough for this task, as it loads entire pages and simulates real user input).
- The introduction of an OCR model might seem like overkill for this project, but its use is necessary when a PDF has not been properly uploaded. We tried several common OCR models, such as pytesseract or EasyOCR, but PaddleOCR achieved the best results in terms of character recognition. Another factor in choosing PaddleOCR was the ease of setup, as PaddleOCR excels here, with its setup consisting simply of installing the libraries via terminal commands.

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

They can be found inside the dependencies file.

### Usage Tutorial

- Install the dependencies by running the following command in the terminal of the project:

```console
pip install -r dependencies.txt
```

- Update the target_url variable with the latest url of the page. The target_url is located inside the `__main__` block.
- Run the code and wait for the pdf files to be generated. It may take up to a couple of minutes, based on the quality of the internet connection and the size of the files.


