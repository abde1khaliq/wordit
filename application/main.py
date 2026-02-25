import fitz
import os
import pytesseract
from PIL import Image


def convert_pdf_to_word_file_image_based(pdf_path):
    pdf = fitz.open(f"{pdf_path}.pdf")

    for page_index in range(len(pdf)):
        page = pdf[page_index]

        try:
            pix = page.get_pixmap()
            img_path = f"page_{page_index}.png"
            pix.save(img_path)

            img = Image.open(img_path)

            text = pytesseract.image_to_string(img, lang="eng")

            with open("output.txt", "a", encoding="utf-8") as file:
                file.write(f"\n--- Page {page_index+1} ---\n")
                file.write(text)

            os.remove(img_path)

        except Exception as e:
            print(f"An error occurred on page {page_index}: {e}")


while True:
    pdf_file_location = input(
        "Enter PDF file location to be converted: ").strip()
    if pdf_file_location:
        if os.path.exists(f"{pdf_file_location}.pdf"):
            convert_pdf_to_word_file_image_based(pdf_file_location)
            break
        else:
            print("File not found, please try again.")
    else:
        print("You must enter a valid location.")
