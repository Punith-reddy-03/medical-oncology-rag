import fitz
import os

DATA_PATH = "data/"

documents = []

print("Loading PDFs...\n")

for file in os.listdir(DATA_PATH):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(DATA_PATH, file)

        print(f"Loading: {file}")

        pdf = fitz.open(pdf_path)

        for page_num in range(len(pdf)):

            text = pdf[page_num].get_text()

            if text.strip():

                documents.append(
                    {
                        "source": file,
                        "page": page_num + 1,
                        "content": text
                    }
                )

        pdf.close()

print(f"\nTotal Pages Loaded: {len(documents)}")

print("\nSample Document:\n")

print(documents[0])