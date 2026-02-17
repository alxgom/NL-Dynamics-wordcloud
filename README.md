# Nonlinear Dynamics Journal Word Cloud Generator

This project scrapes article metadata from the _Nonlinear Dynamics_ journal (Springer), extracts keywords, and generates a word cloud to visualize the most common topics. It specifically targets Volumes 99-111 (January 2020 - early 2023).

## Features

- **Web Scraping:** Fetches article data from Springer Link using `requests` and `BeautifulSoup`.
- **Keyword Extraction:** Parses JSON-LD data embedded in article pages to retrieve author-specified keywords.
- **Text Processing:** Uses `nltk` for tokenization, stopword removal, and lemmatization to clean the keyword data.
- **Visualization:** Generates a word cloud using the `wordcloud` library and `matplotlib`.
- **Data Export:** Saves the frequency of each keyword to a CSV file (`keywords.csv`).

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/alxgom/NL-Dynamics-wordcloud.git
    cd NL-Dynamics-wordcloud
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

    _Note: The project requires `nltk` data. You may need to run the following in Python if not already installed:_

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

## Usage

Run the main script to scrape data and generate the word cloud:

```bash
python webscraping.py
```

### Output

- **`wordcloud.png`**: An image file displaying the generated word cloud.
- **`keywords.csv`**: A CSV file containing the count of each keyword found.
- **Plot**: The script will also display the word cloud in a window using matplotlib.

## Example Output

![Word Cloud Example](https://github.com/user-attachments/assets/28904f14-6672-4594-8ed7-a623d82f8b3a)
