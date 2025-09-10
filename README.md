🧠 Model & Workflow

This project does not use a heavy deep learning model.
Instead, it relies on Natural Language Processing (NLP) preprocessing and a rule-based search mechanism to match queries with book data efficiently.

🔹 Steps in the Pipeline

Preprocessing

*Convert text to lowercase.

*Remove punctuation and special characters.

*Tokenize sentences into words using NLTK.

*Remove stopwords (common words like "and", "the").

*Keep only meaningful tokens (words with length > 2).

-Indexing Book Data

*Each book is stored with metadata (title, author, category, and text).

*After preprocessing, tokens are saved so that searching is faster and more accurate.

*Search Functionality

-Title Search → finds books by title keywords.

*Author Search → matches author name.

*Category Search → filters by subject/genre.

*Content Search → looks into the body of the text.

-User can select whether to search by page, chapter, or whole book.

Display Results

*Streamlit interface shows matched books with relevant snippets.

*A scrollable text area is used for long content previews.# book-search-app
