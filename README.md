# Information Retrieval System

## Overview
This Information Retrieval System is a Flask-based web application designed for academic paper searches. It allows users to query a dataset of papers using different search models: Boolean, Vector, and AI-based models. The application supports searching by title, keywords, or abstracts and uses n-gram analysis to enhance the search capabilities.

![index](https://github.com/LizhengMathAi/information-retrieval-system/blob/main/static/index.png)
![search](https://github.com/LizhengMathAi/information-retrieval-system/blob/main/static/search.png)
![details](https://github.com/LizhengMathAi/information-retrieval-system/blob/main/static/details.png)

## Features
- **Multiple Search Models**: Users can choose among Boolean, Vector, or AI-based models for querying the dataset.
- **Schema Selection**: Search can be conducted based on paper titles, keywords, or abstracts.
- **N-Gram Analysis**: Adjust the granularity of the search with customizable n-gram settings.
- **Interactive Results**: Search results include a list of relevant papers based on the query.

## Technology Stack
- **Flask**: Serves the backend and handles request-response cycles.
- **PySpark**: Manages large-scale data processing.
- **NLTK**: Provides tools for text processing and n-gram analysis.

## Installation

### Prerequisites
- Python 3.10.12
- Flask
- Java 8
- Spark 3.1.1
- Hadoop 3.2
- PySpark
- NLTK

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/LizhengMathAi/information-retrieval-system.git
   cd information-retrieval-system
   pip install -r requirements.txt
   python app.py
   ```
