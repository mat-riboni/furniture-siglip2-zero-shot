# Furniture SigLIP 2 Vector Search

This repository contains a Streamlit-based web application containerized with Docker, designed for visual search of furniture (ideal for upcycling and restoration projects). The system leverages Google's SigLIP 2 vision-language model to extract image embeddings and integrates with PostgreSQL and pgvector to perform real-time similarity searches.

*Note: Remote connection to Hugging Face API is made for a docker application on Hugging Face Spaces to the endpoint /extract which is not provided by me.*

## Core Features

1. **Image Analysis:** Users upload an image of a piece of furniture. The application uses the SigLIP 2 model to generate a vector embedding of the image.
2. **Vector Search:** The embedding is compared against a local PostgreSQL database using the pgvector extension. The system calculates cosine distance to identify the closest matches.
3. **Result Retrieval:** The application returns the top 5 most visually similar furniture items currently stored in the database.
4. **Execution Flexibility:** A UI toggle allows users to choose between running model inference locally within the Docker container or remotely via a Hugging Face Spaces API.

---

## Getting Started

Follow these sequential steps to configure, launch, and use the application.

### 1. Prerequisites
Ensure the following software is installed on your machine:
* Docker
* Docker Compose

### 2. Clone the Repository
```bash
git clone https://github.com/mat-riboni/furniture-siglip2-zero-shot.git
cd furniture-siglip2-vector-search
```
Or using ssh
```bash
git clone git@github.com:mat-riboni/furniture-siglip2-zero-shot.git
cd furniture-siglip2-vector-search
```

### 3. Configure Environment Variables
Create a file named .env in the root directory of the project (at the same level as docker-compose.yml). Do not commit this file to version control.

Paste the following configuration into the .env file and insert your actual Hugging Face token:

```env
# Database Credentials
DB_NAME=postgres
DB_USER=postgres
DB_PASS=your_secure_password
DB_HOST=db
DB_PORT=5432

# Hugging Face Configuration
HF_TOKEN=your_huggingface_token_here
SPACE_ID=your_username/your_space_name
```

### 4. Build and Run with Docker Compose
Run the following command in your terminal to build the images and start the containers (App and Database) in the background:

```bash
docker-compose up --build -d
```

### 5. Initialize the Database
Once the containers are running, you must initialize the database to enable the pgvector extension and create the required tables. Run the following command from your terminal to execute the init_db.sql file inside the database container:

```bash
docker-compose exec -T db psql -U postgres -d postgres < init_db.sql
```
*Note: If you changed the DB_USER or DB_NAME in your .env file, update the -U and -d flags in the command above accordingly.*

### 6. Access the Application and First Launch Expectations
Open your web browser and navigate to:

http://localhost:8501

Please be aware of the following expected delays during your very first session:
* **Dataset Download:** Upon opening the web interface for the first time, the application will take several minutes to download the required image dataset (Arkan0ID/furniture-dataset). The interface might seem unresponsive during this background process.
* **Model Download:** After uploading an image and clicking the search button for the first time, the initial inference step will take a few additional minutes. The system must download the SigLIP 2 model weights and processor (google/siglip2-base-patch16-224) to your local container.

Subsequent page reloads and searches will execute much faster, as the dataset and model will be cached locally.

---

## Project Structure

* app.py: The main Streamlit application script handling the UI, model initialization, and database queries.
* Dockerfile: Defines the environment and dependencies required to run the Python application.
* docker-compose.yml: Orchestrates the application container and the PostgreSQL database container.
* init_db.sql: The SQL script used to enable the vector extension and set up the schema.
* requirements.txt: The list of required Python dependencies.