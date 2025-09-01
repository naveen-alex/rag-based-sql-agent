# Demo Platform

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/pthom/northwind_psql.git
   ```

2. Navigate into the cloned directory:
   ```
   cd northwind_psql
   ```

3. Run Docker Compose to start the services:
   ```
   docker compose up
   ```

4. Set up Qdrant using Docker:
   ```
   docker run -p 6333:6333 -p 6334:6334 \
       -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
       qdrant/qdrant
   ```

5. Navigate to the parent directory:
   ```
   cd ..
   ```

6. Set up a Python virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

7. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

8. Set up the `.env` file with the following content:
   ```
   QDRANT_URL=http://localhost:6333
   QDRANT_COLLECTION=northwind_rag
   GROQ_API_KEY=
   ```

9. Run the embedding insertion script:
   ```
   python embedd_insert.py
   ```

10. Run the Streamlit app:
    ```
    streamlit run app.py
    ```

## Sample Query

- **Query:** List the top 5 customers by total order value.
