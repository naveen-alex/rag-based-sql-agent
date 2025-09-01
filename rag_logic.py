import os
import psycopg2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List, Dict, Tuple
import json
import qdrant_client
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # Fixed import - use langchain_groq

class SQLAgent:
    def __init__(self, qdrant_config: Dict, groq_api_key: str):
        """
        Initialize SQL Agent with Qdrant and GPT OSS 20B on Groq
        """
        # Load environment variables
        load_dotenv()
        
        self.qdrant_config = qdrant_config
        self.groq_api_key = groq_api_key  # Updated variable name
        
        # Configure Qdrant client
        self.qdrant_client = qdrant_client.QdrantClient(
            url=qdrant_config['url'],
        )
        self.collection_name = qdrant_config['collection_name']
        
        # Configure ChatGroq from langchain_groq
        self.model = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="openai/gpt-oss-20b"  # Changed to GPT model
        )
        
        # Initialize sentence transformer for embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Store schema information
        self.schema_info = []
        self.schema_embeddings = None
        
        # Initialize database connection and load schema
        self._load_database_schema()
    
    def _get_db_connection(self):
        """Create database connection"""
        try:
            conn = psycopg2.connect(
                host="localhost",
                database="northwind",
                user="postgres",
                password="postgres",
                port=55432
            )
            return conn
        except Exception as e:
            raise Exception(f"Database connection failed: {str(e)}")
    
    def _load_database_schema(self):
        """Load and embed database schema information"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Get all tables and their columns
            schema_query = """
            SELECT 
                t.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable,
                tc.constraint_type
            FROM information_schema.tables t
            LEFT JOIN information_schema.columns c ON t.table_name = c.table_name
            LEFT JOIN information_schema.key_column_usage kcu ON c.table_name = kcu.table_name 
                AND c.column_name = kcu.column_name
            LEFT JOIN information_schema.table_constraints tc ON kcu.constraint_name = tc.constraint_name
            WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
            ORDER BY t.table_name, c.ordinal_position;
            """
            
            cursor.execute(schema_query)
            results = cursor.fetchall()
            
            # Organize schema information
            tables_info = {}
            for row in results:
                table_name, column_name, data_type, is_nullable, constraint_type = row
                
                if table_name not in tables_info:
                    tables_info[table_name] = {
                        'columns': [],
                        'description': f"Table: {table_name}"
                    }
                
                if column_name:
                    column_info = f"{column_name} ({data_type})"
                    if constraint_type:
                        column_info += f" - {constraint_type}"
                    tables_info[table_name]['columns'].append(column_info)
            
            # Create schema descriptions for embedding
            self.schema_info = []
            for table_name, info in tables_info.items():
                schema_text = f"Table: {table_name}\nColumns: {', '.join(info['columns'])}"
                self.schema_info.append({
                    'table': table_name,
                    'description': schema_text,
                    'columns': info['columns']
                })
            
            # Create embeddings for schema
            schema_texts = [item['description'] for item in self.schema_info]
            self.schema_embeddings = self.embedder.encode(schema_texts)
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            raise Exception(f"Failed to load schema: {str(e)}")
    
    def _find_relevant_tables(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find most relevant tables based on query using Qdrant"""
        query_embedding = self.embedder.encode([query])[0]
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        relevant_tables = []
        for result in search_results:
            table_info = result.payload
            relevant_tables.append({
                'table_info': {
                    'description': table_info.get('text', 'No description available'),
                    'table': table_info.get('table', 'Unknown table')
                },
                'similarity': result.score
            })
        
        return relevant_tables
    
    def _generate_sql_with_gpt(self, user_query: str, relevant_schema: str) -> str:
        """Generate SQL query using ChatGroq"""
        prompt = f"""
        You are an expert SQL developer. Given the user question and database schema, generate a precise PostgreSQL query.
        
        Database Schema:
        {relevant_schema}
        
        User Question: {user_query}
        
        Rules:
        1. Generate only valid PostgreSQL syntax
        2. Use appropriate JOINs when needed
        3. Include proper WHERE clauses for filtering
        4. Use LIMIT when appropriate
        5. Return only the SQL query without explanations
        
        SQL Query:
        """
        
        try:
            response = self.model.invoke(prompt)  # Use invoke method instead of generate
            sql_query = response.content.strip()
            
            # Clean up the response to extract just the SQL
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            
            return sql_query.strip()
        
        except Exception as e:
            raise Exception(f"Failed to generate SQL: {str(e)}")
    
    def _execute_sql_query(self, sql_query: str) -> Tuple[List, List]:
        """Execute SQL query and return results"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute(sql_query)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            
            cursor.close()
            conn.close()
            
            return results, column_names
        
        except Exception as e:
            raise Exception(f"SQL execution failed: {str(e)}")
    
    def _format_results(self, results: List, columns: List) -> str:
        """Format query results for display"""
        if not results:
            return "No results found."
        
        # Convert to pandas DataFrame for better formatting
        df = pd.DataFrame(results, columns=columns)
        
        # Limit to first 10 rows for display
        if len(df) > 10:
            display_df = df.head(10)
            result_text = f"Showing first 10 of {len(df)} results:\n\n"
        else:
            display_df = df
            result_text = f"Found {len(df)} results:\n\n"
        
        result_text += display_df.to_string(index=False)
        
        return result_text
    
    def process_query(self, user_query: str) -> Dict:
        """Main method to process user query using RAG"""
        try:
            # Step 1: Find relevant tables using RAG
            relevant_tables = self._find_relevant_tables(user_query)
            
            # Step 2: Create schema context for LLM
            schema_context = "\n\n".join([
                table['table_info']['description'] 
                for table in relevant_tables
            ])
            
            # Step 3: Generate SQL using GPT OSS 20B
            sql_query = self._generate_sql_with_gpt(user_query, schema_context)
            
            # Step 4: Execute SQL query
            results, columns = self._execute_sql_query(sql_query)
            
            # Step 5: Format results
            formatted_results = self._format_results(results, columns)
            
            return {
                'success': True,
                'sql_query': sql_query,
                'results': formatted_results,
                'relevant_tables': [t['table_info']['table'] for t in relevant_tables],
                'error': None
            }
        
        except Exception as e:
            return {
                'success': False,
                'sql_query': None,
                'results': None,
                'relevant_tables': None,
                'error': str(e)
            }

def create_sql_agent(db_config: Dict, groq_api_key: str) -> SQLAgent:
    """Factory function to create SQL Agent"""
    return SQLAgent(db_config, groq_api_key)  # Updated parameter name
