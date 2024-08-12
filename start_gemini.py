import streamlit as st
from pinecone import Pinecone
import pandas as pd
import google.generativeai as genai
from pinecone_text.sparse import BM25Encoder
import pickle
import nltk
nltk.download('punkt')

with open('bm25_model.pkl', 'rb') as f:
    bm25 = pickle.load(f)

# Define a function to initialize Pinecone
def init_pinecone(api_key):
    pc = Pinecone(api_key=api_key)
    return pc.Index("hybrid-search-poc-gemini")

# Define a function to get embeddings
def get_embedding(query):
    query_embedding = genai.embed_content(model="models/embedding-001",
                                        content=query,
                                        task_type="retrieval_query")
    return query_embedding

# Define a function for hybrid scaling
def hybrid_scale(dense, sparse, alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense['embedding']]
    return hdense, hsparse

# Define a function to generate summaries
def generate_summary(result):
    search_result_text = (
        f"ID: {result['id']}\nTitle: {result['title']}\nBrand: {result['brand']}\nPrice: {result['price']}\nDescription: {result['description']}"
    )
    
    prompt = (f"Generate a human-understandable summary of the following search result:\n\n"
              f"{search_result_text}\n\n"
              f"Please provide a coherent and engaging summary.")
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    answer = model.generate_content(prompt)
    return answer

def main():
    st.title("Pinecone Search Application")
    
    search_intent = st.text_input("Enter your search intent")
    
    pinecone_api_key = st.text_input("Enter Pinecone API Key:", type="password")
    gemini_api_key = st.text_input("Enter Gemini API Key:", type="password")
    alpha = st.slider("Adjust the alpha parameter for hybrid scaling", 0.0, 1.0, 0.5)
    
    categories = ['Skin Care', 'Grocery & Gourmet Foods', 'Bath & Shower', 'Fragrance', 'Hair Care', 'Detergents & Dishwash']
    selected_categories = st.multiselect("Select Categories to Filter", categories)

    if st.button("Search"):
        if search_intent and pinecone_api_key and gemini_api_key:
            genai.configure(api_key=gemini_api_key)

            index = init_pinecone(pinecone_api_key)

            dense = get_embedding(search_intent)
            sparse = bm25.encode_queries(search_intent)
            hdense, hsparse = hybrid_scale(dense, sparse, alpha=alpha)

            # Construct the filter for the query
            filter_query = {"Category": {"$in": selected_categories}} if selected_categories else {}

            query_result = index.query(
                top_k=5,
                vector=hdense,
                sparse_vector=hsparse,
                filter=filter_query,
                include_metadata=True
            )

            results = []
            for match in query_result['matches']:
                id = match.get('metadata', {}).get('id', '')           
                product_title = match.get('metadata', {}).get('Product Title', '')
                brand = match.get('metadata', {}).get('brand', '')
                price = match.get('metadata', {}).get('Price', '')
                description = match.get('metadata', {}).get('Product Description', '')

                result = {
                    "id": id, 
                    "title": product_title, 
                    'brand': brand,
                    'price': price, 
                    'description': description
                }
                
                # Generate summary for each result
                summary = generate_summary(result)
                result['summary'] = summary

                results.append(result)
            
            results_df = pd.DataFrame(results)
            
            st.text('Detailed Results with Summaries')
            st.table(results_df)
        else:
            st.error("Please provide all necessary inputs: search intent, Pinecone API key, and Gemini API key")

if __name__ == "__main__":
    main()
