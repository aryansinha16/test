import streamlit as st
import os
import time
import json
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Load documents function
def load_documents(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

# Split documents function with page numbers
def split_documents(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    
    chunked_with_pages = []
    current_page = None
    for chunk in chunks:
        if "Page " in chunk:
            current_page = chunk.split("Page ")[1].split(":")[0]
        chunked_with_pages.append({"page": current_page, "text": chunk})
    
    return chunked_with_pages

# Create embeddings and knowledge base
def create_embeddings_and_knowledge_base(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    texts = [chunk["text"] for chunk in text_chunks]
    metadatas = [{"page": chunk["page"]} for chunk in text_chunks]
    knowledge_base = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return knowledge_base

# Create QA chain
def create_qa_chain(knowledge_base):
    llm = Ollama(model="llama3:instruct", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=knowledge_base.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# Load documents and create QA chain
text = load_documents("preprocessed_text.txt")
text_chunks = split_documents(text)
knowledge_base = create_embeddings_and_knowledge_base(text_chunks)
qa_chain = create_qa_chain(knowledge_base)

# Function to handle user query
def handle_query(question, corrections):
    start_time = time.time()
    
    # Check if there's a correction for the question
    corrected_answer = next((corr["correct_answer"] for corr in corrections if corr["question"] == question), None)
    if corrected_answer:
        answer_with_context = f"Answer: {corrected_answer} (Corrected Answer)"
    else:
        response = qa_chain.invoke({"query": question})
        end_time = time.time()
        response_time = end_time - start_time

        # Extract the page number from the source documents
        page_numbers = set()
        for doc in response["source_documents"]:
            if "page" in doc.metadata and doc.metadata["page"] is not None:
                page_numbers.add(doc.metadata["page"])

        page_numbers_str = ", ".join(page_numbers)
        answer_with_context = f"Answer: {response['result']} (Page(s): {page_numbers_str})"
    
    return answer_with_context

# Function to load feedback from a JSON file
def load_feedback(file_path="feedback.json"):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            feedback_data = json.load(file)
    else:
        feedback_data = []
    return feedback_data

# Function to save feedback to a JSON file
def save_feedback(feedback_data, file_path="feedback.json"):
    with open(file_path, "w") as file:
        json.dump(feedback_data, file, indent=4)

# Function to handle feedback
def handle_feedback(question, answer, feedback, corrections):
    feedback_data = load_feedback()
    
    if feedback.lower() == "no":
        correct_answer = st.text_input("Please provide the correct answer:")
        if st.button("Submit Correction"):
            feedback_entry = {
                "question": question,
                "incorrect_answer": answer,
                "correct_answer": correct_answer
            }
            feedback_data.append(feedback_entry)
            corrections.append(feedback_entry)
            save_feedback(feedback_data)
            st.success("Feedback saved. Thank you!")
    else:
        st.success("Thank you for your feedback!")
    
    return corrections

# Main function for the Streamlit app
def main():
    st.title("QA System with Feedback Loop")
    
    corrections = load_feedback()
    reward_points = st.session_state.get('reward_points', 0)
    
    question = st.text_input("Type your question here:")
    
    if st.button("Submit Question"):
        answer_with_context = handle_query(question, corrections)
        st.write(answer_with_context)
        
        feedback = st.radio("Was the answer correct?", ("Yes", "No"))
        corrections = handle_feedback(question, answer_with_context, feedback, corrections)

        if feedback.lower() == "yes":
            reward_points += 1
        elif feedback.lower() == "no":
            reward_points -= 1

        st.session_state['reward_points'] = reward_points
        st.write(f"Reward points: {reward_points}")

if __name__ == "__main__":
    main()
