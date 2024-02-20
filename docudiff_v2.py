"""
Copyright 2023 Upekha Ventures

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

###

import logging
from datetime import datetime

log_filename = f"./docudiff_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

# Log that script has started
logging.info("Docudiff script started")

import openai
openai.api_key = "Bearer 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"

def run_openai(text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages = text
    )


    return response

import tiktoken

def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

num_tokens_from_string("tiktoken is great!", "gpt-4")

import pdfminer
from pdfminer.high_level import extract_text

def pdf_to_text(pdf_path):
    text = extract_text(pdf_path)
    return text

import os
from docx import Document
import magic


def docx_to_text(docx_path):
    doc = Document(docx_path)
    return ' '.join([paragraph.text for paragraph in doc.paragraphs])

def txt_to_text(txt_path):
    with open(txt_path, 'r') as file:
        return file.read()

def process_docs(file_path):
    # this function takes in a document, checks if it is a PDF / word / txt file
    # depending on the type, converts it and sets the txt file appropriately 
    
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)

    if file_type == "application/pdf":
        text = pdf_to_text(file_path)
    elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        text = docx_to_text(file_path)
    elif file_type == "text/plain":
        text = txt_to_text(file_path)
    else:
        raise Exception(f"Unsupported file type: {file_type}")

    return text

import pprint
from cleantext import clean

def compare_documents(original_doc, modified_doc):

    original_doc_file = original_doc.name
    modified_doc_file = modified_doc.name

    # Given the documents to compare, process them and get raw text
    original_doc_text = clean(process_docs(original_doc_file))
    pprint.pprint(f"original_doc_text is {original_doc_text}")
    modified_doc_text = clean(process_docs(modified_doc_file))
    pprint.pprint(f"modified_doc_text is {modified_doc_text}")


    # Get token counts
    original_doc_token_count = num_tokens_from_string(original_doc_text, "gpt-4")
    modified_doc_token_count = num_tokens_from_string(modified_doc_text, "gpt-4")

    print(f"token count: original doc is {original_doc_token_count}, modified doc is {modified_doc_token_count}")

    # If total token count is less than 6000, create prompt and ask OpenAI to compare
    if original_doc_token_count + modified_doc_token_count < 6000:
        # Create a prompt for comparison
        prompt = [
            {"role": "system", "content": "I have two documents here."},
            {"role": "user", "content": f"First document: {original_doc_text}"},
            {"role": "user", "content": f"Second document: {modified_doc_text}"},
            {"role": "user", "content": "Can you compare these two documents and report on any differences?"}
        ]

        # Call OpenAI with the prompt
        response = run_openai(prompt)

        return response
    else:
        # Otherwise, handle the situation when the total token count exceeds 3000
        # Here you can do something else or raise an exception
        raise ValueError("Total token count exceeds 3000")
        return

import difflib

def compare_difflib(original_doc, modified_doc):

    original_doc_file = original_doc.name
    modified_doc_file = modified_doc.name

    # Given the documents to compare, process them and get raw text
    original_doc_text = clean(process_docs(original_doc_file))
    modified_doc_text = clean(process_docs(modified_doc_file))

    # Create a SequenceMatcher instance
    matcher = difflib.SequenceMatcher(None, original_doc_text.splitlines(), modified_doc_text.splitlines())

    output = ''
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            output += f"Replace {original_doc_text.splitlines()[i1:i2]} with {modified_doc_text.splitlines()[j1:j2]}\n"
        elif tag == 'delete':
            output += f"Delete {original_doc_text.splitlines()[i1:i2]}\n"
        elif tag == 'insert':
            output += f"Insert {modified_doc_text.splitlines()[j1:j2]}\n"
        elif tag == 'equal':
            output += f"Keep {original_doc_text.splitlines()[i1:i2]}\n"
    return output

def wrapped_compare_documents(original_doc, modified_doc):
    try:
        result = compare_documents(original_doc, modified_doc)
        logging.info("compare_documents completed successfully")
        return result
    except Exception as e:
        logging.exception("Error in compare_documents:")
        return f"An error occurred: {e}"

def wrapped_compare_hybrid(original_doc, modified_doc):
    try:
        result = compare_hybrid(original_doc, modified_doc)
        logging.info("compare_hybrid completed successfully")
        return result
    except Exception as e:
        logging.exception("Error in compare_hybrid:")
        return f"An error occurred: {e}"

from transformers import GPT2TokenizerFast


def chunk_text(text, chunk_size):
    """Chunks text into pieces, none of which exceed chunk_size tokens."""
    # Instantiate the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Initialize variables
    chunks = []
    current_chunk = ""
    for token in text.split(" "):
        if len(tokenizer.encode(current_chunk + " " + token)) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = token
        else:
            current_chunk += " " + token

    # Append the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks




def compare_hybrid(original_doc, modified_doc):
    # Get the diff using difflib
    diff_output = compare_difflib(original_doc, modified_doc)
    print(f"diff output is {diff_output}")

    # Chunk the diff_output into chunks of 6000 tokens
    diff_chunks = chunk_text(diff_output, 6000)

    responses = []

    # Iterate through each diff chunk
    for diff_chunk in diff_chunks:
        # Get token counts
        diff_chunk_token_count = num_tokens_from_string(diff_chunk, "gpt-4")

        print(f"token count: difference chunk is {diff_chunk_token_count}")

        # If token count is less than 6000, create prompt and ask OpenAI to compare
        if diff_chunk_token_count < 7500:
            # Create a prompt for comparison
            prompt = [
                {"role": "system", "content": "I have a document and its changes."},
                {"role": "user", "content": f"Changes in the document: {diff_chunk}"},
                {"role": "user", "content": "Can you interpret these changes and report on the differences in natural language?"}
            ]

            # Call OpenAI with the prompt
            response = run_openai(prompt)

            assistant_response = response['choices'][0]['message']['content']
            responses.append(assistant_response)
        else:
            # Otherwise, handle the situation when the token count exceeds 6000
            # Here you can do something else or raise an exception
            raise ValueError("Token count of a chunk exceeds 6000")

    # Join all responses and return
    return ' '.join(responses)

import gradio as gr
import json

with gr.Blocks() as DocDiffApp:
    
    with gr.Tab("Docudiff"):
        with gr.Row():
            original_doc = gr.File(label="Original Document")
            modified_doc = gr.File(label="Modified Document")
        with gr.Row():
            compare_docs = gr.Button("Compare Documents")
            smart_diff = gr.Button("Smart Compare")
            run_difflib = gr.Button("Run Difflib")
        with gr.Row():
            results_of_diff = gr.Text(label="Diff")

    compare_docs.click(wrapped_compare_documents, inputs=[original_doc, modified_doc], outputs=[results_of_diff]) 
    smart_diff.click(wrapped_compare_hybrid, inputs=[original_doc, modified_doc], outputs=[results_of_diff])
    run_difflib.click(compare_difflib, inputs=[original_doc, modified_doc], outputs=[results_of_diff])

    

DocDiffApp.queue(concurrency_count=5)
DocDiffApp.launch(debug=True, share=False)