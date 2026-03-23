import ollama
import faiss
import PyPDF2
import numpy as np
import pickle



def pdf_to_vector(pdf_path):
    # Read_PDF
    print(f"Reading PDF: {pdf_path}")
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total_pages = len(pdf_reader.pages)

        #Extract text from each page separately

        page_texts = []
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            page_texts.append({
                'text': page_text,
                "page_number":page_num + 1
            })

        # combine all text for chucking
        text=''.join([p['text'] for p in page_texts]) 

        print(f"Total pages: {total_pages}" )
        print(f"Total text length: {len(text):,} characters")
        print(f"Average characters per page: {len(text) // total_pages:} ")

        # Chunks work
        chunks = []
        chunk_metadata = []
        
        # Assume the len(text) = how many char in the text var 

        for i in range(0, len(text),400): # range(0, 1200, 400) in our case its 84,006
            chuck_text = text[i:i+500]
            chunks.append(chuck_text)  

            # Estimate which page this chunk belongs to
            estimated_page = min((i // (len(text) // total_pages)) + 1, total_pages)

            chunk_metadata.append({
                'start_pos':i,
                'estimated_page':estimated_page
            })
        print(f"Created {len(chunks)} chunks")

        # Get embeddings from OpenAI
        embeddings =[]
        for i ,chunck in enumerate(chunks):
            print(f"Processing {i+1}/{len(chunks)}")
            response = ollama.embed(model="nomic-embed-text", input = chunck)
            embeddings.append(response['embeddings'][0])

        # Create FAISS index
        print(" Creating FAISS index...")
        embeddings = np.array(embeddings) 
        index = faiss.IndexFlatIP(embeddings.shape[1])  # Embeddings dimensions size differ for each models
        index.add(embeddings.astype('float32'))

        # save to files
        print("Saving to files...")
        faiss.write_index(index, "vectors.index")
        with open("chunks.pkl","wb") as f:
            pickle.dump({
                'chunks': chunks,
                'metadata': chunk_metadata,
                'total_pages': total_pages
            },f)

        print("Vector database created successfully!")
        print(f"Files saved: vectors. index, chunks.pkt")
        print(f"Vector shape: {embeddings. shape} ")
        print(f"Sample vector (first 5 dims): {embeddings[0][:5]}")

        return embeddings, chunks
