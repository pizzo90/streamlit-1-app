# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import time

import streamlit as st
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Pinecone as PineconeLC
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from pinecone import ServerlessSpec, PodSpec, Pinecone

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['PINECONE_API_KEY'] = st.secrets['PINECONE_API_KEY']


def split_text(text):
    text = (
        "Once upon a time in the vibrant city of Milan, Italy, lived a brilliant and passionate 33-year-old, tall of 1 meter and 80 centimeters and 65 kg of weight "
        "machine learning developer named Andrea Pizzirani. Andrea was not just any developer; he was a visionary "
        "in the world of artificial intelligence and had dedicated his life to pushing the boundaries of what "
        "machines could achieve. \n  Born into a family of artists, Andrea's early years were filled with "
        "creativity and imagination. Yet, from a young age, he found himself drawn to the world of technology. "
        "His love for coding and algorithms manifested itself early on, and by the time he entered university, "
        "Andrea was already making waves in the field of machine learning. \n During his academic years, "
        "Andrea's brilliance became evident to his professors and peers alike. He was known for his insatiable "
        "curiosity, spending countless nights exploring the depths of neural networks and diving into the "
        "intricacies of complex algorithms. His groundbreaking research in natural language processing caught the "
        "attention of renowned experts in the field, earning him accolades and recognition. \n Upon graduating "
        "with top honors, Andrea delved into the world of industry, eager to apply his knowledge in real-world "
        "scenarios. He joined a cutting-edge startup focused on developing innovative solutions for healthcare "
        "using machine learning. His work was instrumental in creating predictive models that could assist "
        "doctors in diagnosing diseases at an early stage, ultimately saving countless lives. \n As Andrea's "
        "reputation grew, so did his ambition. He dreamt of building something that would leave a lasting impact "
        "on society. Inspired by his experiences in healthcare, he envisioned a world where machine learning "
        "could be applied to solve a myriad of challenges, from environmental issues to education and beyond. \n "
        "Fueling his passion, Andrea founded his own startup, aptly named 'Cognitive Horizons.' The company aimed "
        "to explore the limitless potential of machine learning across various domains. With a team of talented "
        "engineers and researchers, Andrea embarked on a journey to create intelligent systems that could "
        "revolutionize the way people lived and worked. \n One of Cognitive Horizons' first major projects was an "
        "ambitious collaboration with environmental organizations. Andrea and his team developed a machine "
        "learning algorithm that analyzed satellite imagery to monitor deforestation in real-time. The system not "
        "only identified areas at risk but also suggested sustainable practices to mitigate the environmental "
        "impact. \nAs news of Andrea's groundbreaking work spread, Cognitive Horizons attracted attention from "
        "investors and tech enthusiasts worldwide. The startup received funding that allowed them to expand their "
        "research and development efforts. Andrea's vision was becoming a reality, and he found himself at the "
        "forefront of a technological revolution. \n With success came challenges, but Andrea faced them head-on. "
        "He believed in the power of collaboration and fostered an environment where creativity thrived. "
        "Cognitive Horizons continued to push boundaries, exploring applications of machine learning in "
        "education, finance, and even space exploration. \n As the years passed, Andrea Pizzirani became a symbol "
        "of innovation and dedication in the tech industry. His TED talks and keynote speeches at conferences "
        "inspired a new generation of developers and engineers. He emphasized the importance of ethical AI and "
        "responsible use of technology, advocating for a future where machines and humans worked together "
        "harmoniously. \n Andrea's journey was not without its ups and downs, but his perseverance and passion "
        "for his work propelled him forward. As he celebrated his 33rd birthday, surrounded by a team of "
        "like-minded individuals, he reflected on the impact he had made on the world. \nThe story of Andrea "
        "Pizzirani, the 33-year-old machine learning developer, serves as a testament to the transformative power "
        "of technology when driven by a visionary mind. His legacy lived on in the intelligent systems he "
        "created, shaping a future where the boundaries between human and machine blurred, creating a world that "
        "was smarter, more efficient, and filled with endless possibilities.")
    splitted_text = CharacterTextSplitter(
        separator='\n',
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    return splitted_text.split_text(text)


def get_conversation(vectorstore):

    ##
    ##qa_chain = RetrievalQA.from_chain_type(llm, retriever=db, return_source_documents=False)
    ##
    ##print(qa_chain.run(question))

    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conv_chain


def get_vectorstore(chunks, vectorstore='FAISS'):
    embeddings = OpenAIEmbeddings(show_progress_bar=True)
    if vectorstore is None or vectorstore == 'FAISS':
        vectors = FAISS.from_texts(texts=chunks, embedding=embeddings)
    else:
        pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        # spec = PodSpec(environment='gcp-starter')
        pc.Index('andrea-pizzirani')
        vectors = PineconeLC.from_existing_index(index_name='andrea-pizzirani', embedding=embeddings)
    return vectors


def handle_question(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        st.write(message.content)


def main():
    embeddings = OpenAIEmbeddings(show_progress_bar=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if 'splitted_text' not in st.session_state:
        st.session_state.splitted_text = None

    if 'vectors' not in st.session_state:
        st.session_state.vectors = None
        # only for debug generate vectors and store it
        st.session_state.vectors = get_vectorstore('test', 'pinecone')

    #texts = split_text('text')
    # print(vindex.describe_index_stats())
    # PineconeLC.from_texts(texts, embedding=embeddings, index_name='andrea-pizzirani')
    # documents = current.similarity_search(query='lucrezia gaiba', k=2)
    ## return_source_documents=True will let you see which source documents were used

    st.title('🫸🫷LangChain APP')

    with st.sidebar:
        with st.echo():
            st.write("This code will be printed to the sidebar.")
        st.success("Done!")

    user_question = st.text_input('ask here about the document')
    if user_question:
        handle_question(user_question)
    st.session_state.conversation = get_conversation(st.session_state.vectors)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
