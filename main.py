# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import time

import streamlit as st
import os
from langchain.llms import OpenAIChat


def main():
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
    llm = OpenAIChat()

    st.title('🫸🫷:heart: LangChain first app')

    with st.sidebar:
        with st.echo():
            st.write("This code will be printed to the sidebar.")

            st.write

        with st.spinner("Loading..."):
            time.sleep(5)
        st.success("Done!")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
