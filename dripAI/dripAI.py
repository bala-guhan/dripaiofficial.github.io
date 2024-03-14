import streamlit as st
import os
import openai
import secret_key
from secret_key import openai_api_key

os.environ["OPENAI_API_KEY"] = openai_api_key
page = st.sidebar.selectbox("Choose the page,", ("Home", "Chatbot", "Image generation", "Video Explanation"))

if page == "Home":
    st.title("DripAI")
    tab1, tab2, tab3 = st.tabs(["Chatbot", "Image generation", "Video Explanation"])
    with tab1:
        st.title("DripAI Chatbot")
        st.image("chat-7767694_1280.jpg")
        st.write("""# ChatBot README

Welcome to our ChatBot powered by OpenAI's large language models and LangChain framework!

## Overview
This ChatBot leverages the cutting-edge capabilities of OpenAI's GPT-3.5-Turbo, a state-of-the-art language model, to provide a wide range of functionalities. It's built upon the LangChain framework, designed specifically for integrating and utilizing large language models efficiently.

## Features
- **Natural Language Understanding:** The ChatBot understands and processes natural language input, allowing for seamless interaction.
- **Versatility:** With the power of GPT-3.5-Turbo and LangChain, the ChatBot can perform a variety of tasks ranging from text generation to information retrieval and more.
- **Customizability:** Developers can extend and customize the ChatBot's functionality to suit specific needs and use cases.
- **Scalability:** Thanks to the underlying framework, the ChatBot can scale to handle a large number of concurrent users and tasks.

## Getting Started
To start using the ChatBot, follow these simple steps:
1. **Installation:** Clone the repository and install any necessary dependencies.
2. **Configuration:** Set up the required API keys and environment variables.
3. **Run:** Launch the ChatBot application and start interacting!

## Usage
Once the ChatBot is up and running, users can interact with it through a chat interface. Simply input text-based queries or commands, and the ChatBot will respond accordingly.

## Contributing
We welcome contributions from the community to enhance the capabilities and features of our ChatBot. Please refer to the contribution guidelines for more information on how to get involved.

## Support
If you encounter any issues or have questions about the ChatBot, feel free to reach out to us through the provided support channels. We're here to help!

## License
This project is licensed under the [MIT License](LICENSE), allowing for open collaboration and distribution.

## Acknowledgements
We would like to express our gratitude to OpenAI for providing the powerful GPT-3.5-Turbo model and to the developers behind the LangChain framework for enabling seamless integration.""")
    with tab2:
        st.title("DripAI Image generation")
        st.image("DripAI-Background.png")
        st.write("""# Image Generation Bot README

Welcome to our Image Generation Bot powered by OpenAI's large language models and LangChain framework!

## Overview
This Image Generation Bot harnesses the advanced capabilities of OpenAI's DALL-E 3, a cutting-edge model designed specifically for image generation from textual descriptions. It's built upon the LangChain framework, which facilitates the seamless integration of large language models for diverse tasks.

## Features
- **Image Generation:** The bot utilizes DALL-E 3 to generate images based on textual input, providing a novel way to visually express ideas and concepts.
- **Precision:** By leveraging LangChain's image modules, the bot ensures the accuracy and fidelity of generated images, producing results that closely match the input descriptions.
- **Versatility:** With the power of DALL-E 3 and LangChain, the bot can generate a wide variety of images, ranging from everyday objects to abstract concepts.
- **Customizability:** Developers can extend and customize the bot's functionality to suit specific image generation tasks and use cases.

## Getting Started
To start using the Image Generation Bot, follow these simple steps:
1. **Installation:** Clone the repository and install any necessary dependencies.
2. **Configuration:** Set up the required API keys and environment variables.
3. **Run:** Launch the bot application and start generating images from text descriptions!

## Usage
Once the Image Generation Bot is up and running, users can interact with it through a text-based interface. Simply input textual descriptions of the desired image, and the bot will generate and display the corresponding image.

## Contributing
We welcome contributions from the community to enhance the capabilities and features of our Image Generation Bot. Please refer to the contribution guidelines for more information on how to get involved.

## Support
If you encounter any issues or have questions about the Image Generation Bot, feel free to reach out to us through the provided support channels. We're here to help!

## License
This project is licensed under the [MIT License](LICENSE), allowing for open collaboration and distribution.

## Acknowledgements
We would like to express our gratitude to OpenAI for providing the powerful DALL-E 3 model and to the developers behind the LangChain framework for enabling seamless integration.""")
    with tab3:
        st.title("Video Explanation")
        st.image("man-3774381_1280.jpg")
        st.write("""# Video Explanation Bot README

Welcome to our Video Explanation Bot powered by OpenAI's large language models and LangChain framework!

## Overview
This Video Explanation Bot leverages the capabilities of OpenAI's large language models (LLMs) and LangChain framework to provide insightful analyses and answers based on video content. It utilizes a combination of video transcripts and natural language processing (NLP) specialized LLMs to analyze videos and answer questions derived from the video content.

## Features
- **Video Analysis:** The bot processes video transcripts using NLP specialized LLMs to gain a deep understanding of the video content.
- **Question Answering:** Based on the analysis of the video content, the bot can accurately answer questions related to the video material.
- **Precision:** LangChain modules are integrated to ensure the accuracy and precision of the bot's output, providing reliable explanations and insights.
- **Versatility:** The bot can handle a wide range of video content, from educational lectures to informative documentaries, enabling diverse applications.

## Getting Started
To start using the Video Explanation Bot, follow these simple steps:
1. **Installation:** Clone the repository and install any necessary dependencies.
2. **Configuration:** Set up the required API keys and environment variables.
3. **Run:** Launch the bot application and start analyzing videos and answering questions!

## Usage
Once the Video Explanation Bot is up and running, users can interact with it by providing video transcripts or links to videos. The bot will then process the content and provide insightful analyses and answers to questions based on the video material.

## Contributing
We welcome contributions from the community to enhance the capabilities and features of our Video Explanation Bot. Please refer to the contribution guidelines for more information on how to get involved.

## Support
If you encounter any issues or have questions about the Video Explanation Bot, feel free to reach out to us through the provided support channels. We're here to help!

## License
This project is licensed under the [MIT License](LICENSE), allowing for open collaboration and distribution.

## Acknowledgements
We would like to express our gratitude to OpenAI for providing the powerful large language models and to the developers behind the LangChain framework for enabling seamless integration.""")

elif page == "Chatbot":
    import openai
    import streamlit as st
    from secret_key import openai_api_key

    st.title("DripAI ChatBot")
    openai.api_key = openai_api_key

    st.subheader("Your Favorite chatbot here!")
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


elif page == "Image generation":
    def generate_image(input_text):
        response = openai.Image.create(
            prompt = input_text,
            n = 3,
            size = "512x512"
        )
        image_url = response['data'][0]['url']
        return image_url

    st.header('DripAI Image generation')
    prompt = st.text_input("Enter your prompt here")
    if prompt is not None:
        if st.button("Generate Image"):
            image = generate_image(prompt)
            st.image(image ,caption= "DripAI image gen powered by Dall-E3")

elif page == "Video Explanation":
    from langchain.document_loaders import YoutubeLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import LLMChain
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate
    )
    import textwrap
    from secret_key import openai_api_key
    import os
    import streamlit as st

    st.header("Transcript AI")
    prompt = st.text_input("Enter the video url")
    query = st.text_input("Enter your question")
    os.environ["OPENAI_API_KEY"] = openai_api_key
    embeddings = OpenAIEmbeddings()


    def create_db_from_youtube_video_url(video_url):
        loader = YoutubeLoader.from_youtube_url(video_url)
        transcript = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(transcript)

        db = FAISS.from_documents(docs, embeddings)
        return db

    def get_response_from_query(db, query, k=4):
        docs = db.similarity_search(query, k=k)
        docs_page_content = " ".join([d.page_content for d in docs])
        question = query
        chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

        template = """
                You are a helpful assistant that can answer questions about youtube videos
                based on the video's transcripts: {docs}
                Only use the factual information from the transcript to answer the questions
                If you feel like you dont have enough information to answer the question, just say "I dont know"
                Your answers should be verbose and detailed
        """

        system_message_prompt = SystemMessagePromptTemplate.from_template(template)

        human_template = "Answer the following questions: {question}"
        human_message_prompt = HumanMessagePromptTemplate.from_template((human_template))

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        chain = LLMChain(llm=chat, prompt=chat_prompt)

        response = chain.run(question=query, docs=docs_page_content)
        response = response.replace("\n", "")
        return response, docs


    def transcript_generate(video_url, query):
        db = create_db_from_youtube_video_url(video_url)
        response, docs = get_response_from_query(db, query)
        return response


    if prompt and query:
        video_url = prompt
        db = create_db_from_youtube_video_url(video_url)
        response, docs = get_response_from_query(db, query)
        st.write(textwrap.fill(response, width=85))
