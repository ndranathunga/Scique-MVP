import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings

# from langchain.question_answering import QuestionAnswering
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from transformers import BertTokenizer, BertLMHeadModel


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )

    chunks = text_splitter.split_text(text)

    return chunks


def get_vectorestore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore


def get_llm():
    # llm = HuggingFaceHub(
    #     model_name="microsoft/deberta-v3-large",
    #     # model_kwargs={"temperature": 0.1, "max_length": 512}
    # )

    llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    # tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    # llm = BertLMHeadModel.from_pretrained("bert-large-uncased")

    return tokenizer, llm


def add_choice(choice_num):
    return st.text_input(f"Enter choice {choice_num}:")


def main():
    load_dotenv()

    if "qa_disabled" not in st.session_state:
        st.session_state["qa_disabled"] = True

    if "context" not in st.session_state:
        st.session_state["context"] = None

    if "llm" not in st.session_state:
        st.session_state["llm"] = None

    if "tokenizer" not in st.session_state:
        st.session_state["tokenizer"] = None

    if "choices" not in st.session_state:
        st.session_state["choices"] = []

    if "choice_count" not in st.session_state:
        st.session_state["choice_count"] = 1

    st.set_page_config(page_title="Scique", page_icon=":book:", layout="wide")
    st.header("Scique")

    question = st.text_input(
        "Enter the question:", disabled=st.session_state.qa_disabled
    )

    # List to store additional choices
    choice = st.text_input(
        f"Enter choice {st.session_state.choice_count}:",
        disabled=st.session_state.qa_disabled,
    )

    if st.button("Add", disabled=st.session_state.qa_disabled):
        if choice != "":
            st.session_state.choices.append(choice)
            st.session_state.choice_count += 1
            st.experimental_rerun()

    if st.button("Clear", disabled=st.session_state.qa_disabled):
        st.session_state.choices = []
        st.session_state.choice_count = 1
        st.experimental_rerun()

    # st.write(st.session_state.choices)

    s = ""

    for i in st.session_state.choices:
        s += "- " + i + "\n"

    st.markdown(s)

    if st.button(
        "Process", key="process_question", disabled=st.session_state.qa_disabled
    ):
        with st.spinner("Processing..."):
            if len(st.session_state.choices) > 0:
                prompt = f'Example: \nWhich planet in our solar system is known as the "Red Planet"? \n(*) Jupiter \n(*) Mars \n(*) Venus \n(*) Neptune \nAnswer: Mars\n\n\n Question: {question}\n'

                for i, choice in enumerate(st.session_state.choices):
                    prompt += f"\n(*) {choice}"

                prompt += f"\nContext: {st.session_state.context}\nAnswer: [MASK]"

                # input_ids = st.session_state.tokenizer.encode(
                #     prompt, return_tensors="pt"
                # )

                # # Generate text from the model
                # output = st.session_state.llm.generate(input_ids, max_length=100)
                # generated_text = st.session_state.tokenizer.decode(
                #     output[0], skip_special_tokens=True
                # )
                # st.success(generated_text)

                encoded_input = st.session_state.tokenizer(prompt, return_tensors="pt")
                output = st.session_state.llm.generate(
                    **encoded_input,
                    max_length=100,
                    # num_beams=5,
                    # early_stopping=True,
                )

                generated_text = st.session_state.tokenizer.decode(
                    output[0], skip_special_tokens=True
                )
                # scores = output[0][0]
                # st.write(scores)
                # st.success(st.session_state.tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0]))
                st.success(
                    generated_text                    
                )
                # # Get the predicted token ids
                # predicted_token_ids = output.logits.argmax(dim=-1)

                # # Decode the predicted token ids
                # decoded_output = st.session_state.tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)

                # # Display the decoded output using Streamlit
                # st.success(decoded_output)

            else:
                st.error("Please enter at least one choice!")

    with st.sidebar:
        st.subheader("Enter context for the Question Answering model:")
        context = st.text_area("Enter your text here:", height=300)

        if st.button("Process", key="process_context"):
            with st.spinner("Processing..."):
                if st.session_state["context"] != "":
                    st.session_state.context = context
                    st.session_state.tokenizer, st.session_state.llm = get_llm()
                    st.session_state.qa_disabled = False

                else:
                    st.error("Please enter a context!")

            st.success("Context processed successfully!")
            st.write("You can now enter your question and choices.")
            # st.write(st.session_state.context)


if __name__ == "__main__":
    main()
