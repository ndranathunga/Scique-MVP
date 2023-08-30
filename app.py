import streamlit as st
import replicate
import os


def get_api_key():
    if "REPLICATE_API_TOKEN" in st.secrets:
        st.session_state.replicate_api = st.secrets["REPLICATE_API_TOKEN"]

        os.environ["REPLICATE_API_TOKEN"] = st.session_state.replicate_api


def get_answer_from_api(question):
    prompt = f'Example: \nWhich planet in our solar system is known as the "Red Planet"? \n(*) Jupiter \n(*) Mars \n(*) Venus \n(*) Neptune \nAnswer: Mars\n\n\n Question: {question}\n'

    for i, choice in enumerate(st.session_state.choices):
        prompt += f"\n(*) {choice}"

    prompt += f"\nContext: {st.session_state.context}\nAnswer: "

    # Generate the MCQ response
    response = replicate.run(
        "replicate/flan-t5-xl:7a216605843d87f5426a10d2cc6940485a232336ed04d655ef86b91e020e9210",
        input={
            "prompt": prompt,
            "temperature": 0.1,
            "top_p": 0.9,
            "max_length": 512,
            "repetition_penalty": 1,
        },
    )
    return response


def add_choice(choice_num):
    return st.text_input(f"Enter choice {choice_num}:")


def main():
    if "qa_disabled" not in st.session_state:
        st.session_state["qa_disabled"] = False

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

    if "replicate_api" not in st.session_state:
        st.session_state["replicate_api"] = None

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

    s = ""

    for i in st.session_state.choices:
        s += "- " + i + "\n"

    st.markdown(s)

    if st.button(
        "Process", key="process_question", disabled=st.session_state.qa_disabled
    ):
        with st.spinner("Processing..."):
            if len(st.session_state.choices) > 0:
                response = get_answer_from_api(question)

                # placeholder = st.empty()
                full_response = ""
                for item in response:
                    full_response += item
                    # placeholder.markdown(full_response)
                st.success("Answer: " + full_response)

            else:
                st.error("Please enter at least one choice!")

    with st.sidebar:
        st.subheader("Enter context for the Question:")
        context = st.text_area("Enter your text here:", height=300)

        if st.button("Process", key="process_context"):
            st.session_state.qa_disabled = False

            with st.spinner("Processing..."):
                if st.session_state["context"] != "":
                    st.session_state.context = context

                else:
                    st.error("Please enter a context!")

            st.success("Context processed successfully!")
            st.write("You can now enter your question and choices.")
            # st.write(st.session_state.context)


if __name__ == "__main__":
    main()
