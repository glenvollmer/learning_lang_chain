from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate

def standard_prompt_template():
    
    prompt_template = PromptTemplate.from_template('Tell me a {adjective} story about {content}.')
    return prompt_template


def chat_prompt_template():

    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ])

    return chat_template



def chat_prompt_template_with_sys_msg():
    sys_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to "
                "be the opposite of what the user inputs."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ])

    return sys_template

def main():
    hf_llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-large",
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 20})

    _standard_prompt_template = standard_prompt_template()

    standard_prompt_chain = _standard_prompt_template | hf_llm
    
    standard_input = {
        'adjective': 'interesting',
        'content': 'banks'
    }

    standard_prompt_response = standard_prompt_chain.invoke(standard_input)
    print(f'standard prompt template response: {standard_prompt_response}')
    print('\n')

    _chat_prompt_template = chat_prompt_template()

    chat_prompt_chain = _chat_prompt_template | hf_llm

    chat_input = {
        'name': 'Glen',
        'user_input': 'elaborate on how are you feeling today in greate detail'
    }

    chat_prompt_response = chat_prompt_chain.invoke(chat_input)
    print(f'chat prompt template response: {chat_prompt_response}')
    print('\n')

    _chat_prompt_template_with_sys_msg = chat_prompt_template_with_sys_msg()

    chat_sys_prompt_chain = _chat_prompt_template_with_sys_msg | hf_llm

    chat_sys_input = {
        'text': 'I can not stand the band Areosmith.',
    }

    chat_sys_prompt_response = chat_sys_prompt_chain.invoke(chat_sys_input)
    print(f'chat system prompt template response: {chat_sys_prompt_response}')
    print('\n')

    return

if __name__ == '__main__':
    main()