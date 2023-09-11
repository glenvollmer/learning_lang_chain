from langchain.llms import HuggingFacePipeline
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser


class FormatAiMessageOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return f'AI: {text}'

def main():
    hf_llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-large",
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 100})


    template = 'You will write a sentance using the following word: '

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = '{text}'
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(
        llm=hf_llm,
        prompt=chat_prompt,
        output_parser=FormatAiMessageOutputParser())
    
    test = chain.run('obnoxious')
    print(test)

    return

if __name__ == '__main__':
    main()