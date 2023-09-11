from langchain.llms import HuggingFacePipeline

def main():
    hf_llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-large",
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 20})


    llm_result = hf_llm.predict("hi!")
    print(f'llm: {llm_result}')

    return

if __name__ == '__main__':
    main()