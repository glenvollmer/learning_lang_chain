from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.chains import LLMChain


from feast import FeatureStore
# import pandas as pd
# from datetime import datetime

# entity_df = pd.DataFrame.from_dict({
#     "driver_id": [1001, 1002, 1003, 1004],
#     "event_timestamp": [
#         datetime(2021, 4, 12, 10, 59, 42),
#         datetime(2021, 4, 12, 8,  12, 10),
#         datetime(2021, 4, 12, 16, 40, 26),
#         datetime(2021, 4, 12, 15, 1 , 12)
#     ]
# })

# store = FeatureStore(repo_path="./feast_repo/feature_repo/")

# training_df = store.get_historical_features(
#     entity_df=entity_df,
#     features = [
#         'driver_hourly_stats:conv_rate',
#         'driver_hourly_stats:acc_rate',
#         'driver_hourly_stats:avg_daily_trips'
#     ],
# ).to_df()

# print(training_df.head())

class FeastPromptTemplate(StringPromptTemplate):
    def format(self, **kwargs):
        driver_id = kwargs.pop('driver_id')
        feast_repo_path = "./feast_repo/feature_repo/"
        store = FeatureStore(repo_path=feast_repo_path)
        
        feature_vector = store.get_online_features(
            features=[
                'driver_hourly_stats:conv_rate',
                'driver_hourly_stats:acc_rate',
                'driver_hourly_stats:avg_daily_trips',
            ],
            entity_rows=[{
                'driver_id': driver_id
                }],
        ).to_dict()

        kwargs['conv_rate'] = feature_vector['conv_rate'][0]
        kwargs['acc_rate'] = feature_vector['acc_rate'][0]
        kwargs['avg_daily_trips'] = feature_vector['avg_daily_trips'][0]
        return prompt().format(**kwargs)


def prompt():
    template = """Given the driver's up to date stats, write them note relaying those stats to them.
        If they have a conversation rate above .5, give them a compliment. Otherwise, make a silly joke about chickens at the end to make them feel better

        Here are the drivers stats:
        Conversation rate: {conv_rate}
        Acceptance rate: {acc_rate}
        Average Daily Trips: {avg_daily_trips}

        Your response:"""
    
    p = PromptTemplate.from_template(template)
    return p

def main():
    hf_llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-large",
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 200})
    
    prompt_template = FeastPromptTemplate(input_variables=['driver_id'])

    chain = LLMChain(llm=hf_llm, prompt=prompt_template)
    result = chain.run(1001)
    print(f'llm: {result}')

    return

if __name__ == '__main__':
    main()