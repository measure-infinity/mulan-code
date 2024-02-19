import os
import openai
import time

openai.api_key = 'Your API key here.'

class Decoder():
    def __init__(self):
        pass
 
    def decode(self, input, max_length, i, k, temp=0):
        response = decoder_for_gpt3(input, max_length, i, k, temp=temp)
        return response


# Sentence Generator (Decoder) for GPT-3 ...
def decoder_for_gpt3(input, max_length, i, k, temp=0):
    
    # GPT-3 API allows each users execute the API within 60 times in a minute ...
    time.sleep(1)
    
    # Specify engine ...
    # Instruct GPT3
    model = 'gpt4'
    model_type = 'completion'
    if model == "gpt3.5":
        engine = "gpt-3.5-turbo-0125" # "gpt-3.5-turbo-0613"
        model_type = 'chat'
    elif model == "gpt4":
        print('[INFO] use GPT4!!!')
        engine = "gpt-4-0613"
        model_type = 'chat'
    else:
        raise ValueError("model is not properly defined ...")
    
    if model_type == 'completion':
        response = openai.Completion.create(
            engine=engine,
            prompt=input,
            max_tokens=max_length,
            temperature=temp,
            stop=None
        )
        output_text = response["choices"][0]["text"]
    else:
        cnt = 10
        output_text = ''
        while output_text == '' and cnt > 0:
            try:
                response = openai.ChatCompletion.create(
                    model=engine,
                    messages=[{"role": "user", "content": input}],
                    request_timeout=30,
                    max_tokens=1000,
                    temperature=temp
                )
                output_text = response['choices'][0]['message']['content']
            except Exception as e:
                cnt -= 1
                if 'rate limit' in str(e).lower():  ## rate limit exceed
                    print('Rate limit exceeded, wait for 30s and retry...')
                    time.sleep(30)
                else:
                    print(f'{e} | Retrying...')
                    time.sleep(5)
        
    return output_text


def query(decoder, p_query, i):
    z_select = decoder.decode(p_query, 3000, i, 1, temp=0.0)

    return z_select


def gpt(p_template='a dog sitting on the ground and a cat sitting on a table'):
    
    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder()

    output = query(decoder, p_template, 0)

    return output


if __name__ == "__main__":
    gpt()
