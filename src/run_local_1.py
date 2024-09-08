from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from src.helper import *

# Instruction Token [B_INST - Begining instruction E_INST - Ending instruction  ]

B_INST, E_INST = "[INST]", "[/INST]"  

# System Token   
 
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# instruction = "Convert the following text from English to Hindi: \n\n {text}"
instruction = "Give a proper summary of the of : \n\n {text}"



SYSTEM_PROMPT = B_SYS + CUSTOM_SYSTEM_PROMPT + E_SYS
template = B_INST + SYSTEM_PROMPT + instruction + E_INST


prompt = PromptTemplate(template=template, input_variables=["text"])

llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q4_0.bin',
                    model_type='llama',
                    config={'max_new_tokens': 128,
                            'temperature': 0.01}
                   )

LLM_Chain=LLMChain(prompt=prompt, llm=llm)
# print(LLM_Chain.run("How are you?"))
print(LLM_Chain.run("Harry Potter"))