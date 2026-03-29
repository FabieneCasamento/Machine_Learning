from huggingface_hub import InferenceClient
from dotenv import load_dotenv

def buscar_token(arquivo):
    with open(arquivo, 'r') as f:
        for linha in f:
            if 'HF_TOKEN' in linha:
                # Divide no '=' e remove espaços ou quebras de linha
                return linha.split('=')[1].strip()

# Uso:
meu_token = buscar_token('../env')
#print(meu_token) # senha token


load_dotenv()


load_dotenv()

def resumir(texto):
    #cliente = InferenceClient()
    cliente = InferenceClient(token=meu_token, model="facebook/bart-large-cnn")
    resposta = cliente.summarization(texto, model="facebook/bart-large-cnn")
    return resposta.summary_text



    