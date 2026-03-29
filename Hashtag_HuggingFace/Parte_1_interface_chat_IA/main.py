import streamlit as st
from hfapi_summarization import resumir


def gerador_texto(prompt):
    st.write("Gerador de texto selecionado")

def resumidor_texto(prompt):
    #st.write("Resumidor de texto selecionado")
    st.markdown("##### Cole na caixa de prompt o texto que deseja resumir")
    # o prompt será enviado aqui
    if prompt:
        texto_resposta = resumir(prompt)
        st.write(texto_resposta)

    
def abrir_chat(prompt):
    st.write("Chat selecionado")

def main_app():
    # titulo -> HashIAs
    st.header("HashIAs", divider=True)
    # subtitulo -> Selecione a IA que mais te ajuda, envie seu prompt e seja feliz
    #  diminuir o tamanho das palavras.    
    st.markdown("#### Selecione a IA que mais te ajuda, envie seu prompt e seja feliz")
    # selectbox -> Gerar Texto, Resumir Texto, Abrir Chat
    # aba ou seleção no selectbox
    opcoes = ["Gerar Texto", "Resumir Texto", "Abrir Chat"]
    # options é a aba de seleção.
    ferramenta_selecionada = st.selectbox("Selecione a ferramenta de IA que você vai usar", options=opcoes)
    
    # Campo de prompt -> Digite aqui seu prompt
    # chat para digitar os prompts
    prompt = st.chat_input("Digite aqui seu prompt")

    if ferramenta_selecionada:
        if ferramenta_selecionada == opcoes[0]: # gerar o texto 
            gerador_texto(prompt)
        elif ferramenta_selecionada == opcoes[1]: # resumir texto fizemos pela ordem.
            resumidor_texto(prompt)
        else:
            abrir_chat(prompt)


main_app()