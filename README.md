# Métodos Numéricos Aplicados à Transferência de Calor

O conteúdo está dividido em três Jupyter Notebooks, encontrados no diretório [Aulas](./Aulas/), são eles:

1. [Introdução](./Aulas/01-Introducao.ipynb): Revisão sobre conceitos de programação em Python, além da demonstração (não exaustiva) das principais bibliotecas para o trabalho com ciência, técnologia e educação;
2. [Condução térmica](./Aulas/02-Exemplos-Conducao-de-calor.ipynb): Exemplos resolvidos de problemas 1D e 2D, estacionários e transientes envolvendo condução de calor;
3. [Transferência de calor por convecção](./Aulas/03-Exemplos-Conveccao.ipynb): Exemplos resolvidos de problemas 1D e 2D, incluindo o escoamento em uma cavidade com transferência de calor.

## Configurando o Tutorial

Inicie o _Pangeo Binder_ (Ambiente interativo para o Jupyter Notebook-lab na nuvem), clicando em:

    [![Binder](https://binder.pangeo.io/badge_logo.svg)](https://binder.pangeo.io/v2/gh/fschuch/Python-Transferencia-de-Calor/master/)

    * Espere a aplicação carregar tudo para você, isso pode levar algum tempo;
    * O próximo passo é abrir o arquivo `aula.ipynb`;
    * No menu superior, procure por `Run > Run all cells`;
    * Ao final da aula, não esqueça de salvar uma cópia do Notebook com suas anotações pessoais.

Se você prefere instalar o tutorial localmente, siga os seguintes passos:

    1. Clone o repositório:

    ```
    git clone https://github.com/fschuch/Python-Transferencia-de-Calor
    ```

    1. Instale o ambiente. O repositório inclui um arquivo `environment.yaml` que contém uma lista de todos os pacotes necessários para executar esse tutorial.
    Para instalá-los usando conda, use o comando:

    ```
    conda env create -f environment.yml
    conda activate transferencia-de-calor
    ```

    1. Inicie uma seção Jupyter:

    ```
    jupyter lab
    ```


> **Felipe N. Schuch**,<br>
> Dr. Eng. Mecânico pela PUCRS. Possui experiência em fluidodinâmica computacional, transferência de calor e massa, computação de alto desempenho, métodos numéricos, educação financeira e outros.<br>
> [felipeschuch@outlook.com](mailto:felipeschuch@outlook.com "Email") [@fschuch](https://twitter.com/fschuch "Twitter") [Aprenda.py](https://fschuch.github.io/aprenda.py "Blog") [@aprenda.py](https://www.instagram.com/aprenda.py/ "Instagram")<br>

## Licença

Esse projeto é licenciado sob os termos do [MIT license](https://github.com/fschuch/Python-Transferencia-de-Calor/blob/master/LICENSE).

© 2020 Felipe N. Schuch
