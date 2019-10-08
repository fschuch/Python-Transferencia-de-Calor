#!/usr/bin/env python
# coding: utf-8

# <h1>Sumário<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introdução" data-toc-modified-id="Introdução-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introdução</a></span><ul class="toc-item"><li><span><a href="#Sobre-o-autor" data-toc-modified-id="Sobre-o-autor-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Sobre o autor</a></span></li><li><span><a href="#Sobre-o-material" data-toc-modified-id="Sobre-o-material-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Sobre o material</a></span></li><li><span><a href="#Porque-Python?" data-toc-modified-id="Porque-Python?-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Porque Python?</a></span></li><li><span><a href="#Porque-Jupyter-Notebooks?" data-toc-modified-id="Porque-Jupyter-Notebooks?-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Porque Jupyter Notebooks?</a></span><ul class="toc-item"><li><span><a href="#Como-acessar" data-toc-modified-id="Como-acessar-1.4.1"><span class="toc-item-num">1.4.1&nbsp;&nbsp;</span>Como acessar</a></span></li></ul></li><li><span><a href="#Material-Complementar" data-toc-modified-id="Material-Complementar-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Material Complementar</a></span></li></ul></li><li><span><a href="#Revisão" data-toc-modified-id="Revisão-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Revisão</a></span><ul class="toc-item"><li><span><a href="#Listas" data-toc-modified-id="Listas-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Listas</a></span></li><li><span><a href="#Dicionários" data-toc-modified-id="Dicionários-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Dicionários</a></span></li><li><span><a href="#Módulos" data-toc-modified-id="Módulos-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Módulos</a></span></li><li><span><a href="#Principais-Bibliotecas" data-toc-modified-id="Principais-Bibliotecas-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Principais Bibliotecas</a></span></li><li><span><a href="#Fortran-vs.-Python" data-toc-modified-id="Fortran-vs.-Python-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Fortran vs. Python</a></span></li></ul></li><li><span><a href="#Exercícios-Resolvidos" data-toc-modified-id="Exercícios-Resolvidos-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Exercícios Resolvidos</a></span><ul class="toc-item"><li><span><a href="#Métodos-numéricos" data-toc-modified-id="Métodos-numéricos-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Métodos numéricos</a></span></li><li><span><a href="#Fenômenos-de-Transporte" data-toc-modified-id="Fenômenos-de-Transporte-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Fenômenos de Transporte</a></span></li><li><span><a href="#Vibrações-Mecânicas" data-toc-modified-id="Vibrações-Mecânicas-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Vibrações Mecânicas</a></span></li><li><span><a href="#Eletrônica" data-toc-modified-id="Eletrônica-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Eletrônica</a></span></li><li><span><a href="#Resistência-dos-Materiais" data-toc-modified-id="Resistência-dos-Materiais-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Resistência dos Materiais</a></span></li></ul></li><li><span><a href="#Exercícios-Propostos" data-toc-modified-id="Exercícios-Propostos-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Exercícios Propostos</a></span></li></ul></div>

# ## Introdução
# 
# ### Sobre o autor
# 
# Possui graduação em Engenharia Mecânica pela Pontifícia Universidade Católica do Rio Grande do Sul (2013) e Mestrado em Engenharia e Tecnologia de Materiais pela mesma instituição. Atualmente atua como doutorando no Laboratório de Simulação de Escoamentos Turbulentos, Escola Politécnica da PUCRS. Possui experiencia em mecânica dos fluidos computacional, simulação numérica direta, simulação de grandes escalas, fenômenos de transporte, programação, programação paralela e métodos numéricos.
# 
# > **Felipe Nornberg Schuch**,<br>
# > Laboratório de Simulação de Escoamentos Turbulentos (LaSET),<br>
# > Escola Politécnica, Pontifícia Universidade Católica do Rio Grande do Sul.<br>
# > felipe.schuch@edu.pucrs.br

# ### Sobre o material
# 
# * O objetivo desta palestra é **introduzir os principais conceitos empregados em programação e Python**, mais especificamente no contexto interativo da plataforma Jupyter Notebook;
# * Além de demonstrar como **solucionar problemas em transferência de calor** por meio de propostas computacionais;
# * Para tanto, o material inclui uma breve **revisão de conceitos fundamentais** e as principais bibliotecas científicas disponíveis. Para maiores detalhes, **pode-se consultar a documentação disponível** ou mesmo as diversas leituras recomendadas que aparecem no decorrer do texto.

# ### Porque Python?

# <img src="notebook.png">

# * [10 motivos para você aprender Python](https://www.hostgator.com.br/blog/10-motivos-para-voce-aprender-python/)

# ### Porque Jupyter Notebooks?
# 
# ![jupyter](https://jupyter.org/assets/main-logo.svg  "jupyter")
# 
# * Ferramenta web interativa, grátis, de código aberto;
# * Exploração de dados. Permite executar o código, ver o que acontece, modificar e repetir, onde o cientista tem uma *"conversa"* com os dados disponíveis;
# * Útil para a criação de tutoriais interativos;
# * Ele fala a nossa língua. Disponível para várias liguagens de programação, como Python, Julia, R e Fortran;
# * É possível combinar o código com células `Markdown`, para renderizar equações e tabelas, inserir figuras e explicações sobre o código;
# * Facilmente extensível para apresentações de slides.
# * Disponível em [jupyter.org](https://jupyter.org)

# #### Como acessar
# 
# ![anaconda](https://www.anaconda.com/wp-content/uploads/2018/06/cropped-Anaconda_horizontal_RGB-1-600x102.png "anaconda")
# 
# 1. Instalação por meio do pacote [Anaconda](https://www.anaconda.com/);
# 
# ![colab](https://colab.research.google.com/img/colab_favicon.ico "colab")
# 
# 2. Ferramenta colaborativa na núvem: [Google colab](https://colab.research.google.com);
# 
# 3. Palestra disponível em: [github.com/fschuch/JAEP-2019.py](https://github.com/fschuch/JAEP-2019.py)

# Material complementar:
# * [Markdown quick reference](https://en.support.wordpress.com/markdown-quick-reference/)
# * [Jupyter tools to increase productivity](https://towardsdatascience.com/jupyter-tools-to-increase-productivity-7b3c6b90be09)
# * [LaTeX/Mathematics](https://en.wikibooks.org/wiki/LaTeX/Mathematics)
# * [Why Jupyter is data scientists’ computational notebook of choice](https://www.nature.com/articles/d41586-018-07196-1)
# * [5 reasons why jupyter notebooks suck](https://towardsdatascience.com/5-reasons-why-jupyter-notebooks-suck-4dc201e27086)

# ### Material Complementar
# 
# * [12 Steps to Navier-Stokes](https://github.com/barbagroup/CFDPython)
# * [An example machine learning notebook](https://nbviewer.jupyter.org/github/rhiever/Data-Analysis-and-Machine-Learning-Projects/blob/master/example-data-science-notebook/Example%2520Machine%2520Learning%2520Notebook.ipynb)
# * [Mythbusters Demo GPU versus CPU](https://www.youtube.com/watch?v=-P28LKWTzrI)
# * [Why I write with LaTeX (and why you should too)](https://medium.com/@marko_kovic/why-i-write-with-latex-and-why-you-should-too-ba6a764fadf9)
# * [New Developer? You should’ve learned Git yesterday](https://codeburst.io/number-one-piece-of-advice-for-new-developers-ddd08abc8bfa)

# ## Revisão

# * As primeiras linhas de código (`Shift+Enter` executam o bloco):

# In[1]:


'''
Isso é um comentário
'''

print("Olá mundo")

#Isso também é um comentário


# * Declarando variáveis:

# In[615]:


i = 5        #inteiro
f = 6.7      #ponto flutuante
g = 1e-2     #notação exponencial
s = 'abcdef' #string
c = 5.0 + 6j #complexo


# * Operações matemáticas

# In[676]:


1. + 1. # Soma
2. - 1. # Subtração
6. * 7. # Multiplicação
8. / 4. # Divisão
2. **3. # Potência


# In[626]:


5. // 2. # Parte inteira da divisão


# In[627]:


5. % 2. # Resto da divisão


# In[3]:


i = i + 1    #Acumulador 
i


# In[4]:


i += 1       #Forma alternativa para acumulador
i


# * Laço de zero a 4

# In[628]:


for i in range(5):
    print(i)


# * Teste lógico

# In[629]:


if i == 4:
    print('i é igual a 4')
else:
    print('i não é igual a 4, i é igual a '+str(i))


# Material complementar:
# 
# * [More Control Flow Tools](https://docs.python.org/3/tutorial/controlflow.html)

# ### Módulos
# 
# Se você sair do interpretador Python e voltar novamente, as definições feitas (funções e variáveis) serão perdidas. Portanto, se você quiser escrever um programa um pouco mais longo, é melhor usar um editor de texto para preparar a entrada para o interpretador e executá-lo com esse arquivo como entrada. **Isso é conhecido como criar um script**. À medida que seu programa fica mais longo, você pode querer dividi-lo em vários arquivos para **facilitar a manutenção**. Você também pode usar uma função útil que tenha escrito em **vários programas sem copiar sua definição** em cada programa.
# 
# Para suportar isso, o Python tem uma maneira de colocar as definições em um arquivo e usá-las em um script ou em uma instância interativa do interpretador. **Esse arquivo é chamado de módulo**; As definições de um módulo podem ser importadas para outros módulos ou para o módulo principal.
# 
# Um módulo é um arquivo contendo definições e instruções do Python. O nome do arquivo é o nome do módulo com o sufixo `.py` acrescentado. Dentro de um módulo, o nome do módulo (como uma string) está disponível como o valor da variável global `__name__`.

# Por exemplo, use seu editor de texto favorito para criar um arquivo chamado `fibo.py` no diretório atual com o seguinte conteúdo:
# 
# ```Python
# #Módulo dos números de Fibonacci
# 
# def fib(n):    #escreve a série de Fibonacci até n
#     a, b = 0, 1
#     while a < n:
#         print(a, end=' ')
#         a, b = b, a+b
#     print()
# 
# def fib2(n):   # retorna a série de Fibonacci até n
#     result = []
#     a, b = 0, 1
#     while a < n:
#         result.append(a)
#         a, b = b, a+b
#     return result
# ```

# In[32]:


import fibo


# In[33]:


fibo.fib(1000)


# In[34]:


fibo.fib2(100)


# In[35]:


fibo.__name__


# In[36]:


dir(fibo)


# Material complementar:
# 
# * [Python - Modules](https://www.tutorialspoint.com/python/python_modules)
# * [The Python Tutorial - Modules](https://docs.python.org/3/tutorial/modules.html)
# * [Data Structures](https://docs.python.org/3/tutorial/datastructures.html)
# * [Dictionaries in Python](https://realpython.com/python-dicts/)
# * [Classes](https://docs.python.org/2/tutorial/classes.html)

# ### Principais Bibliotecas

# 1.  **SciPy**
# 
# ![SciPy](https://www.scipy.org/_static/images/scipy_med.png "SciPy")
# 
# Ferramentas de computação científica para Python. SciPy refere-se a várias entidades relacionadas, mas distintas:
# 
# * O ecossistema SciPy, uma coleção de software de código aberto para computação científica em Python;
# * A comunidade de pessoas que usam e desenvolvem essa biblioteca;
# * Várias conferências dedicadas à computação científica em Python - SciPy, EuroSciPy e SciPy.in;
# * Fazem parte da família os pacotes, que serão melhor descritos a seguir:
#     * Numpy;
#     * Matplotlib;
#     * Sympy;
#     * IPython;
#     * Pandas.

# * Além disso, a própria biblioteca SciPy, um componente do conjunto SciPy, fornecendo muitas rotinas numéricas:
#     * Funções especiais;
#     * Integração numérica;
#     * Diferenciação numérica;
#     * Otimização;
#     * Interpolação;
#     * Transformada de Fourier;
#     * Processamento de sinal;
#     * Algebra linear e Algebra linear esparsa;
#     * Problema de autovalor esparso com ARPACK;
#     * Algoritmos e estruturas de dados espaciais;
#     * Estatistica;
#     * Processamento de imagem multidimensional;
#     * I/O de arquivos;

# In[38]:


import scipy as sp
import scipy.sparse as sps


# Material complementar:
# * [SciPy](https://www.scipy.org/)
# * [Getting Started](https://www.scipy.org/getting-started.html)
# * [Scipy Lecture Notes](http://scipy-lectures.org/index.html)

# 2.  **Numpy**
# 
# ![Numpy](https://www.scipy.org/_static/images/numpylogo_med.png "Numpy")
# 
# Numpy é um pacote fundamental para a **computação científica em Python**. Entre outras coisas, destaca-se:
# * Objetos em arranjos N-dimensionais
# * Funções sofisticadas
# * Ferramentas para integrar código C/C++ e Fortran
# * Conveniente álgebra linear, transformada de Fourier e capacidade de números aleatórios
# 
# Além de seus usos científicos óbvios, o NumPy também pode ser usado como um contêiner multidimensional eficiente de dados genéricos. Tipos de dados arbitrários podem ser definidos. Isso permite que o NumPy integre-se de forma fácil e rápida a uma ampla variedade de bancos de dados.

# In[39]:


import numpy as np #Importando a biblioteca numpy e definindo-a com o codnome de np


# In[677]:


help(np.arange) #É sempre possível ver a documentação de uma dada função


# In[685]:


a = np.arange(15).reshape(3, 5) #Criando um arranjo com 15 elementos e o redimensionando para o formato 3x5


# In[686]:


# Exemplos da obtenção de várias propriedades de a
a.shape # Verificando as dimensões do arranjo
a.ndim # O número de dimensões
a.dtype.name # Classificação quando ao tipo dos elementos
a.itemsize # Tamanho em bytes de cada elemento
a.size # Número total de elementos no arranjo
type(a)

#Escrevendo a
a


# In[49]:


#Outras funções que merecem destaque:
for f in [np.zeros, np.zeros_like, np.ones, np.linspace]:
    print('=============== '+f.__name__+' ===============\n')
    print(f.__doc__+'\n')


# Material complementar:
# * [NumPy](https://www.numpy.org/)
# * [Quickstart tutorial](https://www.numpy.org/devdocs/user/quickstart.html)

# 3. **Pandas**
# 
# ![Pandas](https://www.scipy.org/_static/images/pandas_badge2.jpg "Pandas")
# 
# O pandas é um pacote Python que fornece **estruturas de dados rápidas, flexíveis e expressivas**, projetadas para tornar o trabalho com dados “relacionais” ou “rotulados” fáceis e intuitivos. O objetivo é ser o alicerce fundamental de alto nível para a análise prática de dados do mundo real em Python. Além disso, tem o objetivo mais amplo de se tornar a mais poderosa e flexível ferramenta de análise / manipulação de dados de código aberto disponível em qualquer linguagem.
# 
# Pandas é bem adequado para muitos tipos diferentes de dados:
# * Dados tabulares com colunas de tipos heterogêneos, como em uma **tabela SQL, arquivo `.csv` ou planilha do Excel**;
# * Dados de **séries temporais** ordenados e não ordenados (não necessariamente de frequência fixa);
# * Dados de matriz arbitrária (homogeneamente digitados ou heterogêneos) com rótulos de linha e coluna;
# * Qualquer outra forma de conjuntos de dados observacionais / estatísticos. Os dados realmente não precisam ser rotulados para serem colocados em uma estrutura de dados de pandas.

# In[50]:


import pandas as pd


# In[51]:


df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})


# In[52]:


df2


# Material complementar:
# * [Pandas](https://pandas.pydata.org/)
# * [10 minutes to pandas](https://pandas.pydata.org/pandas-docs/version/0.25.0/getting_started/10min.html)

# 4. **Sympy**
# 
# ![Sympy](https://scipy.org/_static/images/sympy_logo.png "Sympy")
# 
# SymPy é uma biblioteca Python para **matemática simbólica**. O objetivo é tornar-se um sistema de álgebra computacional (CAS) completo, mantendo o código o mais simples possível para ser compreensível e facilmente extensível. SymPy é escrito inteiramente em Python.

# In[54]:


import sympy as sm
sm.init_printing() #Para escrever equações na tela


# In[55]:


x, t = sm.symbols('x t') # Criando símbolos


# \begin{equation}
# \text{calcular } \int (e^x \sin(x) + e^x \cos(x)) dx
# \end{equation}

# In[56]:


sm.integrate(sm.exp(x)*sm.sin(x) + sm.exp(x)*sm.cos(x), x)


# \begin{equation}
# \text{calcular a derivada de }\sin(x)e^x
# \end{equation}

# In[57]:


sm.diff(sm.sin(x)*sm.exp(x), x)


# \begin{equation}
# \text{calcular } \int_{-\infty}^{\infty} \sin(x^2)
# \end{equation}

# In[58]:


sm.integrate(sm.sin(x**2), (x, -sm.oo, sm.oo))


# \begin{equation}
# \text{calcular } \lim_{x \to 0} \dfrac{\sin(x)}{x}
# \end{equation}

# In[59]:


sm.limit(sm.sin(x)/x, x, 0)


# \begin{equation}
# \text{resolver } x^2 - 2 = 0
# \end{equation}

# In[60]:


sm.solve(x**2 - 2, x)


# \begin{equation}
# \text{resolver a equação diferencial } y'' - y = e^t
# \end{equation}

# In[61]:


y = sm.Function('y')
eq1 = sm.dsolve(sm.Eq(y(t).diff(t, t) - y(t), sm.exp(t)), y(t))
eq1


# Material complementar:
# * [Sympy](https://www.sympy.org/en/index.html)
# * [Documentation](https://docs.sympy.org/latest/index.html)

# 5. **Matplotlib**
# 
# ![Matplotlib](https://www.scipy.org/_static/images/matplotlib_med.png "Matplotlib")
# 
# A Matplotlib é uma biblioteca de plotagem 2D do Python, que produz figuras de qualidade de publicação em uma variedade de formatos impressos e ambientes interativos entre plataformas. O Matplotlib pode ser usado em scripts Python, nos shells do Python e do IPython, no notebook Jupyter, nos servidores de aplicativos da web e em quatro kits de ferramentas de interface gráfica do usuário.
# 
# A **Matplotlib tenta tornar as coisas fáceis simples e as coisas difíceis possíveis**. Você pode gerar gráficos, histogramas, espectros de potência, gráficos de barras, gráficos de erros, diagramas de dispersão, etc., com apenas algumas linhas de código.
# 
# Material complementar:
# * [Matplotlib](https://matplotlib.org/)
# * [Style sheets reference](https://matplotlib.org/3.1.0/gallery/style_sheets/style_sheets_reference.html)
# * [Gallery](https://matplotlib.org/3.1.0/gallery/index.html)

# In[63]:


import matplotlib.pyplot as plt


# In[678]:


#Definindo um novo estilo para as figuras [opcional]
plt.style.use(['seaborn-notebook'])


# In[679]:


x = np.arange(0, 5, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()


# 6. **ipywidgets**
# 
# Widgets são objetos python com eventos que têm uma representação no seu navegador, geralmente como um controle como uma barra deslizante, uma caixa de texto ou outros. Você pode usar widgets para criar GUIs interativas para seus notebooks.

# In[660]:


import ipywidgets as widgets


# In[669]:


widgets.interact(fibo.fib, n=(1, 100, 1));


# Material complementar:
# * [Documentação](https://ipywidgets.readthedocs.io/en/latest/)

# ## Exercícios Resolvidos

# ### Métodos numéricos
# 
# 1. Diferenciação
# 
# 
# * Esquema em diferenças finitas, explícito, centrado e com precisão de O($\Delta x^2$):
# 
# \begin{equation}
#     \frac{\partial f}{\partial x} = f_i' = \dfrac{f_{i+1}-f_{i-1}}{2\Delta x}
# \end{equation}
# 
# \begin{equation}
# \begin{split}
# \begin{bmatrix} f'_{0} \\ f'_{1} \\ \vdots \\ f'_{i} \\ \vdots \\ f'_{n-2}\\ f'_{n-1}\end{bmatrix} =
# \underbrace{
# \frac{1}{2 \Delta x}
# \begin{bmatrix}
# 0 & 1 & & & & & -1 \\
# -1 & 0 & 1 & & \\
#  & \ddots & \ddots & \ddots & & & \\
# & & -1 & 0 & 1 & \\
# & & & \ddots & \ddots & \ddots & \\
# & & & & -1 & 0 & 1\\
# 1 & & & & & -1 & 0
# \end{bmatrix}
# }_{D_x = \text{ Operador diferencial de primeira ordem}}
# \begin{bmatrix} f_{0} \\ f_{1} \\ \vdots \\ f_{i} \\ \vdots \\ f_{n-2}\\ f_{n-1}\end{bmatrix}
# \\
# \mbox{com $2\leq i \leq n -3$ e $f_0 = f_n$}
# \end{split}
# \end{equation}
# 
# \begin{equation}
#     f' = D_x f
# \end{equation}

# * Esquema em diferenças finitas, explícito, centrado e com precisão de O($\Delta x^2$):
# 
# \begin{equation}
#     \frac{\partial^2 f}{\partial x^2} = f_i'' = \dfrac{f_{i+1} - 2 f_{i} + f_{i-1}}{(\Delta x)^2}
# \end{equation}
# 
# \begin{equation}
# \begin{split}
# \begin{bmatrix} f''_{0} \\ f''_{1} \\ \vdots \\ f''_{i} \\ \vdots \\ f''_{n-2}\\ f''_{n-1}\end{bmatrix} =
# \underbrace{
# \frac{1}{(\Delta x)^2}
# \begin{bmatrix}
# -2 & 1 & & & & & 1\\
# 1 & -2 & 1 & & \\
#  & \ddots & \ddots & \ddots & & & \\
# & & 1 & -2 & 1 & \\
# & & & \ddots & \ddots & \ddots & \\
# & & & & 1 & -2 & 1\\
# 1 & & & & & 1 & -2
# \end{bmatrix}
# }_{D_x^2 = \text{ Operador diferencial de segunda ordem}}
# \begin{bmatrix} f_{0} \\ f_{1} \\ \vdots \\ f_{i} \\ \vdots \\ f_{n-2}\\ f_{n-1}\end{bmatrix}
# \\
# \mbox{com $2\leq i \leq n -3$ e $f_0 = f_n$}
# \end{split}
# \end{equation}
# 
# \begin{equation}
#     f'' = D_x^2 f
# \end{equation}

# In[70]:


x = np.linspace(0., 2.*np.pi, num=200, endpoint=False)

dx = (x[1]-x[0])

#Operador diferencial de primeira ordem
Dx = sps.diags([-1., 0., 1.],
              offsets=[-1, 0, 1],
              shape=(x.size,x.size)
              ).toarray()

#Operador diferencial de segunda ordem
Dx2 = sps.diags([1., -2., 1.],
               offsets=[-1, 0, 1],
               shape=(x.size,x.size)
               ).toarray()

Dx /= 2*dx
Dx2 /= dx**2.

#Condições de contorno Periódicas
for D in [Dx, Dx2]:
    D[0,-1] = D[1,0]
    D[-1,0] = D[-2,-1]


# Este é um bom momento para a criação de um módulo, já que operadores diferenciais serão muito utilizados durante os exemplos práticos. Para isso, vamos criar um arquivo nomeado `diferencial.py`, com as seguintes linhas de código:
# 
# ```python
# from scipy.sparse import diags
# from numpy import linspace, pi
# def ordem2(f=2*pi, i=0., n=7, P=False):
#     '''ordem2(f=2*pi, i=0., n=7, P=False)
# 
#     Calcula os operadores diferenciais da primeira e segunda derivada, para uma
#     malha equidistante de i a f, com n pontos, e erro da ordem de h**2.
# 
#     Parâmetros
#     ----------
#     f : real
#         Valor do contorno superior de x.
#         Padrão é 2*pi.
#     i : real
#         Valor do contorno inferior de x.
#         Padrão é zero.
#     n : inteiro
#         Número de pontos da malha na direção x.
#         Padrão é 7.
#     P : bool, opcional
#         Define que a condição de contorno é periódica quando True.
#         Padrão é False.
# 
#     Retorna
#     -------
#     x, Dx e Dx2, respectivamente o vetor posição e os operadores diferenciais
#     para primeira e segunda ordem.
#     '''
#     #Vetor posição
#     x = linspace(i, f, num=n, endpoint=not P)
#     #Operador diferencial de primeira ordem
#     Dx = diags([-1., 0., 1.],
#               offsets=[-1, 0, 1],
#               shape=(x.size,x.size)
#               ).toarray()
#     #Operador diferencial de segunda ordem
#     Dx2 = diags([1., -2., 1.],
#                offsets=[-1, 0, 1],
#                shape=(x.size,x.size)
#                ).toarray()
# 
#     if P: #Condições de contorno periódicas
#         for D in [Dx, Dx2]:
#             D[0,-1] = D[1,0]
#             D[-1,0] = D[-2,-1]
#     else: #Não Periódica
#         Dx[0,0], Dx[0,1], Dx[0,2] = -3., 4., -1.
#         Dx[-1,-3], Dx[-1,-2], Dx[-1,-1] = 1., -4., 3.
#         Dx2[0,0], Dx2[0,1], Dx2[0,2] = 1., -2., 1.
#         Dx2[-1,-3], Dx2[-1,-2], Dx2[-1,-1] = 1., -2., 1.
# 
#     h = (x[1]-x[0])
#     Dx /= 2.*h
#     Dx2 /= h**2.
#     return x, Dx, Dx2
# ```

# In[71]:


#Agora pode-se importar o novo módulo com
import diferencial as dv

x, Dx, Dx2 = dv.ordem2(n=100, P=False)

f = np.cos

plt.plot(x, f(x), label='f(x)')
plt.plot(x, Dx.dot(f(x)), label="f'(x)")
plt.plot(x, Dx2.dot(f(x)), label="f''(x)")

plt.legend()
plt.show()


# * Esquema em diferenças finitas, explícito, diferença para frente e com precisão de O($\Delta t^1$):
# 
# \begin{equation}
#     \frac{\partial f}{\partial t} = \dfrac{f_{k+1}-f_{k}}{\Delta t}
# \end{equation}

# * Esquema em diferenças finitas, implícito, centrado e com precisão de O($\Delta x^6$):
# 
# 
# \begin{equation}
#     \frac{1}{3} f_{i-1}' + f_{i}' + \frac{1}{3} f_{i+1}' = \frac{14}{9} \frac{f_{i+1}-f_{i-1}}{2\Delta x} + \frac{1}{9} \frac{f_{i+2}-f_{i-2}}{4\Delta x}
# \end{equation}
# 
# 
# \begin{equation}
#     f' = \underbrace{A^{-1}B}_{D_x} f
# \end{equation}

# \begin{equation}
# \begin{split}
# \underbrace{
# \begin{bmatrix}
# 1 & 1/3 & & & & & 1/3 \\
# 1/3 & 1 & 1/3 & & \\
#  & \ddots & \ddots & \ddots & & & \\
# & & 1/3 & 1 & 1/3 & \\
# & & & \ddots & \ddots & \ddots & \\
# & & & & 1/3 & 1 & 1/3\\
# 1/3 & & & & & 1/3 & 1
# \end{bmatrix}
# }_{A}
# \begin{bmatrix} f'_{0} \\ f'_{1} \\ \vdots \\ f'_{i} \\ \vdots \\ f'_{n-2}\\ f'_{n-1}\end{bmatrix} =
# \underbrace{
# \frac{1}{\Delta x}
# \begin{bmatrix}
# 0 & 7/9 & 1/36 & & & 1/36 & 7/9 \\
# 7/9 & 0 & 7/9 & 1/36 & & & 1/36\\
# & \ddots & \ddots & \ddots & & & \\
# & 1/36 & 7/9 & 0 & 7/9 & 1/36 \\
# & & & \ddots & \ddots & \ddots & \\
# 1/36 & & & 1/36 & 7/9 & 0 & 7/9 \\
# 7/9 & 1/36 & & & 1/36 & 7/9 & 0
# \end{bmatrix}
# }_{B}
# \begin{bmatrix} f_{0} \\ f_{1} \\ \vdots \\ f_{i} \\ \vdots \\ f_{n-2}\\ f_{n-1}\end{bmatrix}
# \\
# \mbox{com $2\leq i \leq n -3$, $f_0 = f_n$ e $f_0' = f_n'$}
# \end{split}
# \label{eq.dxx_matrix}
# \end{equation}

# 2. Integração
# 
# 
# * Regra do Trapézio:

# \begin{equation}
# \int_a^b f(x)dx \approx \dfrac{f(a) + f(b)}{2} (b-a)
# \end{equation}
# 
# Pode-se dividir em $N$ intervalos:

# \begin{equation}
# \int_a^b fdx \approx \sum_{i=0}^{N-1} \dfrac{f_{i} + f_{i+1}}{2} \Delta x = \dfrac{\Delta x}{2} \left( f_0 + 2f_1 + \dots + 2f_{n-1} + f_{n}\right)
# \end{equation}
# 
# Escrito na forma matricial:
# 
# \begin{equation}
# \int_a^b fdx = \sum \left(
# \underbrace{
# \Delta x
# \begin{bmatrix}
# 1/2 & & & & & & \\
# & 1 & & & & & \\
# & & \ddots & & & & \\
# & & & 1 & & \\
# & & & & \ddots & \\
# & & & & & 1 & \\
# & & & & & & 1/2
# \end{bmatrix}
# }_{I_x = \text{ Operador integral}}
# \begin{bmatrix} f_{0} \\ f_{1} \\ \vdots \\ f_{i} \\ \vdots \\ f_{n-1}\\ f_{n}\end{bmatrix}
# \right)
# \end{equation}

# In[72]:


x = np.linspace(0., 2*np.pi, num=201, endpoint=True)

f, dx = np.sin(x), (x[1]-x[0])

Ix = dx*sps.eye(x.size).toarray() #Operador integral

for i in [0, -1]: #Condições de contorno
    Ix[i] *= 0.5

Ix.dot(f).sum() #Integral


# In[73]:


#Ou alternativamente, utilizando o scipy:
import scipy.integrate
sp.integrate.trapz(f,x)


# Material complementar:
# * [Finite Difference Coefficients Calculator](http://web.media.mit.edu/~crtaylor/calculator.html)
# * [Trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule)
# * MOIN, Parviz. **Fundamentals of engineering numerical analysis**. Cambridge University Press, 2010.

# In[74]:


del D, Dx, Dx2, Ix, dx, f, i, x
plt.close('all')


# ### Transferência de Calor

# 1. **Radiação e convecção combinadas em transferência de calor permanente unidirecional**. A superfície interior de uma parede de espessura $L=0,25m$ é mantida a $21^oC$, enquanto a temperatura no meio externo é $T_\infty = 4^oC$. Considere que há troca de calor com a vizinhança $T_{viz} = 250 K$, o coeficiente de convecção é $h=23W/m^2$, a condutividade térmica do material que compõe a parede é $k=0,65 W/m \cdot ^oC$, a emissividade da superfície externa vale $\epsilon = 0,8$ e a constante de Stefan-Boltzmann $\sigma = 5,67 \times 10^{-8} [W/m^2 \cdot K^4]$. Determine a temperatura externa da parede $T_2$.
# 
# <img src="radiacao.png">
# 
# \begin{equation}
#     k \left( \dfrac{T_1-T_2}{L} \right) = \epsilon \sigma \left( T_2^4 - T_{viz}^4 \right) + h \left( T_2 - T_\infty \right)
# \end{equation}

# In[697]:


# A esquerda como as variáveis serão tratadas computacionalmente, a direita como elas serão exibidas visualmente
k, t1, t2, l, epsi, sigma, tviz, too, h = sm.symbols("k t_1 t_2 L \epsilon \sigma t_{viz} t_\infty h")

# Definindo a equação do problema
eq1 = sm.Eq(k*(t1-t2)/l,epsi*sigma*(t2**4.-tviz**4.)+h*(t2-too))

# Dicionário com os valores que serão substituidos na eq1
dic = {k: .65, t1: 21.+273., l: .25, epsi: .8, sigma: 5.67e-8, tviz: 255., too: 4.+273., h: 23.}

# Resolvendo a equação
sol = sm.solve(eq1.subs(dic), t2)

# Procurando a solução desejada
for val in sol:
    if val.is_real: #A solução deve ser real
        if val.is_positive: #E positiva
            print('T2 = {0:3.2f} graus celsius'.format(val-273.))  


# In[76]:


del k, t1, t2, l, epsi, sigma, tviz, too, h, eq1, dic, sol, val


# 2. **Condição de calor transiente bidimensional**. Uma placa de cobre de $50cm \times 50cm$ inicialmente possui temperatura em toda a sua extensão igual a $0^oC$. Instantaneamente, suas bordas são levadas às temperaturas de $60^oC$ em $x=0$; $20^oC$ em $x=50cm$; $0^oC$ em $y=0$ e $100^oC$ em $y=50$. A difusividade térmica do cobre é $1,1532cm^2/s$. Considerando um $\Delta t = 4s$, $\Delta x = \Delta y = 5cm$, calcule a evolução da temperatura para a posição central da placa até o tempo de $400s$. Para o tempo de $200s$ apresente o perfil de temperatura em todos os pontos discretos do domínio.
#     Equação bidimensional:
# \begin{equation}
# \alpha \left( \dfrac{\partial ^2 T}{\partial x^2} + \dfrac{\partial ^2 T}{\partial y^2} \right) =\dfrac{\partial T}{\partial t}
# \end{equation}
#     Discretizando com a derivada segunda numa representação por diferença central e a derivada primeira com diferença ascendente:
# 
# \begin{equation}
# \dfrac{T^{n+1}_{l,j}-T^{n}_{l,j}}{\Delta t}=\alpha \left[ \dfrac{T^{n}_{l-1,j}-2T^{n}_{l,j}+T^{n}_{l+1,j}}{(\Delta x)^2} +\dfrac{T^{n}_{l,j-1}-2T^{n}_{l,j}+T^{n}_{l,j+1}}{(\Delta y)^2}  \right]
# \end{equation}

# In[77]:


x = np.linspace(0., 50., num=11, endpoint=True)
y = np.linspace(0., 50., num=11, endpoint=True)
t = np.linspace(0., 400., num=101, endpoint=True)

a = 1.1532

# Condição inicial
T = np.zeros((x.size,y.size,t.size))

# Condições de contorno
T[0,:,:], T[-1,:,:], T[:,0,:], T[:,-1,:] = 60., 20., 0., 100.


# In[78]:


dt = t[1]-t[0]
dx2 = (x[1]-x[0])**2.
dy2 = (y[1]-y[0])**2.
for n in range(t.size-1):
    for i in range(1,x.size-1):
        for j in range(1,y.size-1):
            T[i,j,n+1] = dt*a*((T[i-1,j,n]-2.*T[i,j,n]+T[i+1,j,n])/dx2+(T[i,j-1,n]-2.*T[i,j,n]+T[i,j+1,n])/dy2)+T[i,j,n]


# In[79]:


#Adicionar subplot
fig, ax = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True)

for i, n in enumerate(T[:,:,::25].T):
    ax[i].imshow(n)
    ax[i].set_title('t={}'.format(t[i*25]))
    
plt.show()


# In[80]:


plt.plot(t, T[5,5,:])
plt.title('Evolução da temperatura no centro da placa')
plt.xlabel('t')
plt.ylabel('T')
plt.show()


# In[81]:


del T, x, y, t, a, dx2, dy2, dt, i, n, ax, fig, j


# 3. **Convecção e difusão transiente unidimensional**. Resolver a EDP:
# \begin{equation}
#     \dfrac{\partial T}{\partial t}+u\dfrac{\partial T}{\partial x}=\alpha \dfrac{\partial^2 T}{\partial x^2} \quad 0\leq x \leq 1 ; 0\leq t \leq 8
# \end{equation}
# Condições de contorno:
# \begin{equation}
#     T(0,t)=T(1,t)=0
# \end{equation}
# Condição inicial:
# \begin{equation}
#     T(x,0) =  1 - ( 10 x - 1 )^2 \quad \text{ se $0 \leq x \leq 0,2$}, \quad \text{ senão } T(x,0) = 0
# \end{equation}
# Discretizando com as derivadas espaciais numa representação por diferença central e a derivada temporal com diferença ascendente:
# \begin{equation}
# \dfrac{T_{i,k+1}-T_{i,k}}{\Delta t}=\alpha \dfrac{T_{i-1,k}-2T_{i,k}+T_{i+1,k}}{(\Delta x)^2} -u\dfrac{T_{i+1,k}-T_{i-1,k}}{2\Delta x}
# \end{equation}

# O problema pode ser escrito na forma matricial como:
# \begin{equation}
# \begin{split}
# \begin{bmatrix} T_{0,k+1} \\ T_{1,k+1} \\ \vdots \\ T_{i,k+1} \\ \vdots \\ T_{n-2,k+1}\\ T_{n-1,k+1}\end{bmatrix} =
# \begin{bmatrix} T_{0,k} \\ T_{1,k} \\ \vdots \\ T_{i,k} \\ \vdots \\ T_{n-2,k}\\ T_{n-1,k}\end{bmatrix} +
# \frac{\alpha \Delta t}{(\Delta x)^2}
# \begin{bmatrix}
# 0 & 0 & & & & & \\
# 1 & -2 & 1 & & \\
#  & \ddots & \ddots & \ddots & & & \\
# & & 1 & -2 & 1 & \\
# & & & \ddots & \ddots & \ddots & \\
# & & & & 1 & -2 & 1\\
#  & & & & & 0 & 0
# \end{bmatrix}
# \begin{bmatrix} T_{0,k} \\ T_{1,k} \\ \vdots \\ T_{i,k} \\ \vdots \\ T_{n-2,k}\\ T_{n-1,k}\end{bmatrix} -
# \frac{u \Delta t}{2\Delta x}
# \begin{bmatrix}
# 0 & 0 & & & & & \\
# -1 & 0 & 1 & & \\
#  & \ddots & \ddots & \ddots & & & \\
# & & -1 & 0 & 1 & \\
# & & & \ddots & \ddots & \ddots & \\
# & & & & -1 & 0 & 1\\
#  & & & & & 0 & 0
# \end{bmatrix}
# \begin{bmatrix} T_{0,k} \\ T_{1,k} \\ \vdots \\ T_{i,k} \\ \vdots \\ T_{n-2,k}\\ T_{n-1,k}\end{bmatrix}
# \\
# \mbox{com $2\leq i \leq n -3$,}
# \end{split}
# \end{equation}

# ou simplesmente:
# \begin{equation}
#    T_{k+1} = T_{k} + \underbrace{ \Delta t \left( \alpha D_x - u D_x^2 \right)}_{A} T_{k},
# \end{equation}
# 
# onde $D_x$ e $D_x^2$ são os operadores diferenciais de primeira e segunda ordem, respectivamente.

# In[680]:


x, Dx, Dx2 = dv.ordem2(1., n=101)

t = np.linspace(0., 8., num=8001, endpoint=True)

dt = t[1]-t[0]

#Condições de contorno
for D in [Dx, Dx2]:
    D[0,:] = 0.
    D[-1,:] = 0.


# In[681]:


def convdiff(alpha, u):

    # Condição inicial
    T = np.zeros((x.size))
    for i, ival in enumerate(x):
        if ival > 0.2:
            break
        T[i] = 1. - (10. * ival - 1)**2.

    # Operador matricial
    A = dt*(alpha*Dx2-u*Dx)
    
    return A, T


# In[682]:


alpha, u, visu = 0.001, 0.08, 2000 #Parâmetro de visualização: a cada quantos passos de tempo se deve graficar os resultados

A, T = convdiff(alpha, u)
for n in range(t.size):
    T += A.dot(T)
    if n % visu == 0:
        plt.plot(x, T, label='t={}'.format(t[n]))

plt.xlabel('x')
plt.ylabel('T(x)')
plt.legend()
plt.show()


# Mas o que acontece quando combinados diferentes valores para $\alpha$ e $u$?

# In[683]:


fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
for i, alpha in enumerate([0., 0.001]):
    for j, u in enumerate([0., 0.08]):
        A, T = convdiff(alpha, u)
        for n in range(t.size):
            T += A.dot(T)
            if n % visu == 0:
                ax[i,j].plot(x, T, label='t={}'.format(t[n]))
                ax[-1,j].set_xlabel(r'$u = {}$'.format(u))
                ax[i,0].set_ylabel(r'$\alpha = {}$'.format(alpha))


# In[334]:


del A,D,Dx,Dx2,T,alpha,ax,dt,fig,i,j,n,t,u,visu,x
plt.close('all')


# 4. **Condução permanente com geração de calor**. Considere uma placa de urânio de espessura $L = 4cm$ e condutividade térmica $k = 28 W/m \cdot K$ em que o calor é gerado uniformemente a uma taxa constante de $\dot{e} = 5 \times 10^6 W/m^3$. Um dos lados da placa é mantido a $0^oC$ com água gelada, enquanto o outro lado está sujeito à convecção para o ambiente a $T_\infty = 30^oC$, com o coeficiente de transferência de calor $h=45 W/m^2 \cdot K$. Considerando o total de três pontos igualmente espaçados no meio, dois nos contornos e um no centro, estime a temperatura da superfície exposta da placa sob condições permanentes usando a abordagem de diferenças finitas.

# In[688]:


L = 0.04 #m
k = 28.0 # W/mK
eg = 5.0e6 # W/m^3
T8 = 30.0 # graus C
h = 45.0 # W/m^2K
n = 3


# <img src="uranio.png">
# 
# Escrevendo a equações para cada nó:
# \begin{equation}
# T_0 = 0 \\
# \dfrac{T_0 - 2 T_1 + T_2}{\Delta x^2} + \dfrac{\dot{e}_1}{k} = 0 \\
# h(T_\infty - T_2) + k\dfrac{T_1 - T_2}{\Delta x} + \dot{e}(\Delta x/2) = 0
# \end{equation}

# O problema pode ser reescrito na forma matricial como:
# \begin{equation}
# \begin{split}
# \underbrace{
# \begin{bmatrix}
# 1 & 0 & 0 \\
# 1/\Delta x^2 & -2/\Delta x^2 & 1/\Delta x^2 \\
# 0 & k/\Delta x & -h-k/\Delta x
# \end{bmatrix}
# }_{A}
# \begin{bmatrix}
# T_{0} \\ T_{1} \\ T_{2}
# \end{bmatrix}
# =
# \underbrace{
# \begin{bmatrix} 0 \\ -\dot{e}/k \\ -\dot{e}\Delta x/2 -hT_\infty \end{bmatrix}
# }_{B}
# \end{split}
# \end{equation}

# In[689]:


x = np.linspace(0., L, num=n, endpoint=True)
dx = x[1] - x[0]

A = np.array([[1., 0., 0.],
              [1./dx**2., -2./dx**2., 1./dx**2.],
              [0., k/dx, -h-k/dx]])
B = np.array([0., -eg/k, -eg*(dx/2.)-h*T8])

# Solver de álgebra linear do pacote numpy
T = np.linalg.solve(A,B)


# In[690]:


# Escrevendo o resultado na tela
for i, t in enumerate(T):
    print(f'T{i} = {t:5.2f}')


# Esse problema pode também ser resolvido analiticamente como:
# \begin{equation}
#     T(x) = \dfrac{0,5 \dot{e}hL^2/k + \dot{e}L + T_\infty h}{hL + k}x - \dfrac{\dot{e}x^2}{2k}
# \end{equation}

# In[691]:


def T_exata(x):
    return x*(0.5*eg*h*L**2./k + eg*L + T8*h)/(h*L +k) - 0.5*eg*x**2./k


# In[692]:


# Resposta numérica obtida anteriormente
plt.plot(x,T, label=f'n={n}')

# Solução exata
x_exato = np.linspace(0., L, num=101, endpoint=True)
plt.plot(x_exato,T_exata(x_exato),
         label='Analítica',
         color='silver',
         ls='--')

plt.legend()
plt.xlabel('x[m]')
plt.ylabel(r'T[$^o$C]')
plt.show()


# O sistema matricial pode ser escrito de maneira genérica para uma decomposição em $n$ pontos, como:
# \begin{equation}
# \begin{split}
# \begin{bmatrix}
# 1 & 0 & 0 \\
# & \ddots & \ddots & \ddots \\
# & & 1/\Delta x^2 & -2/\Delta x^2 & 1/\Delta x^2 \\
# & & & \ddots & \ddots & \ddots \\
# & & & 0 & k/\Delta x & -h-k/\Delta x
# \end{bmatrix}
# \begin{bmatrix}
# T_{0} \\ \vdots \\ T_{i} \\ \vdots \\ T_{n}
# \end{bmatrix}
# =
# \begin{bmatrix} 0 \\ \vdots \\ -\dot{e}_i/k \\ \vdots \\ -\dot{e}_n\Delta x/2 -hT_\infty \end{bmatrix}
# \\
# \mbox{com $1\leq i \leq n -1$,}
# \end{split}
# \end{equation}

# In[694]:


# Agora vamos testar a convergência da solução, para três valores de n
for n in [3, 5, 10]:
    x, Dx, Dx2 = dv.ordem2(L, n=n)
    dx = x[1] - x[0]

    A = Dx2
    B = 0. * x - (eg/k)

    ## Condição de contorno em x=0
    A[0,0] = 1.
    A[0,1:] = 0.
    B[0] = 0.
    ## Condição de contorno em x=L    
    A[-1,-3] = 0.
    A[-1,-2] = k/dx
    A[-1,-1] = -h-k/dx
    B[-1] = -eg*(dx/2.)-h*T8

    T = np.linalg.solve(A,B)

    plt.plot(x,T, label=f'n={n}')

plt.plot(x_exato,T_exata(x_exato),
         label='Analítica',
         color='silver',
         ls='--')

plt.xlabel('x[m]')
plt.ylabel(r'T[$^o$C]')
plt.legend()
plt.show()


# In[695]:


del L, k, eg, T8, h, n, x, dx, A, B, T, i, t, x_exato
plt.close('all')


# 5. **Armazenamento de energia solar em paredes de Trombe**.
# 
# As paredes de alvenaria pintadas de cor escura, chamadas paredes de Trombe, são comumente usadas do lado sul das casas com energia solar passiva para absorver energia solar, armazenálas durante o dia e liberá-la para a casa durante a noite (mais informações [aqui](https://en.wikipedia.org/wiki/Trombe_wall)). Normalmente uma camada única ou dupla de vidro é colocada fora da parede e transmite a maior parte da energia solar, bloqueando as perdas de calor da superfície exposta da parede externa. Além disso, saídas de ar são comumente instaladas nas partes inferior e superior entre a parede de Trombe e os vidros. Em sua passagem, o ar tem a temperatura elevada e entra na sala através da abertura no lado superior da parece.
# 
# <img src="casa_1.png" width="300">
# 
# Considere uma parede Trombe com $30cm$ de espessura, condutividade térmica de $k=0,69W/m\cdot K$ e difusividade térmica $\alpha = 4,44 \times 10^-7 m^2/s$. A variação de temperatura ambiente $T_{sai}$ e do fluxo de calor solar $\dot{q}_{solar}$ inidente sobre a face vertical da superfície virada ao Sul ao longo de um dia típico de janeiro é dada por
# 
# |Período do dia| Temperatura ambiente, $^oC$|Radiação solar, $W/m^2$|
# |--|--|--|
# |7h-10h|0,6|360|
# |10h-13|6,1|763|
# |13h-16h|7,2|562|
# |16h-19h|2,8|0|
# |19h-22h|0|0|
# |22h-1h|-2,8|0|
# |1h-4h|-3,3|0|
# |4h-7h|-3,9|0|
# 
# A parede de Trombe tem vidro simples com um produto de absortividade e transmissividade de $ \kappa = 0,77$ (ou seja, 77% da energia solar incidente é absorvida pela superfície exposta da parede de Trombe), e o coeficiente médio de transferência de calor combinada para perda de calor da parede de Trombe no ambiente é determinada a $h_{sai} = 4 W/m^2 \cdot C$. O interior da casa é mantido a $T_{ent} = 21 ^oC$ todo o tempo, e o coeficiente de transferência de calor na superfície interna da parede de Trombe é $h_{ent} = 10 W/m^2 \cdot ^oC$. Além disso, as aberturas na parede são mantidas fechadas, portanto a transferência de calor entre o ar da casa e a parede de Trombe ocorre apenas através da superfície interior da parede. Pressupondo que a temperatura da parede de Trombe varie linearmente entre $21 ^oC$ na superfície interna e $-1 ^oC$ na superfície externa às 7 h da manhã e usando o método explícito das diferenças finitas com espaçamento nodal uniforme de $\Delta x = 6 cm$, determine a distribuição de temperatura ao longo da espessura da parede de Trombe após 12, 24, 36 e 48 h. Além disso, determine a quantidade líquida de calor transferido da parede de Trombe para a casa durante o primeiro e o segundo dia. Considere que a parede tem $3m$ de altura e $7,5m$ de comprimento e utilize um passo de tempo de $\Delta t = 15min$. 

# In[706]:


L = 0.3 # m
dx = 0.06 # m
dth = 0.25 # h
k = 0.69 # W/mK
alpha = 4.44e-7 # m^2/s
kappa = 0.77
hsai = 4 # W/m^2K
Tent = 21.0 # graus C
hent = 10 # W/m^2K


# In[707]:


table = pd.DataFrame({
    "ti": [7., 10., 13., 16., 19., 22., 1., 4.],
    "tf": [10., 13., 16., 19., 22., 1., 4., 7.],
    "Tsai": [0.6, 6.1, 7.2, 2.8, 0.0, -2.8, -3.3, -3.9],
    "qsol": [360., 763., 562., 0., 0., 0., 0., 0.]
})


# <img src="casa_0.png" width="300">
# 
# Escrevendo a equações para cada nó:
# \begin{equation}
# T_0^{j+1} = \left( 1 - 2\tau -2 \tau \dfrac{h_{ent}\Delta x}{k} \right) T_0^{j} + 2\tau T_1^{j} + 2\tau \dfrac{h_{ent}\Delta x}{k} T_{ent} \\
# T_i^{j+1} = \underbrace{\tau (T_{i-1}^j + T_{i+1}^j) + (1-2\tau)T_{i}^j}_{\Delta t \alpha D_x^2 + I}, \mbox{com $1\leq i \leq n-1$,}\\
# T_n^{j+1} = \left( 1 - 2\tau -2 \tau \dfrac{h_{sai}\Delta x}{k} \right) T_n^{j} + 2\tau T_{n-1}^{j} + 2\tau \dfrac{h_{sai}\Delta x}{k} T_{sai}^j + 2\tau \dfrac{\kappa \dot{q}_{solar}^j \Delta x}{k}
# \end{equation}
# onde
# 
# \begin{equation}
# \tau = \dfrac{\alpha \Delta t}{\Delta x^2}
# \end{equation}

# O problema pode ser reescrito na forma matricial como:
# \begin{equation}
# \begin{split}
# \begin{bmatrix}
# T_{0}^{j+1} \\ \vdots \\ T_{i}^{j+1} \\ \vdots \\ T_{n}^{j+1}
# \end{bmatrix}
# =
# \underbrace{
# \begin{bmatrix}
# 1 - 2\tau -2 \tau \dfrac{h_{ent}\Delta x}{k} & 2\tau & 0 \\
# \ddots & \ddots & \ddots \\
# \tau & 1-2\tau & \tau \\
# \ddots & \ddots & \ddots \\
# 0 & 2\tau & 1 - 2\tau -2 \tau \dfrac{h_{sai}\Delta x}{k}
# \end{bmatrix}
# }_{A}
# \begin{bmatrix}
# T_{0}^j \\ \vdots \\ T_{i}^j \\ \vdots \\ T_{n}^j
# \end{bmatrix}
# +
# \underbrace{
# \begin{bmatrix} -2\tau \dfrac{h_{ent}\Delta x}{k} T_{ent} \\ \vdots \\ 0 \\ \vdots \\ - 2\tau \dfrac{h_{sai}\Delta x}{k} T_{sai}^j - 2\tau \dfrac{\kappa \dot{q}_{solar}^j \Delta x}{k} \end{bmatrix}
# }_{B}
# \\
# \mbox{com $1 \leq i \leq n -1$,}
# \end{split}
# \end{equation}

# O critério de estabilidade para esse problema é dado por:
# \begin{equation}
# \Delta t \le \dfrac{\Delta x^2}{3,74 \alpha} = 36,13min
# \end{equation}

# In[723]:


x, Dx, Dx2 = dv.ordem2(L, n=int(L/dx + 1))
time = np.arange(0., 48.+dth, dth)

T = np.zeros((x.size,time.size))

#convertendo para segundos
dt = dth * 3600.

tau = alpha * dt / dx**2.

# Condição Inicial
T[:,0] = 21.0 - (22.0/L)*x
time += 7.

# Construção dos Operadores
A = dt*(alpha*Dx2) + np.eye(x.size)
B = np.zeros_like(x)

# Laço de avanço temporal
for j in range(time.size-1):
    # Obter os valores do contorno à partir da tabela
    index = int((time[j] - 7. )% 24. // 3)
    Tsai = table.loc[index].Tsai
    qsol = table.loc[index].qsol
    # Condição de contorno em x=0
    A[0,0] = 1.-2*tau-2.*tau*hent*dx/k
    A[0,1] = 2.*tau
    A[0,2] = 0.
    B[0] = 2.*tau*hent*dx*Tent/k
    # Condição de contorno em x=L    
    A[-1,-3] = 0.
    A[-1,-2] = 2.*tau
    A[-1,-1] = 1.+-2.*tau -2.*tau*hsai*dx/k
    B[-1] = 2.*tau*hsai*dx*Tsai/k + 2*tau*kappa*qsol*dx/k
    #
    T[:,j+1] = A.dot(T[:,j]) + B


# * Visualizando os resultados:

# In[727]:


for j in range(0,time.size,20):
    plt.plot(x,T[:,j],label=f't={time[j]}')

plt.legend(title='tempo [h]')
plt.xlabel('x [m]')
plt.ylabel(r'Temperatura [$^o$C]')
plt.show()


# In[749]:


plt.pcolormesh(x,time,T.T, cmap='coolwarm')

plt.colorbar(label='Temperatura [$^o$C]', orientation='vertical') 

plt.xlabel('x [m]')
plt.ylabel(r'Tempo [h]')
plt.show()


# * Variação da temperatura em três pontos da parede, em função do tempo

# In[732]:


for i in [0, x.size//2, -1]:
    plt.plot(time,T[i,:],label=f'{x[i]}')

plt.legend(title='x [m]')
plt.xlabel('tempo [h]')
plt.ylabel(r'Temperatura [$^o$C]')
plt.show()


# In[656]:


plt.close('all')

