from scipy.sparse import diags
from numpy import linspace, pi
def ordem2(f=2*pi, i=0., n=7, P=False):
    '''ordem2(f=2*pi, i=0., n=7, P=False)

    Calcula os operadores diferenciais da primeira e segunda derivada, para uma
    malha equidistante de i a f, com n pontos, e erro da ordem de h**2.

    Parâmetros
    ----------
    f : real
        Valor do contorno superior de x.
        Padrão é 2*pi.
    i : real
        Valor do contorno inferior de x.
        Padrão é zero.
    n : inteiro
        Número de pontos da malha na direção x.
        Padrão é 7.
    P : bool, opcional
        Define que a condição de contorno é periódica quando True.
        Padrão é False.

    Retorna
    -------
    x, Dx e Dx2, respectivamente o vetor posição e os operadores diferenciais
    para primeira e segunda ordem.
    '''
    #Vetor posição
    x = linspace(i, f, num=n, endpoint=not P)
    #Operador diferencial de primeira ordem
    Dx = diags([-1., 0., 1.],
              offsets=[-1, 0, 1],
              shape=(x.size,x.size)
              ).toarray()
    #Operador diferencial de segunda ordem
    Dx2 = diags([1., -2., 1.],
               offsets=[-1, 0, 1],
               shape=(x.size,x.size)
               ).toarray()
    #
    if P: #Condições de contorno Periódicas
        for D in [Dx, Dx2]:
            D[0,-1] = D[1,0]
            D[-1,0] = D[-2,-1]
    else: #Não Periódica
        Dx[0,0], Dx[0,1], Dx[0,2] = -3., 4., -1.
        Dx[-1,-3], Dx[-1,-2], Dx[-1,-1] = 1., -4., 3.
        Dx2[0,0], Dx2[0,1], Dx2[0,2] = 1., -2., 1.
        Dx2[-1,-3], Dx2[-1,-2], Dx2[-1,-1] = 1., -2., 1.
    #
    h = (x[1]-x[0])
    Dx /= 2.*h
    Dx2 /= h**2.
    return x, Dx, Dx2
