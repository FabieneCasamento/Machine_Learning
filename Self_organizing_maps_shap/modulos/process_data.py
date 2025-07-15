import pandas as pd


def tratar_dominician_rep(x):
    if x == 'Dominican Republic':
        return 'Dom. R.'
    return x


def niveis_facilidade(score: pd.Series) -> list:
    valor_quartils = score.quantile([.25, .5, .75]).to_list()
    q1, q2, q3 = valor_quartils

    def classificacao(x, q1, q2, q3):
        if x < q1:
            return "Low facility"
        if x < q2:
            return "Lower middle facility"
        if x < q3:
            return "Upper middle facility"
        else:
            return "High facility"

    niveis = [classificacao(x, q1, q2, q3) for x in score.to_list()]
    return niveis


if __name__ == '__main__':
    pass
