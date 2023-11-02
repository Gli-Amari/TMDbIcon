




% Calcola gradimento bilanciato con normalizzazione e determina Ratio
gradimento_bilanciato(VoteAverage, VoteCount, Popularity, Ratio) :-
    Ratio is ((VoteAverage * 0.5 + VoteCount * 0.5 + Popularity * 0.7)*100).










