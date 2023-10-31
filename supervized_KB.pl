




% Calcola gradimento bilanciato con normalizzazione e determina Ratio
gradimento_bilanciato(VoteAverage, VoteCount, Popularity, Ratio) :-
    Rating is ((VoteAverage * 0.5 + VoteCount * 0.5 + Popularity * 0.7) * 100),
    MinValue = 0,
    MaxValue = 100,
    NormalizedValue is (Rating - MinValue) / (MaxValue - MinValue),
    (NormalizedValue >= 0.50 -> Ratio = 1; Ratio = 0).










