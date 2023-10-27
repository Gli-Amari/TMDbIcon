

% Regola: Calcola Gradimento
gradimento(VoteAverage, VoteCount, Popularity, Rating, Result) :-
    Rating is ((VoteAverage * 0.4 + VoteCount * 0.4 + Popularity * 0.5) * 100),
    (Rating >= 30 -> Result = 1; Result = 0).










