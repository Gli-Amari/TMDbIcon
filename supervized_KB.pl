

% Regola: Calcola Gradimento
gradimento(VoteAverage, VoteCount, Popularity, Rating, Result) :-
    Rating is ((VoteAverage * 0.3 + VoteCount * 0.3 + Popularity * 0.5) * 100),
    (Rating >= 40 -> Result = 1; Result = 0).


%regola logistica







