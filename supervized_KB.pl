

% Regola: Calcola Gradimento
gradimento(VoteAverage, VoteCount, Popularity, Rating, Result) :-
    Rating is ((VoteAverage * 0.4 + VoteCount * 0.4 + Popularity * 0.5) * 100),
    (Rating >= 50 -> Result = 1; Result = 0).

%Relazioni
has_rating('Film1', rating, gradimento).
has_rating('Film2', rating, gradimento).
has_rating('Film3', rating, gradimento).
has_rating('Film4', rating, gradimento).

has_vote_average('Film1', vote_average, 1.000000).
has_vote_average('Film2', vote_average, 0.800000).
has_vote_average('Film3', vote_average, 0.950000).
has_vote_average('Film4', vote_average, 0.700000).

has_vote_count('Film1', vote_count, 1.000000).
has_vote_count('Film2', vote_count, 0.800000).
has_vote_count('Film3', vote_count, 0.950000).
has_vote_count('Film4', vote_count, 0.700000).

has_popularity('Film1', popularity, 1.000000).
has_popularity('Film2', popularity, 0.800000).
has_popularity('Film3', popularity, 0.950000).
has_popularity('Film4', popularity, 0.700000).



