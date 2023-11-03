
% importo dei fatti provenienti da normalizedDataset
:- consult('factsKB.pl').

% Definizione della funzione sigmoide
sigmoide(X, Y) :-
    Y is 1 / (1 + exp(-X)).

% Calcolo del rating utilizzando la sigmoide
ratio(Rating, Title) :-
    hasVoteC(Title, hasVoteC, VoteC),
    hasVoteAv(Title, hasVoteAv, VoteAv),
    hasPop(Title, hasPop, Pop),
    RawRating is (VoteC * 0.5 + VoteAv * 0.5 + Pop * 0.7),
    sigmoide(RawRating, Rating).










