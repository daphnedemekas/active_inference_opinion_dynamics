

""" 

SELF ESTEEM 

observation: [what i tweeted, what my neighbours tweeted, who i observed] 

We want to add an additional observation modality which corresponds to how well I correspond to my neighbours (my self esteem)

This will be based on some metric that gets higher if me and my neighbours tweet the same thing (or the same sequence of things) and lower if not

OBSERVATOIN - self esteem [ reward, rejection] 

reward vs rejection is just defined by a threshold on the metric described above ^ 

you have a preference over level 1 (Reward) in the C matrix so you prefer to agree and that should drive you to tweet what your neighbours tweet 
if you are being rejected, this should drive you to change who you choose to select to observe 


The agent will then have a preference encoded in the C matrix to be closer to the group 

if rejection occurs with agent A then preference to sample agent A goes down 

if rejection occurs then self esteem gets lower 


SEQUENCING 

B matrix p (st | s(t-1) )

for some agent, tweeting hashtag 0 implies that it will have a higher probability to tweet hashtag 1 after, and tweeting hashtag 1 increases the probability of tweeting hashtag 2 

012, 12, 012, 

given the fact that i tweeted 0 AND i'm still "interacting" with the same agent (i observed the same agent again) then the probability would increase to tweet hashtag 1 
otherwise if the agent changes i get reset 

need to give the agents some kind of memory 

we can have another observation modality which is [no sequence sequencing, sequenced ] 

we write a function that defines the existence of a sequence in terms of the neighbour and the tweets which creates that observation

and the agent has a preference over sequencing and a higher preference over sequenced 


sequences are like [[AAB, BAB, BBB ]]etc and each agent gets initialized with a preference over one of the sequences 
that defines the B matrix 


def track_sequence(agent_observed, own_tweet, neighbour_tweet): 
    if agent_observed is the same, and tweet follows sequence:
        sequencing += 1

parameters of the B matrix -- 

a parameter which is > 1 / number of neighbours which is the probability of re-interacting (learning rate) 
a way of coding in general how your selected sequence (AAB for example) will affect your transition probability 

A--A if before we did not have A the question of memory 

AAB

if you are in A- A or B? ABA BABABABABAB

ABCDE 

ABCDE 
EDABC 

evade the necessity for memory to start with (no repetition)


Create a new state which is mapped in A to both self esteem and sequencing 
and that influences the probability of resampling in the B matrix 
"""