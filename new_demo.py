

""" 

SELF ESTEEM 
Desired outcome: seeing groups formed based on the reward observation modality rather than only the beliefs 

observation: [what i tweeted, my reward, what my neighbours tweeted, my neighbours reward,  who i observed] 

[H, R] 
my neighbours reward, - a function of the generative process 

We want to add an additional observation modality which corresponds to how well I correspond to my neighbours (my self esteem)

OBSERVATION - self esteem [ reward, neutral, rejection] 

the mapping in the A matrix p(my reward | states) is a function of similarity between beliefs about idea and beliefs about the neighbours' beliefs

p( my reward | states) -- if what you believe your neighbour believes aligns with what you believe then you observe reward (depending on the extent of agreement)

p(my neighbours reward | states) -- observing your neighbours' reward lends evidence to the whole network (yuor  beliefs about all of your neighbours beliefs) 
bumps it up in the direction of what your observed neighbour believes (smaller update than the update to the directly observed neighbour)

--C--
preference over high reward 


Next level: include a hidden state factor and another observation modality, hidden state factor that maps to the action of "liking" a tweet and the observation modality 
which incentivizes liking a person's tweet 



SEQUENCING (mirroring)

ABCDEF

hidden state: [no sequence, sequencing, sequence finished]
p(observe A, : | sequence started)
p(observe not A or C | in sequence)
p(mirroring | in sequence)
p(not mirroring | not in sequence)
p(observe F | sequence finished)


p(sampling not X | sampled X, in sequence) = 0

preference over mirroring -- observation modality of mirroring 



seuqence observation [no sequence, sequencing, sequence complete]
you have a temporal preference to start and complete 





p(neighbour tweeted A, : | sequence started)
p(observe not A or C, neighbour hasn't changed | in sequence)
p(observe F, neighbour hasn't changed | sequence finished)

hidden state that corresponds to what stage of the sequence you're in [no sequence, sequencing, sequenced ] 
this hidden state maps to the observation of what your agents will tweet 

p(neighbour_i tweeting B | and they tweeted A)

that observation maps to whether or 

add a hidden state factor about what they are tweeting (isomorphic to observation)
your belief about what teh sequence is affects your belief about what they will tweet 

learning a b matrix <--> identifying a sequence 

preference over mirroring 

first iteration: you have a hardcoded B matrix and you learn the B matrix of your agents
B[indexed at the states that correspond to your neighbours tweet 

beliefs about sequences of who you aren't looking at decays if you're not looking at them 


ABC - sequence itself is encoded in B matrix 

hidden state factor: where in the sequence you are [no sequence sequencing, sequenced ] 

A A -- p(observing A | tweeted A,)

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