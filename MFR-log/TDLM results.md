## limiting valid max_true_trans

I've implemented limiting the maximum number of valid transitions in the permutations. Results change, but not as much as expected. 
Weird: For 32 it decreases as expected, for 64/128 not much change, for 512 it fluctuates. This makes no sense to me!

![[Pasted image 20250124120255.png]]
## 2-step TDLM on supertrials

The 2step TDLM looks weird. It does show peaks at times when we expect it, but their significance is drastically below what we would expect. This might be because of the permutation problem though, as we have many overlapping permutations in the set that are tested against! 
And the current implementation can only remove overlapping permutations for the second step (A\*B) -> C and not the first. I'm not sure if that actually removes the problem, because we are still testing in the first loop against all A->X.

![[Pasted image 20250124115316.png]]
## TDLM on supertrials

Decoding on individual trials didn't seem to work very well, so for the next analysis step, I made a mean of all trials that had the same sequence and the same speed. That is, afaik, 4 trials per combination. Then, the TDLM sequenceness works excellently! However, It is barely significant.
Somehow, this looks very different, I can't replicate it :-/


![[Pasted image 20250108100623.png]]

## Sequenceness fast images
#TDLM
I have run TDLM on the above mentioned images sequenceness. The results look interesting, but not very pretty :-/ it seems like there is something out of order. I'm not sure what is going on. But there is definitively *something* going on!
![[fast_images_sequenceness_zscored_all.png]]
Weird: Some participants have very odd oscillations around 5 Hz. I assume that this is too regular to be biological. Maybe the decoding doesn't work at all and the signals are saturate the classifier? I'll have to investigate later.

![[fast_images_sequenceness_zscored_16.png]]
