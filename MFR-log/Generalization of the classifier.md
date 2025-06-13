
To run the Wittkuhn+Schuck Method, we need our classifiers to better generalize across time, else we have no overlapping stimuli. Therefore I'll try some experiments how to achieve this in the MEG.

### Train on several timesteps
Currently, we train on the peak timepoint of decodability. What happens if we train on several time steps? Will the generalization increase?

First to get a nice overview, the accuracies for various ranges. Image onset was at `t=20`. It does seem to generalize a bit better, but not nearly enough to be relevant at the indicated peaks of 32/64/128 image presentations. Even though a slight match might make it to the 32ms peak (~182ms = 100+32+150). Interestingly enough, training on the entire time series (pink) does not seem to be very good either, as this removes the peak again. Very weird! I assume that the logistic regression is not able to capture the changes in the dynamic very well. Actually training on 45-50 seems to be pretty decent.

300 x 306
5x300 x 306
![[Pasted image 20250110113928.png]]
Edit: Nope, using a RFC does not solve the problem and can't deal with the data better
![[Pasted image 20250110133051.png]]

Additionally, I created a TimeEnsembleVoting, e.g. train one classifier per time step and do a voting in the end. To my surprise this did not improve the classification accuracy at all!
![[Pasted image 20250110154338.png]]

Here is the accuracy evaluated at the dotted lines above, when training a classifier from `t1:t2`. Interesting that the there seems to be a dip when training at 100-200? which is also visible in the plot above, something strange going on. Generally, training on later time courses without the actual peak seems to be superior, actually 250ms-300ms after image onset? 
I wonder if this might bring the LogReg to it's limits and it can't handle the data diversity. Will try again with a random forest, but takes some days to train. I'm actually puzzled that the classifier doesn't simply generalize to all datapoints when trained on more data.

![[Pasted image 20250110100151.png]]
On average, training from 250-350 ms seems to have the broadest effect on all three markers.
![[Pasted image 20250110112515.png]]

### Training on down-sampled data
I down-sampled the data to 10 Hz and looked if the decoding time course is now significantly different and longer, as Gordon suspected. That indeed spreads the activation nicely and sinusoidal, it even looks very similar to fMRI.

![[Pasted image 20250110132848.png]]

I do need to downsample both signals though to get to that results. no idea how that will behave with several classifiers
![[Pasted image 20250110141219.png]]

I guess I should try a combination of the down/upsampling plus the differently trained classifiers?

This is what it looks like then when decoding the fast trials using these classifiers. However note, that individual trials are still extremely noisy.
![[Pasted image 20250115112134.png]]