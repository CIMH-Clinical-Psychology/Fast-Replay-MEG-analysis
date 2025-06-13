I've created the first decoders and it seems like they can decode the 5 localizer images very well! The C parameter seems to be slightly different than in my previous studies (being 20 or something? don't remember ?).
However, things that were buggy: Somehow during downsampling, the trigger channel became useless as some values were shifted. Need to fix that
![[Pasted image 20241125112351.png]]

Here, visualized across different values for C. Seems to be stable after `C=5-20`
![[decodingmfr_L1.gif]]
Peak C is at 
![[Pasted image 20241125161353.png]]

Now looking at each class individually, like in Lennarts paper
![[Pasted image 20241127171844.png]]
## OVA not working?
However, there is a small dip in the other class at the time when the target class peaks. That makes sense, as the classifiers are trained multiclass. However, what is weird, is that even when I train them OVA there still seems to be the same dip! how is that possible? My intuition is that is should remove the dip, at least in the summary plot?

![[Pasted image 20241127172943.png]]