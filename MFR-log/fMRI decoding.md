### Decoding on real wall time instead of TRs

Nico reminded me that the trials aren't TR locked. That means I should be able to reconstruct the exactl probability curve per participant as the TRs are distributed across the time window after stimulus presentation. It was quite a hassle to reverse-engineer the wall-time that belongs to each TR, but after all I managed. 

I get this beautiful smooth curve. However, something is still off as there are clear shifts within the data. I guess something about the `tr_onset` calculation variable and rounding is still off. Using this I should be able to estimate individual response curves for each participants.

![[Pasted image 20250710150242.png]]]]

Similarly looking when decoding the sequence. Something is slightly off with the jittering of the probabilities. I assume it would look much better when getting precise timings. However, I only have the `round(onset)` in the original script, not sure how to fix that?
![[Pasted image 20250718110157.png]]

Additionally, it seems like this picture only emerges after we average across all participants. When looking at individual participants, there's no such picture visible. I will probably not get very far with trying to interpolate this data any way I try. The returning to 0 probability once in a while makes this much harder.
![[Pasted image 20250718102823.png]]

I interpolated individual trials either by linear or by polyfit. Still no effect visible. Too bad! It makes sense that it fails given that there's only 4x32ms difference but the search window is 16 seconds long. However, there IS a peak at e.g. 164ms, which is already pretty cool!

![[Pasted image 20250718151511.png]]]]