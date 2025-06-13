


### Decoding with generalizing classifiers

I trained classifiers on down&upsampled data and then also down&upsampled the fast images to 100 Hz ->10 Hz -> 100 Hz, now there is some overlap!
![[Pasted image 20250115112207.png]]


## Decoding working!

I finally got the decoding of the sequence trials working. Seems like I made a mistake in the assignment of the sequence number and the image number, which was easy to fix.

Now there are clear peaks visible.
Interesting: 
 - First image is best decodable in most cases.
 - There is little overlap, so W&S Method will not work


![[Pasted image 20250108100218.png]]

Individual trials are borderline to non-decodable. That's probably also the reason why the decoding doesn't work for TDLM on individual trials. 
![[Pasted image 20250108100336.png]]
Interesting also: Phase shifted alpha oscillations in my data :)
![[Pasted image 20250108100440.png]]


### First decoding try: not working
I tested two variants of the visualization of showing the four images to remove the baseline classifier drift. One uses dividing by the mean, the others zscoring, for each trial, for each probability time series per person, that is..
Seems like dividing by the mean is the better option. It is apparent that there is a peak when the images are being processed. But doesn't seem to be super predictive for which image exactly is shown? weird.
Additionally, it seems like the order is somehow wrong. There is a clear peak, but at the wrong times!

**divided by mean proba**
![[Pasted image 20241127161017.png]]

**zscored:**
![[Pasted image 20241127161144.png]]

## Debugging the images
It seems like the decoding is working, but somehow the positions seems to be mixed up. Therefore we also don't see any effect on the sequenceness, which makes total sense!
Very strange! I'll have to dig deeper.

The decoding seems to work, so at least within the localizer everything seems to be consistent.

#### check trigger sending
First of all, trigger values seem to be hard coded to images. So I doubt that there can be any mixup happening here! absolutely impossible!

```python
trigger_img = {}
trigger_img['Gesicht'] = 1
trigger_img['Haus']   = 2
trigger_img['Katze']   = 3
trigger_img['Schuh']   = 4
trigger_img['Stuhl']   = 5
```
During the localizer, they are sent accordingly
```python
img_idx = trigger_img[get_image_name(df_trial.img)]
if is_distractor:
	send_trigger(trigger_base_val_localizer_distractor + img_idx)
else:
	send_trigger(trigger_base_val_localizer + img_idx)
```
Also during the sequence trial they are sent accordingly
```python
for img_x, state in enumerate(img_started):
    if state: continue
    component_img = locals()[f'sequence_img_{img_x+1}']
    if component_img.status==STARTED:
        img_started[img_x] = 1
        img_id = trigger_img[get_image_name(component_img.image)]
        send_trigger(trigger_base_val_sequence + img_id)

```
#### Check that images have same IDX within subject
**Participant 02**

**csv**
	For the localizer first trial, I have `Schuh, Haus, Gesicht, Gesicht -> 4, 2, 1, 1
- for the first sequence I have `Stuhl, Gesicht, Katze, Schuh, Haus -> 5, 1, 3, 4, 2`
**logs**
	uuuuh the logs dont seem to match at all with the CSV! 
- here we have localizer `Katze, Haus, Gesicht, Gesicht, Stuhl` 
- and for the sequence `Katze Haus Stuhl Schuh Gesicht`
**FIF**
	phew! at least the FIF file matches the logs.
- we have localizer `103, 2, 1, 1, 5, 105, 3, 1, 2, 2 -> Katze, Haus, Gesicht, Gesicht, Stuhl, Stuhl, Katze, Gesicht`  
- for sequence `23, 22, 25, 24, 21 -> Katze, Haus, Stuhl, Schuh, Gesicht`
- Interval of first sequence trials is also correct! ~128+100 ms between images in the FIF
**events_beh.tsv**
- localizer: `Katze, Haus, Gesicht, Gesicht, Stuhl -> 3, 2, 1, 1, 5 `
- sequence: `Katze, Haus, Stuhl, Schuh, Gesicht -> 3, 2, 5, 4, 1`
**python**
	indexing starts at 0 here. It matches, minus the distractors with >100 trigger
- localizer: `1, 0, 0, 4, 2 ->  Haus, Gesicht, Gesicht, Stuhl, Katze`
- sequence: `2, 1, 4, 3, 0 -> Katze, Haus, Stuhl, Schuh, Gesicht,`
I guess someone just selected the wrong number while starting the exp
let's check again for 

**Participant 15**
**csv**
- localizer: `Schuh, Haus, Haus, Katze, Gesicht -> 4, 2, 2, 3, 1`
- sequence `Stuhl, Gesicht, Katze, Haus, Schuh -> 5, 1, 3, 2, 4`
**logs**
 * localizer `Katze, Haus, Haus, Schuh, Gesicht -> 3, 2, 2, 4, 1`
* sequence: `Stuhl, Gesicht, Katze, Haus, Schuh -> 5, 1, 3, 2, 4`

Ok the sequence matches, but the localizer doesn't??? wtf??

Very weird, but I'm a bit clueless to how that can happen. Maybe a different random seed while creating the images and I altered them later? sounds like the most likely possibility.

ToDo: 
  - [ ] check sequence intervals are matching
  - [ ] check raw files once again
  - [ ] check that experiment shows what it logs
  - [ ] decode fast images individually per class
  - [ ] OVA vs multiclass

