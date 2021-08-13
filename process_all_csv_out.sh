set -x
# set -e
for el in $(ls /data_disk/ds_ro/discarded_object_dataset/videos/*);do
	x=$(basename $el)
	x="${x%.*}"	
	SCRIPT="deepsort_gen_lineplot.py"
	BBOXES_INFERRED="/data_disk/ds_ro/discarded_object_dataset/bboxes_inferred"
	PROCESSED_VIDEOS="/data_disk/ds_ro/discarded_object_dataset/processed_videos/deepsort_gen1"
	echo $el
	python $SCRIPT $el "$BBOXES_INFERRED/$x.csv" "$PROCESSED_VIDEOS/${x}.mp4"  "$PROCESSED_VIDEOS/${x}.csv" 
done
