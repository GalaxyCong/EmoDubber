rm all_scores.txt
yourfilenames=`ls $1`

for eachfile in $yourfilenames
do
   python run_pipeline_junchen.py --videofile $1/$eachfile --reference wav2lip --data_dir tmp_dir
   python calculate_scores_real_videos.py --videofile $1/$eachfile --reference wav2lip --data_dir tmp_dir >> all_scores.txt
done


awk '{sum2 += $2; sum3 += $3; count++} END {if (count > 0) print sum2/count, sum3/count}' all_scores.txt


