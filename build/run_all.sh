#!/bin/bash
LOG=accuracy_log.txt
echo "Batch,Correct,Accuracy(%),Diff,Time(s)" > $LOG
total_correct=0
total_diff=0

for b in {0..9}
do
    echo "===== Running batch $b ====="
    ./main --batch=$b > tmp_$b.log 2>&1

    acc_line=$(grep "Accuracy=" tmp_$b.log)
    echo "Batch $b $acc_line" >> $LOG

    correct=$(echo "$acc_line" | awk -F'[=/]' '{print $2}')
    diff=$(echo "$acc_line" | awk -F'[=/]' '{print $4}')
    total_correct=$(( total_correct + correct ))
    total_diff=$(( total_diff + diff ))

    cat tmp_$b.log >> full_run.log
    rm tmp_$b.log
done

echo "===================================" >> $LOG
echo "Total correct: $total_correct / 1000" >> $LOG
echo "Total diff: $total_diff / 1000" >> $LOG
acc=$(awk "BEGIN {printf \"%.2f\", ($total_correct/1000)*100}")
diffp=$(awk "BEGIN {printf \"%.2f\", ($total_diff/1000)*100}")
echo "Total accuracy: $acc%" >> $LOG
echo "Total diff rate: $diffp%" >> $LOG

