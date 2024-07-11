if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=SegRNN

root_path_name=./dataset/
data_path_name=growth_gunwi.csv
model_id_name=custom
data_name=custom
target_name=petiole

seq_len=3
for pred_len in 3
do
    python -u run_longExp.py \
      --is_training 100 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len'_'$target_name \
      --model $model_name \
      --data $data_name \
      --features MS \
      --target 'petiole' \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --seg_len 1 \
      --enc_in 13 \
      --d_model 516 \
      --dropout 0.1 \
      --train_epochs 100 \
      --patience 10 \
      --rnn_type gru \
      --dec_way pmf \
      --channel_id 1 \
      --use_gpu True \
      --itr 3 --batch_size 1 --learning_rate 0.0001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
